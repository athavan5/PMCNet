import os
import cv2
import random
import math
import argparse
import numpy as np 
import tensorflow as tf
from keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras.metrics import *
from models import *
from utils import *

NUM_CLASSES = 5  # background + 4 lesion classes

# One-hot channel index / mask label id for IDRiD-style data: 1=EX, 2=HE, 3=MA, 4=SE
# NOTE: OLD version: VAL_AUPR_LESION_NAMES = ("EX", "HE", "MA", "SE")
# NEW version: VAL_AUPR_LESION_NAMES = ("EX", "MA", "SE", "HE")
VAL_AUPR_LESION_NAMES = ("EX", "MA", "SE", "HE")

def multi_class(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)   

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def step_decay(epoch):
    initial_lrate = 0.0001
    epochs_drop = 60
    lrate = initial_lrate * math.pow(1 - (1 + epoch) / epochs_drop, 0.9)
    return lrate

def parse_args():
    parser = argparse.ArgumentParser(description="Train PMCNet")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, required=True)
    parser.add_argument("--dataset-type", dest="dataset_type", type=str, default="IDRiD_from_raw", choices=["DDR", "IDRiD", "IDRiD_w_PBDA", "MAPLES-DR", "epoch"])
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--use-wandb", dest="use_wandb", type=int, default=1, choices=[0, 1])
    return parser.parse_args()

def get_dataset_config(dataset_type, dataset_name):
    if dataset_type == 'DDR':
        return 1024, 1024, f"./data/{dataset_name}/train", f"./data/{dataset_name}/val"
    if dataset_type == 'IDRiD':
        return 960, 1440, f"./data/{dataset_name}/train", f"./data/{dataset_name}/val"
    if dataset_type == 'IDRiD_w_PBDA': # TODO: Check how sizing works for this dataset
        return 1024, 1024, f"./data/{dataset_name}/train", f"./data/{dataset_name}/val"
    if dataset_type == 'MAPLES-DR': # TODO: Check how sizing works for this dataset
        return 1024, 1024, f"./data/{dataset_name}/train", f"./data/{dataset_name}/val"
    if dataset_type == 'epoch':
        return 960, 1440, f"./data/{dataset_name}/train", f"./data/{dataset_name}/valid"
    raise ValueError(f"Unsupported dataset: {dataset_type}")

def get_optimizer(name, learning_rate):
    if name == "adam":
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    if name == "sgd":
        return tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    if name == "rmsprop":
        return tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
    raise ValueError(f"Unsupported optimizer: {name}")

args = parse_args()
dataset_type = args.dataset_type
dataset_name = args.dataset_name
image_h, image_w, train_path, valid_path = get_dataset_config(dataset_type, dataset_name)

batch_size = 1
train_num = len(os.listdir(train_path+'/images/'))
valid_num  = len(os.listdir(valid_path+'/images/'))

lr_decay = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
# include dataset name directly in the path, only let Keras format epoch
output_dir = f'./weights/{dataset_name}'
os.makedirs(output_dir, exist_ok=True)
weights_dir = os.path.join(output_dir, "weights")
os.makedirs(weights_dir, exist_ok=True)
checkpointer = ModelCheckpoint(
    filepath=os.path.join(weights_dir, 'PMCNet_' + dataset_name + '_{epoch:02d}.h5'),
    verbose=1,
    monitor='val_accuracy'
)

model = PMCNet(image_h, image_w, color_type=3, num_class=NUM_CLASSES) 

# metrics = [
#     'accuracy',
#     dice_coef,
#     iou_coef,
#     # AUPR is computed only on validation via ValAUPRCallback below.
# ]

metrics = [
    'accuracy',
    dice_coef,
    IoU(name='iou_coef', num_classes=NUM_CLASSES, target_class_ids=[1, 2, 3, 4])
]

# optimizer=keras.optimizers.get("adam", use_legacy_optimizer=True)
model.compile(
    optimizer=get_optimizer(args.optimizer, args.learning_rate),
    loss=multi_class,
    metrics=metrics,
)
                                                                                  
train_data = dataGenerator(
    batch_size=batch_size,
    target_size=(image_h, image_w),
    train_path=train_path,
    image_color_mode="rgb",
    mask_color_mode="grayscale"
)

valid_data = dataGenerator(
    batch_size=batch_size,
    target_size=(image_h, image_w),
    train_path=valid_path
)

class ValAUPRCallback(Callback):
    """
    Compute AUPR per lesion class on the validation set at epoch end.

    Logged keys use lesion names (val_aupr_EX, val_aupr_HE, …), matching mask ids 1–4.
    """
    def __init__(self, val_gen_factory, val_steps, num_classes):
        super().__init__()
        self.val_gen_factory = val_gen_factory
        self.val_steps = int(val_steps)
        self.num_classes = int(num_classes)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_gen = self.val_gen_factory()
        # Accumulate validation predictions/labels across all batches, then compute
        # one PR-AUC per class over the whole validation set (more typical than
        # averaging per-batch PR-AUC values).
        pred_list = [[] for _ in range(self.num_classes)]
        true_list = [[] for _ in range(self.num_classes)]

        for _ in range(self.val_steps):
            x_batch, y_true = next(val_gen)
            y_pred = self.model.predict_on_batch(x_batch)

            for c in range(1, self.num_classes):  # skip background=0
                pred_list[c].append(np.asarray(y_pred[..., c]).reshape(-1))
                true_list[c].append(np.asarray(y_true[..., c]).reshape(-1))

        for c in range(1, self.num_classes):
            name = VAL_AUPR_LESION_NAMES[c - 1]
            if len(pred_list[c]) == 0:
                logs[f"val_aupr_{name}"] = 0.0
                continue

            pred_all = np.concatenate(pred_list[c], axis=0)
            true_all = np.concatenate(true_list[c], axis=0)
            logs[f"val_aupr_{name}"] = float(PR_AUC(pred_all, true_all))

        logs["val_mean_aupr_per_class"] = float(
            np.mean([logs[f"val_aupr_{VAL_AUPR_LESION_NAMES[c - 1]}"] for c in range(1, self.num_classes)])
        )


class OneIndexedCSVLogger(CSVLogger):
    """
    Align CSV epoch numbers with Keras checkpoint naming (1-based).
    """
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch + 1, logs)


class WandbOneIndexedEpochLogger(Callback):
    """
    Log metrics to W&B using 1-based epoch indexing.
    """
    def __init__(self, wandb_module):
        super().__init__()
        self.wandb = wandb_module

    def on_epoch_end(self, epoch, logs=None):
        payload = dict(logs or {})
        payload["epoch"] = epoch + 1
        self.wandb.log(payload, step=epoch + 1)

val_steps = valid_num // batch_size
val_gen_factory = lambda: dataGenerator(
    batch_size=batch_size,
    target_size=(image_h, image_w),
    train_path=valid_path
)
val_aupr_cb = ValAUPRCallback(val_gen_factory, val_steps, NUM_CLASSES)
csv_logger = OneIndexedCSVLogger(filename=f"{output_dir}/training.csv")

callbacks = [checkpointer, val_aupr_cb, csv_logger]
if args.use_wandb == 1:
    import wandb

    with wandb.init(
        project="PMCNet",
        mode="offline",
        name=f"train-{dataset_name}",
        config={
            "dataset": dataset_name,
            "dataset_type": dataset_type,
            "train_path": train_path,
            "valid_path": valid_path,
            "image_h": image_h,
            "image_w": image_w,
            "batch_size": batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "model": "PMCNet",
            "num_classes": NUM_CLASSES,
            "workers": args.workers
        },
    ):
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        model.fit(
            train_data,
            steps_per_epoch=train_num // batch_size,
            epochs=args.epochs,
            validation_data=valid_data,
            workers=args.workers,
            validation_steps=valid_num // batch_size,
            callbacks=callbacks + [WandbOneIndexedEpochLogger(wandb)],
        )
else:
    model.fit(
        train_data,
        steps_per_epoch=train_num // batch_size,
        epochs=args.epochs,
        validation_data=valid_data,
        workers=args.workers,
        validation_steps=valid_num // batch_size,
        callbacks=callbacks,
    )

