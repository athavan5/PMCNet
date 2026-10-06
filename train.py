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

# One-hot channel index / mask label id: 1=EX, 2=MA, 3=SE, 4=HE
# (matches LESION_NAMES / LESION_INDICES in test_pmcnet.py)
VAL_AUPR_LESION_NAMES = ("EX", "MA", "SE", "HE")

def multi_class(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)   

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
    parser.add_argument(
        "--dice-threshold", dest="dice_threshold", type=float, default=0.5,
        help="Probability threshold for converting soft predictions to binary "
             "masks when computing validation Dice and IoU (same as test_pmcnet.py). Default: 0.5",
    )
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

# Dice, IoU and AUPR are computed on the validation set only, at epoch end,
# by ValMetricsCallback below (same method as test_pmcnet.py).
metrics = ['accuracy']

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

class ValMetricsCallback(Callback):
    """
    Compute AUPR, Dice and IoU per lesion class on the validation set at epoch end,
    using the same method as test_pmcnet.py.

    AUPR : soft probabilities, pooled over the whole validation set, one PR-AUC per class.
    Dice / IoU : each lesion channel is thresholded independently at `dice_threshold`
        (no argmax, background excluded). The Evaluator accumulates intersection /
        union / pred-area / label-area counts across ALL validation images and divides
        once at the end. Values are percentages rounded to 2 d.p.; the means are a
        plain mean over the lesion classes, rounded to 2 d.p.

    Logged keys use lesion names: val_aupr_EX, val_dice_EX, val_iou_EX, ...
    plus val_mean_aupr_per_class, val_mean_dice, val_mean_iou.
    """
    def __init__(self, val_gen_factory, val_steps, num_classes, dice_threshold=0.5):
        super().__init__()
        self.val_gen_factory = val_gen_factory
        self.val_steps = int(val_steps)
        self.num_classes = int(num_classes)
        self.dice_threshold = float(dice_threshold)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_gen = self.val_gen_factory()
        # AUPR needs all predictions pooled, so accumulate them per class.
        pred_list = [[] for _ in range(self.num_classes)]
        true_list = [[] for _ in range(self.num_classes)]
        # Dice / IoU only need pixel counts, so update one Evaluator per class
        # image by image, exactly as the test script does.
        evaluators = {c: Evaluator() for c in range(1, self.num_classes)}

        for _ in range(self.val_steps):
            x_batch, y_true = next(val_gen)
            y_pred = self.model.predict_on_batch(x_batch)
            y_pred = np.asarray(y_pred)
            y_true = np.asarray(y_true)

            for c in range(1, self.num_classes):  # skip background=0
                pred_list[c].append(y_pred[..., c].reshape(-1))
                true_list[c].append(y_true[..., c].reshape(-1))

                for i in range(y_pred.shape[0]):
                    pred_bin = (y_pred[i, :, :, c] >= self.dice_threshold).astype(np.float32)
                    gt_bin = y_true[i, :, :, c]
                    evaluators[c].update(pred_bin, gt_bin)

        aupr_vals, dice_vals, iou_vals = [], [], []
        for c in range(1, self.num_classes):
            name = VAL_AUPR_LESION_NAMES[c - 1]

            # ---- AUPR (soft) ----
            if len(pred_list[c]) == 0:
                aupr = 0.0
            else:
                pred_all = np.concatenate(pred_list[c], axis=0)
                true_all = np.concatenate(true_list[c], axis=0)
                aupr = float(PR_AUC(pred_all, true_all))
            logs[f"val_aupr_{name}"] = aupr
            aupr_vals.append(aupr)

            # ---- Dice & IoU (hard, global accumulation) ----
            dice, iou = evaluators[c].show()
            logs[f"val_dice_{name}"] = float(dice)
            logs[f"val_iou_{name}"] = float(iou)
            dice_vals.append(dice)
            iou_vals.append(iou)

        logs["val_mean_aupr_per_class"] = float(np.mean(aupr_vals))
        logs["val_mean_dice"] = round(float(np.mean(dice_vals)), 2)
        logs["val_mean_iou"] = round(float(np.mean(iou_vals)), 2)


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
val_metrics_cb = ValMetricsCallback(val_gen_factory, val_steps, NUM_CLASSES, dice_threshold=args.dice_threshold)
csv_logger = OneIndexedCSVLogger(filename=f"{output_dir}/training.csv")

# val_metrics_cb must come before csv_logger so its keys are in `logs` when the CSV row is written
callbacks = [checkpointer, val_metrics_cb, csv_logger]
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
            "dice_threshold": args.dice_threshold,
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

