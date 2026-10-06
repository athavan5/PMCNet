import numpy as np
import os
import sys
import argparse
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from sklearn.metrics import *
from utils import *
from tensorflow.keras.models import load_model
from efficientnet import *
import tensorflow as tf
from tensorflow.keras import backend as K
import time, datetime


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description="Test PMCNet")
    parser.add_argument("--ckpt_epoch", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, required=True)
    parser.add_argument("--dataset-type", dest="dataset_type", type=str, default="IDRiD", choices=["DDR", "IDRiD", "IDRiD_w_PBDA", "MAPLES-DR", "epoch"])
    parser.add_argument("--save-dir", dest="save_dir", type=str, default=None)
    parser.add_argument("--use-wandb", dest="use_wandb", type=int, default=1, choices=[0, 1])
    return parser.parse_args()

def get_dataset_config(dataset_type, dataset_name):
    if dataset_type == 'DDR':
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == 'IDRiD':
        return 960, 1440, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == 'IDRiD_w_PBDA':
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == 'MAPLES-DR':
        return 1024, 1024, f"./data/IDRiD_MAPLES_combined/test_maples/images/", f"./data/IDRiD_MAPLES_combined/test_maples/masks/"
    if dataset_type == 'epoch':
        return 960, 1440, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    raise ValueError(f"Unsupported dataset: {dataset_type}")

args = parse_args()
dataset_type = args.dataset_type
dataset_name = args.dataset_name
h, w, image_dir, label_dir = get_dataset_config(dataset_type, dataset_name)

save_dir = f'./result/{dataset_name}/epoch_{args.ckpt_epoch}/test_maples'
predictions_dir = os.path.join(save_dir, "predictions")
os.makedirs(predictions_dir, exist_ok=True)
log_path = os.path.join(save_dir, "test_output.txt")

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


log_file = open(log_path, "w")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = TeeStream(original_stdout, log_file)
sys.stderr = TeeStream(original_stderr, log_file)

def run_test(wandb_run=None):
    images, image_names, labels = get_full_test_data(image_dir, label_dir, h, w)
    new_labels = make_label(labels)

    model_name = './weights/' + dataset_name + '/weights/PMCNet_' + dataset_name + f'_{args.ckpt_epoch}.h5'

    print(f"[INFO] Loading model from: {model_name}\n")
    model = load_model(model_name, compile=False)
    probs = model.predict(images, batch_size=1, verbose=1)
    save_results(images, image_names, probs, labels, predictions_dir, h, w)

    # IDRiD label ids: 1=EX, 2=HE, 3=MA, 4=SE (matches idrid_from_athavan / idrid_from_raw)
    EX = PR_AUC(probs[:, :, :, 1], new_labels[:, :, :, 1])
    HE = PR_AUC(probs[:, :, :, 2], new_labels[:, :, :, 2])
    MA = PR_AUC(probs[:, :, :, 3], new_labels[:, :, :, 3])
    SE = PR_AUC(probs[:, :, :, 4], new_labels[:, :, :, 4])

# Optional: show if any class has no positives (explains 0.0 or former NaN)
    for name, lab in [
        ("EX", new_labels[:,:,:,1]),
        ("HE", new_labels[:,:,:,2]),
        ("MA", new_labels[:,:,:,3]),
        ("SE", new_labels[:,:,:,4])]:
            n_pos = np.sum(lab)
            if n_pos == 0:
                print(f"[INFO] No positive pixels for class {name} in test set — PR-AUC set to 0.0")

    print("AUPR Results:")
    print("EX: " + str(EX))
    print("HE: " + str(HE))
    print("MA: " + str(MA))
    print("SE: " + str(SE))
    mean_aupr = np.nanmean([MA, HE, EX, SE])
    print("Mean AUPR: " + str(mean_aupr))

    predictions = np.argmax(probs, axis=-1)

    pred = make_label(predictions)   
    evaluator_MA = Evaluator()
    evaluator_HE = Evaluator()
    evaluator_EX = Evaluator()
    evaluator_SE = Evaluator()

    num = len(pred)
    for i in range(num):
        evaluator_EX.update(pred[i,:,:,1], new_labels[i,:,:,1])
        evaluator_HE.update(pred[i,:,:,2], new_labels[i,:,:,2])
        evaluator_MA.update(pred[i,:,:,3], new_labels[i,:,:,3])
        evaluator_SE.update(pred[i,:,:,4], new_labels[i,:,:,4])

    ma_dice, ma_iou = evaluator_MA.show()
    he_dice, he_iou = evaluator_HE.show()
    ex_dice, ex_iou = evaluator_EX.show()
    se_dice, se_iou = evaluator_SE.show()
    mean_dice = (ma_dice + he_dice + ex_dice + se_dice)/4
    mean_iou = (ma_iou + he_iou + ex_iou + se_iou)/4
    print('mean_dice:', mean_dice, ' mean_iou: ', mean_iou)
    print('\n')

    if wandb_run is not None:
        wandb_run.log(
            {
                "test/aupr_ma": float(MA),
                "test/aupr_he": float(HE),
                "test/aupr_ex": float(EX),
                "test/aupr_se": float(SE),
                "test/aupr_mean": float(mean_aupr),
                "test/dice_ma": float(ma_dice),
                "test/dice_he": float(he_dice),
                "test/dice_ex": float(ex_dice),
                "test/dice_se": float(se_dice),
                "test/dice_mean": float(mean_dice),
                "test/iou_ma": float(ma_iou),
                "test/iou_he": float(he_iou),
                "test/iou_ex": float(ex_iou),
                "test/iou_se": float(se_iou),
                "test/iou_mean": float(mean_iou),
            }
        )

try:
    if args.use_wandb == 1:
        import wandb

        with wandb.init(
            project="PMCNet",
            mode="offline",
            name=f"test-{dataset_name}-idrid-epoch-{args.ckpt_epoch}",
            config={
                "dataset": dataset_name,
                "dataset_type": dataset_type,
                "height": h,
                "width": w,
                "model_name": f"PMCNet_{dataset_name}_{args.ckpt_epoch}.h5",
                "ckpt_epoch": args.ckpt_epoch,
                "workers": args.workers
            },
        ) as run:
            run_test(wandb_run=run)
    else:
        run_test(wandb_run=None)
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()

