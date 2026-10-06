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
    parser.add_argument(
        "--test-dir",
        dest="test_dir",
        type=str,
        default=None,
        help="Override test data root: uses <test-dir>/images/ and <test-dir>/masks/. "
        "If omitted, image and mask paths come from get_dataset_config(dataset_type, dataset_name).",
    )
    parser.add_argument(
        "--dice-threshold", dest="dice_threshold", type=float, default=0.5,
        help="Probability threshold for converting soft predictions to binary "
             "masks when computing Dice and IoU. Default: 0.5",
    )
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
h, w, default_image_dir, default_label_dir = get_dataset_config(dataset_type, dataset_name)
if args.test_dir is not None:
    test_root = os.path.normpath(args.test_dir)
    image_dir = os.path.join(test_root, "images/")
    label_dir = os.path.join(test_root, "masks/")
else:
    image_dir, label_dir = default_image_dir, default_label_dir

save_dir = f'./result/{dataset_name}/epoch_{args.ckpt_epoch}/{label_dir.split("/")[-4]}_test'
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

# Lesion class names and their one-hot channel indices
LESION_NAMES   = ["EX", "MA", "SE", "HE"]
LESION_INDICES = [1, 2, 3, 4]

def run_test(wandb_run=None):
    images, image_names, labels = get_full_test_data(image_dir, label_dir, h, w)
    new_labels = make_label(labels)

    model_name = './weights/' + dataset_name + '/weights/PMCNet_' + dataset_name + f'_{args.ckpt_epoch}.h5'

    print(f"[INFO] Loading model from: {model_name}\n")
    print(f"[INFO] Dice/IoU threshold: {args.dice_threshold}\n")
    model = load_model(model_name, compile=False)
    probs = model.predict(images, batch_size=1, verbose=1)
    save_results(images, image_names, probs, labels, predictions_dir, h, w)

    # ---- AUPR (soft, per class) -------------------------------------------
    EX = PR_AUC(probs[:, :, :, 1], new_labels[:, :, :, 1])
    MA = PR_AUC(probs[:, :, :, 2], new_labels[:, :, :, 2])
    SE = PR_AUC(probs[:, :, :, 3], new_labels[:, :, :, 3])
    HE = PR_AUC(probs[:, :, :, 4], new_labels[:, :, :, 4])

    # Optional: warn if any class has no positives
    for name, lab in [
        ("EX", new_labels[:,:,:,1]),
        ("MA", new_labels[:,:,:,2]),
        ("SE", new_labels[:,:,:,3]),
        ("HE", new_labels[:,:,:,4])]:
            n_pos = np.sum(lab)
            if n_pos == 0:
                print(f"[INFO] No positive pixels for class {name} in test set — PR-AUC set to 0.0")

    print("AUPR Results:")
    print("EX: " + str(EX))
    print("MA: " + str(MA))
    print("SE: " + str(SE))
    print("HE: " + str(HE))
    mean_aupr = np.nanmean([EX, MA, SE, HE])
    print("Mean AUPR: " + str(mean_aupr))

    # ---- Dice & IoU (hard, global accumulation) ---------------------------
    # Mirrors test_all_epochs_aupr_updated.py: threshold each class channel
    # independently at dice_threshold rather than using argmax across classes.
    # The Evaluator accumulates pixel counts across all images before dividing,
    # matching the intersect_and_union approach from metrics.py.
    dice_vals = []
    iou_vals  = []
    evaluators = {}
    for name, ci in zip(LESION_NAMES, LESION_INDICES):
        ev = Evaluator()
        for i in range(len(images)):
            pred_bin = (probs[i, :, :, ci] >= args.dice_threshold).astype(np.float32)
            gt_bin   = new_labels[i, :, :, ci]
            ev.update(pred_bin, gt_bin)
        evaluators[name] = ev
        dice, iou = ev.show()
        dice_vals.append(dice)
        iou_vals.append(iou)

    ex_dice, ex_iou = evaluators["EX"].show()
    ma_dice, ma_iou = evaluators["MA"].show()
    se_dice, se_iou = evaluators["SE"].show()
    he_dice, he_iou = evaluators["HE"].show()

    mean_dice = round(float(np.mean(dice_vals)), 2)
    mean_iou  = round(float(np.mean(iou_vals)),  2)

    print(f"\nDICE Results (per class, threshold={args.dice_threshold} vs GT):")
    print("EX:", ex_dice)
    print("MA:", ma_dice)
    print("SE:", se_dice)
    print("HE:", he_dice)
    print(f"\nIoU Results (per class, threshold={args.dice_threshold} vs GT):")
    print("EX:", ex_iou)
    print("MA:", ma_iou)
    print("SE:", se_iou)
    print("HE:", he_iou)
    print('mean_dice:', mean_dice, ' mean_iou: ', mean_iou)
    print('\n')

    if wandb_run is not None:
        wandb_run.log(
            {
                "test/aupr_ex": float(EX),
                "test/aupr_ma": float(MA),
                "test/aupr_se": float(SE),
                "test/aupr_he": float(HE),
                "test/aupr_mean": float(mean_aupr),
                "test/dice_ex": float(ex_dice),
                "test/dice_ma": float(ma_dice),
                "test/dice_se": float(se_dice),
                "test/dice_he": float(he_dice),
                "test/dice_mean": float(mean_dice),
                "test/iou_ex": float(ex_iou),
                "test/iou_ma": float(ma_iou),
                "test/iou_se": float(se_iou),
                "test/iou_he": float(he_iou),
                "test/iou_mean": float(mean_iou),
            }
        )

try:
    if args.use_wandb == 1:
        import wandb

        with wandb.init(
            project="PMCNet",
            mode="offline",
            name=f"test-{dataset_name}-{dataset_type}-epoch-{args.ckpt_epoch}-on-{args.test_dir.split('/')[-4]}-{args.test_dir.split('/')[-3]}",
            config={
                "dataset": dataset_name,
                "dataset_type": dataset_type,
                "height": h,
                "width": w,
                "model_name": f"PMCNet_{dataset_name}_{args.ckpt_epoch}.h5",
                "ckpt_epoch": args.ckpt_epoch,
                "workers": args.workers,
                "test_dir_override": args.test_dir,
                "image_dir": image_dir,
                "label_dir": label_dir,
                "dice_threshold": args.dice_threshold,
            },
        ) as run:
            run_test(wandb_run=run)
    else:
        run_test(wandb_run=None)
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
