import argparse
import csv
import os
import re

import numpy as np
from tensorflow.keras.models import load_model

from efficientnet import *  # noqa: F401,F403 (needed for model loading)
from utils import PR_AUC, Evaluator, get_full_test_data, make_label

# Lesion class names and their one-hot channel indices (bg=0 is skipped)
LESION_NAMES   = ["EX", "MA", "SE", "HE"]
LESION_INDICES = [1, 2, 3, 4]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoint epochs and save per-class AUPR / Dice / IoU CSV."
    )
    parser.add_argument(
        "--dataset-name", dest="dataset_name", type=str, required=True,
        help="Dataset name used in checkpoint filenames (e.g. IDRiD_MAPLES_combined).",
    )
    parser.add_argument(
        "--dataset-type", dest="dataset_type", type=str, default="IDRiD",
        choices=["DDR", "IDRiD", "IDRiD_w_PBDA", "MAPLES-DR", "epoch"],
    )
    parser.add_argument(
        "--weights-dir", dest="weights_dir", type=str, default=None,
        help="Directory containing .h5 checkpoints. Defaults to ./weights/<dataset>/weights/",
    )
    parser.add_argument(
        "--output-csv", dest="output_csv", type=str, default=None,
        help="Output CSV path. Defaults to ./result/<dataset>/all_epochs_aupr.csv",
    )
    parser.add_argument(
        "--test-dir", dest="test_dir", type=str, default=None,
        help="Optional test set root directory containing images/ and masks/.",
    )
    parser.add_argument(
        "--dice-threshold", dest="dice_threshold", type=float, default=0.5,
        help="Probability threshold for converting soft predictions to binary "
             "masks when computing Dice and IoU. Default: 0.5",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1)
    return parser.parse_args()


def get_dataset_config(dataset_type, dataset_name):
    if dataset_type == "DDR":
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "IDRiD":
        return 960, 1440, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "IDRiD_w_PBDA":
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "MAPLES-DR":
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "epoch":
        return 960, 1440, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def parse_epoch_from_filename(filename):
    match = re.search(r"_(\d+)\.h5$", filename)
    return int(match.group(1)) if match else None


def list_checkpoints(weights_dir, dataset_name):
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory does not exist: {weights_dir}")
    prefix = f"PMCNet_{dataset_name}_"
    ckpts  = []
    for name in os.listdir(weights_dir):
        if not (name.startswith(prefix) and name.endswith(".h5")):
            continue
        epoch = parse_epoch_from_filename(name)
        if epoch is None:
            continue
        ckpts.append((epoch, os.path.join(weights_dir, name)))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found in {weights_dir} matching {prefix}<epoch>.h5"
        )
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def evaluate_epoch(model, images, new_labels, raw_labels, batch_size, dice_threshold):
    """
    Run inference and compute AUPR, Dice, and IoU for all 4 lesion classes.

    AUPR  — computed on raw softmax probabilities (soft metric).
    Dice / IoU — computed using the global pixel-count accumulation from
                 metrics.py (intersect_and_union approach), applied to hard
                 binary predictions obtained by thresholding each class
                 channel at `dice_threshold`.

    Parameters
    ----------
    model          : loaded Keras model
    images         : np.ndarray  (N, H, W, 3)
    new_labels     : np.ndarray  (N, H, W, 5)  — one-hot ground truth
    raw_labels     : np.ndarray  (N, H, W)     — integer ground truth (for argmax path)
    batch_size     : int
    dice_threshold : float

    Returns
    -------
    dict with keys: EX_aupr, MA_aupr, SE_aupr, HE_aupr, Mean_AUPR,
                    EX_dice, MA_dice, SE_dice, HE_dice, Mean_Dice,
                    EX_iou,  MA_iou,  SE_iou,  HE_iou,  Mean_IoU
    """
    probs = model.predict(images, batch_size=batch_size, verbose=0)
    # probs shape: (N, H, W, 5)

    results = {}

    # ---- AUPR (soft, per class) -------------------------------------------
    aupr_vals = []
    for name, ci in zip(LESION_NAMES, LESION_INDICES):
        #val = PR_AUC(probs[:, :, :, ci], new_labels[:, :, ci]) <- old line
        val = PR_AUC(probs[:, :, :, ci], new_labels[:, :, :, ci])
        results[f"{name}_aupr"] = float(val)
        aupr_vals.append(val)
    results["Mean_AUPR"] = float(np.nanmean(aupr_vals))

    # ---- Dice & IoU (hard, global accumulation — mirrors metrics.py) ------
    # For each lesion class, threshold the predicted probability map to get a
    # binary prediction, then feed both prediction and ground-truth into an
    # Evaluator that accumulates pixel counts across all N images before
    # computing the final score.  This matches the
    #   2 * total_intersect / (total_pred + total_label)
    # formula used in metrics.py → total_area_to_metrics.
    dice_vals = []
    iou_vals  = []
    for name, ci in zip(LESION_NAMES, LESION_INDICES):
        ev = Evaluator()
        for i in range(len(images)):
            pred_bin = (probs[i, :, :, ci] >= dice_threshold).astype(np.float32)
            gt_bin   = new_labels[i, :, :, ci]
            ev.update(pred_bin, gt_bin)
        dice, iou = ev.show()
        results[f"{name}_dice"] = dice
        results[f"{name}_iou"]  = iou
        dice_vals.append(dice)
        iou_vals.append(iou)

    results["Mean_Dice"] = round(float(np.mean(dice_vals)), 2)
    results["Mean_IoU"]  = round(float(np.mean(iou_vals)),  2)

    return results


def main():
    args = parse_args()
    h, w, image_dir, label_dir = get_dataset_config(args.dataset_type, args.dataset_name)

    if args.test_dir is not None:
        image_dir = os.path.join(args.test_dir, "images") + "/"
        label_dir = os.path.join(args.test_dir, "masks")  + "/"

    weights_dir = args.weights_dir or f"./weights/{args.dataset_name}/weights/"
    output_csv  = (
        args.output_csv
        or f"./result/{args.dataset_name}/{label_dir.split('/')[-3]}/all_epochs_aupr_dice_iou.csv"
    )
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"[INFO] Loading test set from: {image_dir}")
    images, _, raw_labels = get_full_test_data(image_dir, label_dir, h, w)
    new_labels = make_label(raw_labels)  # (N, H, W, 5) one-hot

    checkpoints = list_checkpoints(weights_dir, args.dataset_name)
    print(f"[INFO] Found {len(checkpoints)} checkpoints in: {weights_dir}")
    print(f"[INFO] Dice/IoU threshold: {args.dice_threshold}")

    # CSV columns
    fieldnames = [
        "epoch",
        "EX_aupr",  "MA_aupr",  "SE_aupr",  "HE_aupr",  "Mean_AUPR",
        "EX_dice",  "MA_dice",  "SE_dice",  "HE_dice",  "Mean_Dice",
        "EX_iou",   "MA_iou",   "SE_iou",   "HE_iou",   "Mean_IoU",
    ]

    rows = []
    for epoch, ckpt_path in checkpoints:
        print(f"[INFO] Evaluating epoch {epoch}: {ckpt_path}")
        model   = load_model(ckpt_path, compile=False)
        metrics = evaluate_epoch(
            model, images, new_labels, raw_labels,
            args.batch_size, args.dice_threshold
        )
        row = {"epoch": epoch, **metrics}
        rows.append(row)

        print(
            f"[INFO] AUPR  — "
            f"EX: {metrics['EX_aupr']:.4f}, MA: {metrics['MA_aupr']:.4f}, "
            f"SE: {metrics['SE_aupr']:.4f}, HE: {metrics['HE_aupr']:.4f}, "
            f"Mean: {metrics['Mean_AUPR']:.4f}"
        )
        print(
            f"[INFO] Dice  — "
            f"EX: {metrics['EX_dice']:.2f}%, MA: {metrics['MA_dice']:.2f}%, "
            f"SE: {metrics['SE_dice']:.2f}%, HE: {metrics['HE_dice']:.2f}%, "
            f"Mean: {metrics['Mean_Dice']:.2f}%"
        )
        print(
            f"[INFO] IoU   — "
            f"EX: {metrics['EX_iou']:.2f}%, MA: {metrics['MA_iou']:.2f}%, "
            f"SE: {metrics['SE_iou']:.2f}%, HE: {metrics['HE_iou']:.2f}%, "
            f"Mean: {metrics['Mean_IoU']:.2f}%"
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved per-epoch metrics CSV to: {output_csv}")


if __name__ == "__main__":
    main()
