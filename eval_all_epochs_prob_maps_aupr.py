import argparse
import csv
import os
import re

import numpy as np

from eval_fused_prob_maps import align_to_target_hw, load_prob_npy, parse_order, renormalize_probs, reorder_channels
from metrics import eval_metrics
from utils import PR_AUC, get_full_test_data, make_label

CLASS_NAMES = ["bg", "ex", "ma", "se", "he"]
NUM_CLASSES = len(CLASS_NAMES)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AUPR for all epoch subfolders under a probability-map root. "
            "Each epoch folder must contain {basename}_probs.npy files."
        )
    )
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, required=True)
    parser.add_argument(
        "--dataset-type",
        dest="dataset_type",
        type=str,
        default="IDRiD",
        choices=["DDR", "IDRiD", "IDRiD_w_PBDA", "MAPLES-DR", "epoch"],
    )
    parser.add_argument(
        "--split",
        dest="split",
        type=str,
        default="test",
        choices=["test", "val", "validation"],
        help="Which dataset split to evaluate against.",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        type=str,
        default=None,
        help="Override split root containing images/ and masks/.",
    )
    parser.add_argument(
        "--prob-root",
        dest="prob_root",
        type=str,
        required=True,
        help=(
            "Root directory containing one subfolder per epoch. "
            "NPY files may live directly in each subfolder or under a nested "
            "'prob_maps/' directory (DSR-U-Net get-prob-maps layout); see --prob-subdir."
        ),
    )
    parser.add_argument(
        "--prob-subdir",
        dest="prob_subdir",
        type=str,
        default=None,
        help=(
            "If set, load *_probs.npy from <epoch_dir>/<this> (e.g. prob_maps). "
            "If omitted, uses <epoch_dir>/prob_maps when that directory exists, "
            "otherwise <epoch_dir>."
        ),
    )
    parser.add_argument(
        "--order",
        dest="order",
        type=str,
        default="bg,ex,ma,se,he",
        help="Comma-separated on-disk channel order for loaded prob maps.",
    )
    parser.add_argument(
        "--output-csv",
        dest="output_csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to <prob-root>/all_epochs_aupr_<split>.csv",
    )
    parser.add_argument(
        "--include-seg-metrics",
        dest="include_seg_metrics",
        action="store_true",
        help=(
            "Also compute multiclass Dice and IoU via metrics.eval_metrics "
            "(argmax probabilities vs GT labels; histogram IoU/Dice) and append columns to the CSV."
        ),
    )
    parser.add_argument(
        "--ignore-index",
        dest="ignore_index",
        type=int,
        default=255,
        help="Label value ignored in Dice/IoU when --include-seg-metrics is set.",
    )
    return parser.parse_args()


def get_dataset_config(dataset_type, dataset_name):
    if dataset_type == "DDR":
        return 1024, 1024
    if dataset_type == "IDRiD":
        return 960, 1440
    if dataset_type == "IDRiD_w_PBDA":
        return 1024, 1024
    if dataset_type == "MAPLES-DR":
        return 1024, 1024
    if dataset_type == "epoch":
        return 960, 1440
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def normalize_split(split):
    if split == "validation":
        return "val"
    return split


def parse_epoch(folder_name):
    """
    Training epoch index from folder name.

    DSR-U-Net layout uses a trailing ``-ep<N>`` (e.g. ``...-eyepacs_1440x960-ep12`` → 12).
    We must not use the first digit run (e.g. 1440 in ``1440x960``).
    """
    m = re.search(r"-ep(\d+)\s*$", folder_name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    found = list(re.finditer(r"-ep(\d+)", folder_name, flags=re.IGNORECASE))
    if found:
        return int(found[-1].group(1))
    # Older layouts: e.g. epoch_12, or any digit run
    m2 = re.search(r"epoch_(\d+)", folder_name, flags=re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    m3 = re.search(r"(\d+)", folder_name)
    return int(m3.group(1)) if m3 else None


def list_epoch_dirs(prob_root):
    if not os.path.isdir(prob_root):
        raise FileNotFoundError(f"Probability-map root does not exist: {prob_root}")

    entries = []
    for name in os.listdir(prob_root):
        path = os.path.join(prob_root, name)
        if not os.path.isdir(path):
            continue
        epoch = parse_epoch(name)
        if epoch is None:
            continue
        entries.append((epoch, name, path))

    if not entries:
        raise FileNotFoundError(f"No epoch subfolders found in: {prob_root}")

    entries.sort(key=lambda x: (x[0], x[1]))
    return entries


def resolve_epoch_prob_dir(epoch_dir, prob_subdir=None):
    """
    Directory that actually contains {basename}_probs.npy for one epoch.

    DSR-U-Net often writes: <epoch_run>/prob_maps/*.npy
    """
    if prob_subdir:
        p = os.path.join(epoch_dir, prob_subdir)
        if not os.path.isdir(p):
            raise FileNotFoundError(
                f"Expected probability maps under {p} (--prob-subdir={prob_subdir!r})"
            )
        return p
    nested = os.path.join(epoch_dir, "prob_maps")
    if os.path.isdir(nested):
        return nested
    return epoch_dir


def load_epoch_probs(epoch_dir, image_names, h, w, order):
    stack = []
    for image_name in image_names:
        base = os.path.splitext(os.path.basename(image_name))[0]
        path = os.path.join(epoch_dir, f"{base}_probs.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing prob map for image '{base}': {path}")
        prob = load_prob_npy(path)
        prob = reorder_channels(prob, order, CLASS_NAMES)
        prob = align_to_target_hw(prob, h, w, base)
        prob = renormalize_probs(np.clip(prob, 0.0, 1.0))
        stack.append(prob)
    return np.stack(stack, axis=0)


def compute_seg_metrics_mmseg(probs, labels_int, ignore_index):
    """Dice / IoU from metrics.py (eval_metrics, mDice branch includes IoU)."""
    n = probs.shape[0]
    pred_cls = np.argmax(probs, axis=-1).astype(np.int64)
    gt_maps = [labels_int[i].astype(np.int64) for i in range(n)]
    pred_maps = [pred_cls[i] for i in range(n)]
    ret = eval_metrics(
        results=pred_maps,
        gt_seg_maps=gt_maps,
        num_classes=NUM_CLASSES,
        ignore_index=ignore_index,
        metrics=["mDice"],
    )
    iou = ret["IoU"]
    dice = ret["Dice"]
    row = {
        "Dice_EX": float(dice[1]),
        "Dice_MA": float(dice[2]),
        "Dice_SE": float(dice[3]),
        "Dice_HE": float(dice[4]),
        "IoU_EX": float(iou[1]),
        "IoU_MA": float(iou[2]),
        "IoU_SE": float(iou[3]),
        "IoU_HE": float(iou[4]),
        "Mean_Dice": float(np.nanmean(dice[1:5])),
        "Mean_IoU": float(np.nanmean(iou[1:5])),
        "aAcc": float(ret["aAcc"]),
    }
    return row


def main():
    args = parse_args()
    split = normalize_split(args.split)
    h, w = get_dataset_config(args.dataset_type, args.dataset_name)
    order = parse_order(args.order)

    if args.data_dir is not None:
        image_dir = os.path.join(args.data_dir, "images")
        label_dir = os.path.join(args.data_dir, "masks")
    else:
        image_dir = f"./data/{args.dataset_name}/{split}/images"
        label_dir = f"./data/{args.dataset_name}/{split}/masks"

    image_dir = os.path.join(image_dir, "")
    label_dir = os.path.join(label_dir, "")

    output_csv = args.output_csv or os.path.join(args.prob_root, f"all_epochs_aupr_{split}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"[INFO] split={split}")
    print(f"[INFO] image_dir={image_dir}")
    print(f"[INFO] label_dir={label_dir}")
    print(f"[INFO] prob_root={args.prob_root}")
    print(f"[INFO] include_seg_metrics={args.include_seg_metrics}")

    _, image_names, labels = get_full_test_data(image_dir, label_dir, h, w)
    new_labels = make_label(labels)

    epoch_dirs = list_epoch_dirs(args.prob_root)
    print(f"[INFO] Found {len(epoch_dirs)} epoch folders.")

    rows = []
    for epoch, folder_name, epoch_dir in epoch_dirs:
        prob_dir = resolve_epoch_prob_dir(epoch_dir, args.prob_subdir)
        print(f"[INFO] Evaluating epoch {epoch} ({folder_name}) prob_dir={prob_dir}")
        probs = load_epoch_probs(prob_dir, image_names, h, w, order)

        ex = PR_AUC(probs[:, :, :, 1], new_labels[:, :, :, 1])
        ma = PR_AUC(probs[:, :, :, 2], new_labels[:, :, :, 2])
        se = PR_AUC(probs[:, :, :, 3], new_labels[:, :, :, 3])
        he = PR_AUC(probs[:, :, :, 4], new_labels[:, :, :, 4])
        mean_aupr = float(np.nanmean([ex, ma, se, he]))

        row = {
            "epoch": epoch,
            "folder": folder_name,
            "EX": float(ex),
            "MA": float(ma),
            "SE": float(se),
            "HE": float(he),
            "Mean_AUPR": mean_aupr,
        }
        if args.include_seg_metrics:
            row.update(compute_seg_metrics_mmseg(probs, labels, args.ignore_index))

        rows.append(row)

        print(
            "[INFO] AUPR - "
            f"EX: {ex:.6f}, MA: {ma:.6f}, SE: {se:.6f}, HE: {he:.6f}, Mean: {mean_aupr:.6f}"
        )
        if args.include_seg_metrics:
            print(
                "[INFO] Seg (metrics.py) - "
                f"Mean_Dice: {row['Mean_Dice']:.4f}, Mean_IoU: {row['Mean_IoU']:.4f}, "
                f"aAcc: {row['aAcc']:.4f}"
            )

    base_fields = ["epoch", "folder", "EX", "MA", "SE", "HE", "Mean_AUPR"]
    seg_fields = [
        "Dice_EX",
        "Dice_MA",
        "Dice_SE",
        "Dice_HE",
        "IoU_EX",
        "IoU_MA",
        "IoU_SE",
        "IoU_HE",
        "Mean_Dice",
        "Mean_IoU",
        "aAcc",
    ]
    fieldnames = base_fields + (seg_fields if args.include_seg_metrics else [])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved per-epoch CSV to: {output_csv}")


if __name__ == "__main__":
    main()
