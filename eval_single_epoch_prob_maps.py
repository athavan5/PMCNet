import argparse
import csv
import os

import numpy as np

from eval_fused_prob_maps import align_to_target_hw, load_prob_npy, parse_order, renormalize_probs, reorder_channels
from metrics import eval_metrics
from utils import PR_AUC, get_full_test_data, make_label

CLASS_NAMES = ["bg", "ex", "ma", "se", "he"]
NUM_CLASSES = len(CLASS_NAMES)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AUPR (and optionally Dice/IoU) for a single DSR-U-Net epoch "
            "whose probability maps live under --prob-root/--prob-subdir/. "
            "No epoch-folder scanning or parse_epoch logic is needed — "
            "PROB_ROOT points directly at the single run directory."
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
        "--data-dir",
        dest="data_dir",
        type=str,
        required=True,
        help="Root directory containing images/ and masks/ for the test split.",
    )
    parser.add_argument(
        "--prob-root",
        dest="prob_root",
        type=str,
        required=True,
        help=(
            "Single run directory, e.g. "
            ".../get-prob-maps-<dataset>-ep<N>-on-<test_dir>. "
            "The .npy files live directly here or under --prob-subdir."
        ),
    )
    parser.add_argument(
        "--prob-subdir",
        dest="prob_subdir",
        type=str,
        default=None,
        help=(
            "If set, load *_probs.npy from <prob-root>/<this> (e.g. prob_maps). "
            "If omitted, uses <prob-root>/prob_maps when that directory exists, "
            "otherwise <prob-root> directly."
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
        help="Output CSV path. Defaults to <prob-root>/metrics_test.csv",
    )
    parser.add_argument(
        "--include-seg-metrics",
        dest="include_seg_metrics",
        action="store_true",
        help=(
            "Also compute multiclass Dice and IoU via metrics.eval_metrics "
            "(argmax predictions vs GT labels) and append columns to the CSV."
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


def resolve_prob_dir(prob_root, prob_subdir=None):
    """
    Resolve the directory that actually contains {basename}_probs.npy files.

    Checks for a nested prob_maps/ subfolder automatically if --prob-subdir
    is not explicitly provided.
    """
    if prob_subdir:
        p = os.path.join(prob_root, prob_subdir)
        if not os.path.isdir(p):
            raise FileNotFoundError(
                f"Expected probability maps under {p} (--prob-subdir={prob_subdir!r})"
            )
        return p
    nested = os.path.join(prob_root, "prob_maps")
    if os.path.isdir(nested):
        return nested
    return prob_root


def load_probs(prob_dir, image_names, h, w, order):
    """Load and align all .npy probability maps for this epoch."""
    stack = []
    for image_name in image_names:
        base = os.path.splitext(os.path.basename(image_name))[0]
        path = os.path.join(prob_dir, f"{base}_probs.npy")
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
    pred_cls  = np.argmax(probs, axis=-1).astype(np.int64)
    gt_maps   = [labels_int[i].astype(np.int64) for i in range(n)]
    pred_maps = [pred_cls[i] for i in range(n)]
    ret = eval_metrics(
        results=pred_maps,
        gt_seg_maps=gt_maps,
        num_classes=NUM_CLASSES,
        ignore_index=ignore_index,
        metrics=["mDice"],
    )
    iou  = ret["IoU"]
    dice = ret["Dice"]
    return {
        "Dice_EX":   float(dice[1]),
        "Dice_MA":   float(dice[2]),
        "Dice_SE":   float(dice[3]),
        "Dice_HE":   float(dice[4]),
        "IoU_EX":    float(iou[1]),
        "IoU_MA":    float(iou[2]),
        "IoU_SE":    float(iou[3]),
        "IoU_HE":    float(iou[4]),
        "Mean_Dice": float(np.nanmean(dice[1:5])),
        "Mean_IoU":  float(np.nanmean(iou[1:5])),
        "aAcc":      float(ret["aAcc"]),
    }


def main():
    args   = parse_args()
    h, w   = get_dataset_config(args.dataset_type, args.dataset_name)
    order  = parse_order(args.order)

    image_dir = os.path.join(args.data_dir, "images", "")
    label_dir = os.path.join(args.data_dir, "masks",  "")

    output_csv = args.output_csv or os.path.join(args.prob_root, "metrics_test.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    print(f"[INFO] image_dir={image_dir}")
    print(f"[INFO] label_dir={label_dir}")
    print(f"[INFO] prob_root={args.prob_root}")
    print(f"[INFO] include_seg_metrics={args.include_seg_metrics}")

    _, image_names, labels = get_full_test_data(image_dir, label_dir, h, w)
    new_labels = make_label(labels)  # (N, H, W, 5) one-hot

    # Resolve the directory that holds the .npy files
    prob_dir = resolve_prob_dir(args.prob_root, args.prob_subdir)
    print(f"[INFO] Loading prob maps from: {prob_dir}")

    probs = load_probs(prob_dir, image_names, h, w, order)
    # probs shape: (N, H, W, 5)

    # ---- AUPR (soft, per class) -------------------------------------------
    ex        = PR_AUC(probs[:, :, :, 1], new_labels[:, :, :, 1])
    ma        = PR_AUC(probs[:, :, :, 2], new_labels[:, :, :, 2])
    se        = PR_AUC(probs[:, :, :, 3], new_labels[:, :, :, 3])
    he        = PR_AUC(probs[:, :, :, 4], new_labels[:, :, :, 4])
    mean_aupr = float(np.nanmean([ex, ma, se, he]))

    print(
        "[INFO] AUPR - "
        f"EX: {ex:.6f}, MA: {ma:.6f}, SE: {se:.6f}, HE: {he:.6f}, Mean: {mean_aupr:.6f}"
    )

    row = {
        "EX":        float(ex),
        "MA":        float(ma),
        "SE":        float(se),
        "HE":        float(he),
        "Mean_AUPR": mean_aupr,
    }

    # ---- Dice / IoU (optional) --------------------------------------------
    if args.include_seg_metrics:
        seg = compute_seg_metrics_mmseg(probs, labels, args.ignore_index)
        row.update(seg)
        print(
            "[INFO] Seg (metrics.py) - "
            f"Mean_Dice: {seg['Mean_Dice']:.4f}, Mean_IoU: {seg['Mean_IoU']:.4f}, "
            f"aAcc: {seg['aAcc']:.4f}"
        )

    # ---- Write CSV --------------------------------------------------------
    base_fields = ["EX", "MA", "SE", "HE", "Mean_AUPR"]
    seg_fields  = [
        "Dice_EX", "Dice_MA", "Dice_SE", "Dice_HE",
        "IoU_EX",  "IoU_MA",  "IoU_SE",  "IoU_HE",
        "Mean_Dice", "Mean_IoU", "aAcc",
    ]
    fieldnames = base_fields + (seg_fields if args.include_seg_metrics else [])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)

    print(f"[INFO] Saved metrics CSV to: {output_csv}")


if __name__ == "__main__":
    main()
