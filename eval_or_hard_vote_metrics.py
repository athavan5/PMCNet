"""
eval_or_hard_vote_metrics.py

Evaluate AUPR, DICE, and IoU for OR hard-vote binary masks produced by
or_hard_voting.py, in the same style as eval_fused_prob_maps_metrics.py.

OR mask layout written by or_hard_voting.py:
    <save-or-dir>/<class>/<image_stem>_probs.npy   shape: (H, W), uint8 {0,1}

Because OR masks are binary (not soft probabilities), AUPR is computed from
the binary mask values directly (0 or 1).  DICE and IoU are computed the
same way as eval_fused_prob_maps_metrics.py — no threshold is needed since
the mask is already binary.

CLI usage:
    python eval_or_hard_vote_metrics.py \
        --or-mask-dir  <save-or-dir from or_hard_voting.py> \
        --test-dir     ./data/<dataset>/test \
        --dataset-name <dataset_name> \
        --dataset-type IDRiD \
        --classes      ex,ma,se,he \
        --log-txt      <out-dir>/results.txt \
        --save-or-dir  <out-dir>
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from utils import Evaluator, PR_AUC, get_full_test_data, make_label


CLASS_NAMES   = ["bg", "ex", "ma", "se", "he"]
CLASS_TO_IDX  = {n: i for i, n in enumerate(CLASS_NAMES)}
# Index inside the (N, H, W, C) new_labels array returned by make_label
LESION_LABEL_IDX = {"ex": 1, "ma": 2, "se": 3, "he": 4}


# ── Tee (identical to eval_fused_prob_maps_metrics.py) ────────────────────────

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ── Dataset config (identical to eval_fused_prob_maps_metrics.py) ─────────────

def get_dataset_config(dataset_type, dataset_name):
    if dataset_type == "DDR":
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "IDRiD":
        return 960, 1440, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "IDRiD_w_PBDA":
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "MAPLES-DR":
        return (
            1024, 1024,
            "./data/IDRiD_MAPLES_combined/test/images/",
            "./data/IDRiD_MAPLES_combined/test/masks/",
        )
    if dataset_type == "epoch":
        return 960, 1440, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


# ── OR mask loader ─────────────────────────────────────────────────────────────

def load_or_mask(or_mask_dir: str, cls: str, image_stem: str) -> np.ndarray:
    """
    Load the binary OR mask for a single (class, image) pair.

    or_hard_voting.py saves masks using the full source stem, e.g.
    IDRiD_55_probs.npy, while image_stem from get_full_test_data is
    just IDRiD_55.  We try the _probs suffix first, then fall back to
    the bare stem so the function works regardless of naming convention.

    Expects shape (H, W), uint8.
    Returns float32 array with values in {0.0, 1.0}.
    """
    base = Path(or_mask_dir) / cls
    # Primary: stem as saved by or_hard_voting.py (<image_stem>_probs.npy)
    path = base / f"{image_stem}_probs.npy"
    if not path.exists():
        # Fallback: bare stem
        path = base / f"{image_stem}.npy"
    if not path.exists():
        raise FileNotFoundError(f"OR mask not found: {base / (image_stem + '_probs.npy')}")
    mask = np.load(path)
    if mask.ndim != 2:
        raise ValueError(f"Expected (H, W) mask, got {mask.shape} for {path}")
    return mask.astype(np.float32)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate AUPR/DICE/IoU for OR hard-vote binary masks."
    )
    p.add_argument("--or-mask-dir", required=True,
                   help="Root dir of OR masks (--save-or-dir from or_hard_voting.py).")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--dataset-type", default="IDRiD",
                   choices=["DDR", "IDRiD", "IDRiD_w_PBDA", "MAPLES-DR", "epoch"])
    p.add_argument("--test-dir", default=None,
                   help="Override test root: uses <test-dir>/images/ and <test-dir>/masks/.")
    p.add_argument("--classes", default="ex,ma,se,he",
                   help="Lesion classes to evaluate (comma-separated). Default: ex,ma,se,he.")
    p.add_argument("--save-or-dir", default=None,
                   help="If set, copy OR masks here as <stem>_or_<cls>.npy for downstream use.")
    p.add_argument("--log-txt", default=None,
                   help="If set, tee stdout to this .txt path.")
    return p.parse_args()


# ── Main evaluation ────────────────────────────────────────────────────────────

def _run_eval(args):
    h, w, default_image_dir, default_label_dir = get_dataset_config(
        args.dataset_type, args.dataset_name
    )
    if args.test_dir is not None:
        test_root = os.path.normpath(args.test_dir)
        image_dir = os.path.join(test_root, "images/")
        label_dir = os.path.join(test_root, "masks/")
    else:
        image_dir, label_dir = default_image_dir, default_label_dir

    classes = [c.strip() for c in args.classes.split(",")]

    _, image_names, labels = get_full_test_data(image_dir, label_dir, h, w)
    new_labels = make_label(labels)   # (N, H, W, C) one-hot GT

    n = len(image_names)
    print(f"Dataset : {args.dataset_name} ({args.dataset_type})")
    print(f"Images  : {n}")
    print(f"Classes : {classes}")
    print(f"OR masks: {args.or_mask_dir}")
    print()

    if args.save_or_dir is not None:
        os.makedirs(args.save_or_dir, exist_ok=True)

    # Accumulators keyed by class name
    evaluators  = {cls: Evaluator() for cls in classes}
    mask_stacks = {cls: [] for cls in classes}   # for AUPR (needs full stack)
    
    per_image_aupr = {cls: [] for cls in classes}
    
    per_image_dice = {cls: [] for cls in classes}
    missing     = []

    for i, name in enumerate(image_names):
        stem = os.path.splitext(os.path.basename(name))[0]

        # Check all class masks exist before accumulating anything for this image
        try:
            masks = {cls: load_or_mask(args.or_mask_dir, cls, stem) for cls in classes}
        except FileNotFoundError as e:
            missing.append(str(e))
            continue

        for cls in classes:
            mask = masks[cls]                          # (H, W) float32 {0,1}
            gt_i = new_labels[i, :, :, LESION_LABEL_IDX[cls]]      # (H, W) GT for this image
            
            mask_stacks[cls].append(mask)
            
            # Per-image AUPR: expand to (1, H, W) so PR_AUC receives the expected batch dim
            aupr_i = PR_AUC(mask[np.newaxis], gt_i[np.newaxis])
            per_image_aupr[cls].append((stem, aupr_i))
            
            # Per-image DICE: fresh Evaluator per image so the score reflects
            # this single image only (not accumulated across the dataset).
            # No threshold needed -- AND masks are already binary {0,1}.
            ev_i = Evaluator()
            ev_i.update(mask, gt_i)
            dice_i, _ = ev_i.show()
            per_image_dice[cls].append((stem, dice_i))

            # Optionally save a copy
            if args.save_or_dir is not None:
                out_path = os.path.join(args.save_or_dir, f"{stem}_or_{cls}.npy")
                np.save(out_path, mask.astype(np.uint8))

    if missing:
        print(f"[ERROR] Missing OR masks for {len(missing)} image(s):", file=sys.stderr)
        for m in missing[:5]:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    n_eval = len(mask_stacks[classes[0]])
    if n_eval != n:
        print(f"[WARN] Evaluated {n_eval} of {n} images.")

    # ── Per-class metrics ──────────────────────────────────────────────────────
    aupr_values, dice_values, iou_values = {}, {}, {}

    for cls in classes:
        label_idx = LESION_LABEL_IDX[cls]
        gt = new_labels[:n_eval, :, :, label_idx]          # (N, H, W)
        pred_stack = np.stack(mask_stacks[cls], axis=0)    # (N, H, W) float32

        # AUPR — binary masks used directly as "probabilities"
        aupr = PR_AUC(pred_stack, gt)
        aupr_values[cls] = aupr

        # DICE / IoU — mask is already binary, no threshold needed
        ev = evaluators[cls]
        for i in range(n_eval):
            ev.update(pred_stack[i], gt[i])
        dice, iou = ev.show()
        dice_values[cls] = dice
        iou_values[cls]  = iou
        
    # ── Print per-image AUPR and DICE ───────────────────────────────────────────
    print("Per-image AUPR and DICE (AND hard vote):")
    for cls in classes:
        print(f"  Class {cls.upper()}:")
        aupr_scores = [aupr_i for _, aupr_i in per_image_aupr[cls]]
        dice_scores = [dice_i for _, dice_i in per_image_dice[cls]]
        for (stem, aupr_i), (_, dice_i) in zip(per_image_aupr[cls], per_image_dice[cls]):
            print(f"    {stem}: AUPR={aupr_i:.6f}  DICE={dice_i:.6f}")
        print(f"    Mean AUPR across images: {np.nanmean(aupr_scores):.6f}")
        print(f"    Mean DICE across images: {np.nanmean(dice_scores):.6f}")
    print()
    
    '''
    # ── Print per-image AUPR ───────────────────────────────────────────────────
    print("Per-image AUPR (AND hard vote):")
    for cls in classes:
        print(f"  Class {cls.upper()}:")
        scores = [aupr_i for _, aupr_i in per_image_aupr[cls]]
        for stem, aupr_i in per_image_aupr[cls]:
            print(f"    {stem}: {aupr_i:.6f}")
        print(f"    Mean across images: {np.nanmean(scores):.6f}")
    print()
    '''

    # ── Print whole test-set results ───────────────────────────────────────────
    print("Whole test-set AUPR (AND hard vote, pooled):")
    for cls in classes:
        print(f"  {cls.upper()}: {aupr_values[cls]:.6f}")
    print(f"  Mean AUPR: {np.nanmean(list(aupr_values.values())):.6f}")
    print()

    print("DICE Results (AND hard vote, binary mask vs GT):")
    for cls in classes:
        print(f"  {cls.upper()}: {dice_values[cls]:.6f}")
    print(f"  Mean DICE: {np.nanmean(list(dice_values.values())):.6f}")
    print()

    print("IoU Results (AND hard vote, binary mask vs GT):")
    for cls in classes:
        print(f"  {cls.upper()}: {iou_values[cls]:.6f}")
    print(f"  Mean IoU: {np.nanmean(list(iou_values.values())):.6f}")


def main():
    args = parse_args()
    log_file = None
    saved_stdout = sys.stdout

    if args.log_txt is not None:
        log_dir = os.path.dirname(os.path.abspath(args.log_txt))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_file = open(args.log_txt, "w", encoding="utf-8")
        sys.stdout = Tee(saved_stdout, log_file)

    try:
        _run_eval(args)
    finally:
        if log_file is not None:
            sys.stdout = saved_stdout
            log_file.close()


if __name__ == "__main__":
    main()