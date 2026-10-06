"""
Evaluate one folder of per-image probability maps (*_probs.npy) against test masks.

Same metrics as eval_fused_prob_maps.py (AUPR / DICE / IoU) but **no fusion** — each
image loads a single HxWx5 array from --prob-dir.

Channel layout defaults to bg,ex,ma,se,he on disk; override with --order if needed.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from eval_fused_prob_maps import (
    Tee,
    align_to_target_hw,
    get_dataset_config,
    load_prob_npy,
    make_label,
    parse_order,
    renormalize_probs,
    reorder_channels,
)
from utils import Evaluator, PR_AUC, get_full_test_data

# Must match eval_fused_prob_maps.py / test.py
CLASS_NAMES = ["bg", "ex", "ma", "se", "he"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a single prob_maps folder (*_probs.npy per basename)."
    )
    p.add_argument(
        "--prob-dir",
        dest="prob_dir",
        type=str,
        required=True,
        help="Directory containing {basename}_probs.npy files.",
    )
    p.add_argument("--dataset-name", dest="dataset_name", type=str, required=True)
    p.add_argument(
        "--dataset-type",
        dest="dataset_type",
        type=str,
        default="IDRiD",
        choices=["DDR", "IDRiD", "IDRiD_w_PBDA", "MAPLES-DR", "epoch"],
    )
    p.add_argument(
        "--test-dir",
        dest="test_dir",
        type=str,
        default=None,
        help="Override test root: <test-dir>/images/ and <test-dir>/masks/",
    )
    p.add_argument(
        "--class-mode",
        dest="class_mode",
        type=str,
        default="all",
        choices=["bg", "ex", "ma", "se", "he", "all"],
    )
    p.add_argument(
        "--dice-threshold",
        dest="dice_threshold",
        type=float,
        default=0.5,
        help="Threshold for DICE/IoU when class_mode is a single lesion class.",
    )
    p.add_argument(
        "--order",
        dest="order",
        type=str,
        default="bg,ex,ma,se,he",
        help="Comma-separated on-disk channel order in prob-dir.",
    )
    p.add_argument(
        "--log-txt",
        dest="log_txt",
        type=str,
        default=None,
        help="Tee stdout to this path.",
    )
    p.add_argument(
        "--save-copy-dir",
        dest="save_copy_dir",
        type=str,
        default=None,
        help="If set, save aligned HxWx5 arrays as {base}_probs_eval.npy here.",
    )
    return p.parse_args()


def _run_eval(args):
    h, w, default_image_dir, default_label_dir = get_dataset_config(args.dataset_type, args.dataset_name)
    if args.test_dir is not None:
        test_root = os.path.normpath(args.test_dir)
        image_dir = os.path.join(test_root, "images/")
        label_dir = os.path.join(test_root, "masks/")
    else:
        image_dir, label_dir = default_image_dir, default_label_dir

    order = parse_order(args.order)
    class_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}

    _, image_names, labels = get_full_test_data(image_dir, label_dir, h, w)
    new_labels = make_label(labels)

    if args.save_copy_dir is not None:
        os.makedirs(args.save_copy_dir, exist_ok=True)

    stack = []
    missing = []

    for name in image_names:
        base = os.path.splitext(os.path.basename(name))[0]
        path = os.path.join(args.prob_dir, f"{base}_probs.npy")
        if not os.path.isfile(path):
            missing.append((base, path))
            continue
        prob = load_prob_npy(path)
        prob = reorder_channels(prob, order, CLASS_NAMES)

        if args.class_mode == "all":
            prob = align_to_target_hw(prob, h, w, base)
            prob = renormalize_probs(np.clip(prob, 0.0, 1.0))
            stack.append(prob)
            if args.save_copy_dir is not None:
                np.save(
                    os.path.join(args.save_copy_dir, f"{base}_probs_eval.npy"),
                    prob.astype(np.float32),
                )
        else:
            ci = class_to_idx[args.class_mode]
            ch = align_to_target_hw(prob[:, :, ci], h, w, base)
            stack.append(ch)
            if args.save_copy_dir is not None:
                np.save(
                    os.path.join(args.save_copy_dir, f"{base}_prob_{args.class_mode}_eval.npy"),
                    ch.astype(np.float32),
                )

    if missing:
        print(f"[ERROR] Missing prob files for {len(missing)} image(s). Example: {missing[0]}", file=sys.stderr)
        sys.exit(1)
    if not stack:
        print("[ERROR] No probability maps evaluated.", file=sys.stderr)
        sys.exit(1)

    n = len(stack)
    if n != len(image_names):
        print(f"[WARN] Evaluated count {n} != image count {len(image_names)}")

    if args.class_mode == "all":
        probs = np.stack(stack, axis=0)
        EX = PR_AUC(probs[:, :, :, 1], new_labels[:n, :, :, 1])
        MA = PR_AUC(probs[:, :, :, 2], new_labels[:n, :, :, 2])
        SE = PR_AUC(probs[:, :, :, 3], new_labels[:n, :, :, 3])
        HE = PR_AUC(probs[:, :, :, 4], new_labels[:n, :, :, 4])

        for name, lab in [
            ("EX", new_labels[:n, :, :, 1]),
            ("MA", new_labels[:n, :, :, 2]),
            ("SE", new_labels[:n, :, :, 3]),
            ("HE", new_labels[:n, :, :, 4]),
        ]:
            if np.sum(lab) == 0:
                print(f"[INFO] No positive pixels for class {name} in evaluated set — PR-AUC may be 0.0")

        print("AUPR Results:")
        print("EX:", EX)
        print("MA:", MA)
        print("SE:", SE)
        print("HE:", HE)
        mean_aupr = np.nanmean([EX, MA, SE, HE])
        print("Mean AUPR:", mean_aupr)

        predictions = np.argmax(probs, axis=-1)
        pred = make_label(predictions)
        evaluator_EX = Evaluator()
        evaluator_MA = Evaluator()
        evaluator_SE = Evaluator()
        evaluator_HE = Evaluator()
        for i in range(n):
            evaluator_EX.update(pred[i, :, :, 1], new_labels[i, :, :, 1])
            evaluator_MA.update(pred[i, :, :, 2], new_labels[i, :, :, 2])
            evaluator_SE.update(pred[i, :, :, 3], new_labels[i, :, :, 3])
            evaluator_HE.update(pred[i, :, :, 4], new_labels[i, :, :, 4])

        ex_dice, ex_iou = evaluator_EX.show()
        ma_dice, ma_iou = evaluator_MA.show()
        se_dice, se_iou = evaluator_SE.show()
        he_dice, he_iou = evaluator_HE.show()
        mean_dice = (ex_dice + ma_dice + se_dice + he_dice) / 4
        mean_iou = (ex_iou + ma_iou + se_iou + he_iou) / 4

        print("DICE Results (per class, argmax vs GT):")
        print("EX:", ex_dice)
        print("MA:", ma_dice)
        print("SE:", se_dice)
        print("HE:", he_dice)
        print("IoU Results (per class, argmax vs GT):")
        print("EX:", ex_iou)
        print("MA:", ma_iou)
        print("SE:", se_iou)
        print("HE:", he_iou)
        print("mean_dice:", mean_dice, " mean_iou: ", mean_iou)
    else:
        ci = class_to_idx[args.class_mode]
        fused_ch = np.stack(stack, axis=0)
        gt = new_labels[:n, :, :, ci]
        aupr = PR_AUC(fused_ch, gt)
        if np.sum(gt) == 0:
            print(f"[INFO] No positive pixels for class {args.class_mode.upper()} — PR-AUC may be undefined.")

        print(f"AUPR ({args.class_mode}):", aupr)

        pred_bin = (fused_ch >= args.dice_threshold).astype(np.float32)
        ev = Evaluator()
        for i in range(n):
            ev.update(pred_bin[i], gt[i])
        dice, iou = ev.show()
        print(f"DICE ({args.class_mode}, threshold={args.dice_threshold}):", dice)
        print(f"IoU  ({args.class_mode}, threshold={args.dice_threshold}):", iou)


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
        print(f"[INFO] prob-dir: {args.prob_dir}")
        print(f"[INFO] order: {args.order}")
        _run_eval(args)
    finally:
        if log_file is not None:
            sys.stdout = saved_stdout
            log_file.close()


if __name__ == "__main__":
    main()
