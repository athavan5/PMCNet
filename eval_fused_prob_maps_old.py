"""
Fuse two PMCNet probability-map folders (from test_prob_maps.py) and evaluate
AUPR / DICE / IoU like test.py.

For each image, loads {basename}_probs.npy from both directories (H x W x 5,
channels: bg, ex, he, ma, se).

- Single class: multiplies that channel only; AUPR vs GT; DICE/IoU via threshold.
- all: element-wise multiply full tensors, renormalize per pixel to a distribution,
  then same multiclass metrics as test.py (argmax + Evaluator per class).
"""

import argparse
import os
import sys

import cv2
import numpy as np

from utils import Evaluator, PR_AUC, get_full_test_data, make_label


class Tee:
    """Write to multiple streams (e.g. console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def parse_args():
    p = argparse.ArgumentParser(description="Fuse two prob_maps folders and evaluate metrics.")
    p.add_argument("--prob-dir-a", dest="prob_dir_a", type=str, required=True, help="First folder with *_probs.npy")
    p.add_argument("--prob-dir-b", dest="prob_dir_b", type=str, required=True, help="Second folder with *_probs.npy")
    p.add_argument("--dataset-name", dest="dataset_name", type=str, required=True)
    p.add_argument(
        "--dataset-type",
        dest="dataset_type",
        type=str,
        default="IDRiD",
        choices=["DDR", "IDRiD", "IDRiD_w_PBDA", "MAPLES-DR", "epoch"],
    )
    p.add_argument(
        "--class-mode",
        dest="class_mode",
        type=str,
        default="ex",
        choices=["bg", "ex", "he", "ma", "se", "all"],
        help="Fuse one channel only, or all channels (renormalize) for full multiclass metrics.",
    )
    p.add_argument(
        "--dice-threshold",
        dest="dice_threshold",
        type=float,
        default=0.5,
        help="Threshold on fused probability for DICE/IoU when class_mode is a single class.",
    )
    p.add_argument(
        "--save-fused-dir",
        dest="save_fused_dir",
        type=str,
        default=None,
        help="If set, save fused HxWx5 arrays as {base}_probs_fused.npy here.",
    )
    p.add_argument(
        "--log-txt",
        dest="log_txt",
        type=str,
        default=None,
        help="If set, tee stdout to this .txt path (parent dirs are created).",
    )
    p.add_argument(
        "--save-png",
        dest="save_png",
        type=str,
        default=None,
        choices=["all", "bg", "ex", "he", "ma", "se"],
        help="Save grayscale PNG heatmaps from fused maps for one class or all classes.",
    )
    p.add_argument(
        "--order-a",
        dest="order_a",
        type=str,
        default="bg,ex,he,ma,se",
        help="Comma-separated class order in prob-dir-a channels, e.g. bg,ex,he,ma,se",
    )
    p.add_argument(
        "--order-b",
        dest="order_b",
        type=str,
        default="bg,ex,he,ma,se",
        help="Comma-separated class order in prob-dir-b channels, e.g. bg,ex,ma,he,se",
    )
    return p.parse_args()


def get_dataset_config(dataset_type, dataset_name):
    if dataset_type == "DDR":
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "IDRiD":
        return 960, 1440, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "IDRiD_w_PBDA":
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "MAPLES-DR":
        return (
            1024,
            1024,
            "./data/IDRiD_MAPLES_combined/test/images/",
            "./data/IDRiD_MAPLES_combined/test/masks/",
        )
    if dataset_type == "epoch":
        return 960, 1440, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    raise ValueError(f"Unsupported dataset: {dataset_type}")


# Old order: CLASS_NAMES = ["bg", "ex", "he", "ma", "se"]
CLASS_NAMES = ["bg", "ex", "ma", "se", "he"]
LESION_INDICES = [1, 3, 4, 2]  # EX, MA, SE, HE — same as test.py


def parse_order(order_str):
    order = [x.strip().lower() for x in order_str.split(",")]
    if len(order) != len(CLASS_NAMES):
        raise ValueError(
            f"Order must have {len(CLASS_NAMES)} classes, got {len(order)}: {order_str}"
        )
    if sorted(order) != sorted(CLASS_NAMES):
        raise ValueError(
            f"Order must be a permutation of {CLASS_NAMES}, got: {order}"
        )
    return order


def reorder_channels(prob_hwc, src_order, dst_order):
    if prob_hwc.shape[-1] != len(src_order):
        raise ValueError(
            f"Channel count {prob_hwc.shape[-1]} does not match src order length {len(src_order)}"
        )
    src_idx = {name: i for i, name in enumerate(src_order)}
    gather_idx = [src_idx[name] for name in dst_order]
    return prob_hwc[:, :, gather_idx]


def load_prob_npy(path):
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC prob map, got shape {arr.shape} in {path}")
    return arr.astype(np.float32)


def align_spatial(a, b, path_a, path_b):
    """Resize b to a's H,W if needed (nearest)."""
    if a.shape[:2] == b.shape[:2]:
        return b
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    print(f"[WARN] Shape mismatch {path_a} {a.shape[:2]} vs {path_b} {b.shape[:2]}; resizing B to A.")
    b_resized = cv2.resize(b, (wa, ha), interpolation=cv2.INTER_NEAREST)
    return b_resized.astype(np.float32)


def align_to_target_hw(arr, target_h, target_w, tag):
    """Resize array to target H,W if needed."""
    if arr.shape[:2] == (target_h, target_w):
        return arr
    print(
        f"[WARN] Fused map {tag} has shape {arr.shape[:2]}; "
        f"resizing to evaluation size {(target_h, target_w)}."
    )
    resized = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def fuse_multiply(a, b):
    return np.clip(a * b, 0.0, 1.0)


def renormalize_probs(fused_hwc, eps=1e-8):
    s = np.sum(fused_hwc, axis=-1, keepdims=True)
    return fused_hwc / (s + eps)


def save_fused_pngs(fused_hwc, base, out_dir, save_png):
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    if save_png == "all":
        class_indices = list(range(fused_hwc.shape[-1]))
    else:
        class_indices = [class_to_idx[save_png]]

    for c in class_indices:
        ch = np.clip(fused_hwc[:, :, c], 0.0, 1.0)
        ch_u8 = (ch * 255.0).astype(np.uint8)
        out_name = f"{base}_prob_{CLASS_NAMES[c]}_fused.png"
        cv2.imwrite(os.path.join(out_dir, out_name), ch_u8)


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


def _run_eval(args):
    h, w, image_dir, label_dir = get_dataset_config(args.dataset_type, args.dataset_name)
    order_a = parse_order(args.order_a)
    order_b = parse_order(args.order_b)
    effective_save_png = args.save_png if args.save_png is not None else (
        "all" if args.class_mode == "all" else args.class_mode
    )

    _, image_names, labels = get_full_test_data(image_dir, label_dir, h, w)
    new_labels = make_label(labels)

    class_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}

    if args.save_fused_dir is not None:
        os.makedirs(args.save_fused_dir, exist_ok=True)

    fused_stack = []  # N x H x W x C or N x H x W for single-class
    missing = []

    for name in image_names:
        base = os.path.splitext(os.path.basename(name))[0]
        pa = os.path.join(args.prob_dir_a, f"{base}_probs.npy")
        pb = os.path.join(args.prob_dir_b, f"{base}_probs.npy")
        if not os.path.isfile(pa) or not os.path.isfile(pb):
            missing.append((base, pa, pb))
            continue
        prob_a = load_prob_npy(pa)
        prob_b = load_prob_npy(pb)
        prob_b = align_spatial(prob_a, prob_b, pa, pb)
        prob_a = reorder_channels(prob_a, order_a, CLASS_NAMES)
        prob_b = reorder_channels(prob_b, order_b, CLASS_NAMES)
        if prob_a.shape != prob_b.shape:
            raise ValueError(f"Channel mismatch after align: {pa} {prob_a.shape} vs {pb} {prob_b.shape}")
        if args.class_mode == "all":
            fused = renormalize_probs(fuse_multiply(prob_a, prob_b))
            fused = align_to_target_hw(fused, h, w, base)
            fused = renormalize_probs(np.clip(fused, 0.0, 1.0))
        else:
            ci = class_to_idx[args.class_mode]
            fused_ch = fuse_multiply(prob_a[:, :, ci], prob_b[:, :, ci])
            fused = align_to_target_hw(fused_ch, h, w, base)
        fused_stack.append(fused)
        if args.save_fused_dir is not None:
            if args.class_mode == "all":
                np.save(os.path.join(args.save_fused_dir, f"{base}_probs_fused.npy"), fused.astype(np.float32))
                if effective_save_png is not None:
                    save_fused_pngs(fused, base, args.save_fused_dir, effective_save_png)

    if missing:
        print(f"[ERROR] Missing prob files for {len(missing)} image(s). Example: {missing[0]}", file=sys.stderr)
        sys.exit(1)
    if not fused_stack:
        print("[ERROR] No fused maps produced.", file=sys.stderr)
        sys.exit(1)

    n = len(fused_stack)
    if n != len(image_names):
        print(f"[WARN] Fused count {n} != image count {len(image_names)}")

    if args.class_mode == "all":
        probs = np.stack(fused_stack, axis=0)
        EX = PR_AUC(probs[:, :, :, 1], new_labels[:n, :, :, 1])
        HE = PR_AUC(probs[:, :, :, 2], new_labels[:n, :, :, 2])
        MA = PR_AUC(probs[:, :, :, 3], new_labels[:n, :, :, 3])
        SE = PR_AUC(probs[:, :, :, 4], new_labels[:n, :, :, 4])

        for name, lab in [
            ("EX", new_labels[:n, :, :, 1]),
            ("HE", new_labels[:n, :, :, 2]),
            ("MA", new_labels[:n, :, :, 3]),
            ("SE", new_labels[:n, :, :, 4]),
        ]:
            if np.sum(lab) == 0:
                print(f"[INFO] No positive pixels for class {name} in evaluated set — PR-AUC may be 0.0")

        print("AUPR Results (fused, renormalized full tensor):")
        print("EX:", EX)
        print("HE:", HE)
        print("MA:", MA)
        print("SE:", SE)
        mean_aupr = np.nanmean([MA, HE, EX, SE])
        print("Mean AUPR:", mean_aupr)

        predictions = np.argmax(probs, axis=-1)
        pred = make_label(predictions)
        evaluator_MA = Evaluator()
        evaluator_HE = Evaluator()
        evaluator_EX = Evaluator()
        evaluator_SE = Evaluator()
        for i in range(n):
            evaluator_EX.update(pred[i, :, :, 1], new_labels[i, :, :, 1])
            evaluator_HE.update(pred[i, :, :, 2], new_labels[i, :, :, 2])
            evaluator_MA.update(pred[i, :, :, 3], new_labels[i, :, :, 3])
            evaluator_SE.update(pred[i, :, :, 4], new_labels[i, :, :, 4])

        ma_dice, ma_iou = evaluator_MA.show()
        he_dice, he_iou = evaluator_HE.show()
        ex_dice, ex_iou = evaluator_EX.show()
        se_dice, se_iou = evaluator_SE.show()
        mean_dice = (ma_dice + he_dice + ex_dice + se_dice) / 4
        mean_iou = (ma_iou + he_iou + ex_iou + se_iou) / 4
        print("mean_dice:", mean_dice, " mean_iou:", mean_iou)
    else:
        ci = class_to_idx[args.class_mode]
        fused_ch = np.stack(fused_stack, axis=0)
        gt = new_labels[:n, :, :, ci]
        aupr = PR_AUC(fused_ch, gt)
        if np.sum(gt) == 0:
            print(f"[INFO] No positive pixels for class {args.class_mode.upper()} — PR-AUC set to 0.0 if undefined.")

        print(f"AUPR ({args.class_mode}, fused product):", aupr)

        pred_bin = (fused_ch >= args.dice_threshold).astype(np.float32)
        ev = Evaluator()
        for i in range(n):
            ev.update(pred_bin[i], gt[i])
        dice, iou = ev.show()
        print(f"DICE ({args.class_mode}, threshold={args.dice_threshold}):", dice)
        print(f"IoU  ({args.class_mode}, threshold={args.dice_threshold}):", iou)

        if args.save_fused_dir:
            for i, name in enumerate(image_names[:n]):
                base = os.path.splitext(os.path.basename(name))[0]
                np.save(
                    os.path.join(args.save_fused_dir, f"{base}_prob_{args.class_mode}_fused.npy"),
                    fused_ch[i].astype(np.float32),
                )
                if effective_save_png is not None:
                    ch = np.clip(fused_ch[i], 0.0, 1.0)
                    ch_u8 = (ch * 255.0).astype(np.uint8)
                    out_name = f"{base}_prob_{args.class_mode}_fused.png"
                    cv2.imwrite(os.path.join(args.save_fused_dir, out_name), ch_u8)


if __name__ == "__main__":
    main()
