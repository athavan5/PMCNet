"""
Fuse two or three PMCNet probability-map folders (from test_prob_maps.py) and evaluate
AUPR / DICE / IoU like test.py.

For each image, loads {basename}_probs.npy from two required directories and
an optional third directory (H x W x 5,
channels: bg then lesions ex, ma, se, he — i.e. bg, ex, ma, se, he unless
--order-a / --order-b / --order-c specify a different on-disk layout).

- Single class: weighted sum on that channel only; AUPR vs GT; DICE/IoU via threshold.
- all: weighted sum on full tensors, renormalize per pixel to a distribution,
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
    p = argparse.ArgumentParser(description="Fuse two or three prob_maps folders and evaluate metrics.")
    p.add_argument("--prob-dir-a", dest="prob_dir_a", type=str, required=True, help="First folder with *_probs.npy")
    p.add_argument("--prob-dir-b", dest="prob_dir_b", type=str, required=True, help="Second folder with *_probs.npy")
    p.add_argument(
        "--prob-dir-c",
        dest="prob_dir_c",
        type=str,
        default=None,
        help="Optional third folder with *_probs.npy",
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
        help="Override test data root: uses <test-dir>/images/ and <test-dir>/masks/. "
        "If omitted, image and mask paths come from get_dataset_config(dataset_type, dataset_name).",
    )
    p.add_argument(
        "--class-mode",
        dest="class_mode",
        type=str,
        default="ex",
        choices=["bg", "ex", "ma", "se", "he", "all"],
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
        choices=["all", "bg", "ex", "ma", "se", "he"],
        help="Save grayscale PNG heatmaps from fused maps for one class or all classes.",
    )
    p.add_argument(
        "--order-a",
        dest="order_a",
        type=str,
        default="bg,ex,ma,se,he",
        help="Comma-separated on-disk channel order in prob-dir-a (default: bg,ex,ma,se,he).",
    )
    p.add_argument(
        "--order-b",
        dest="order_b",
        type=str,
        default="bg,ex,ma,se,he",
        help="Comma-separated on-disk channel order in prob-dir-b (default: bg,ex,ma,se,he).",
    )
    p.add_argument(
        "--order-c",
        dest="order_c",
        type=str,
        default="bg,ex,ma,se,he",
        help="Comma-separated on-disk channel order in prob-dir-c (default: bg,ex,ma,se,he).",
    )
    p.add_argument(
        "--weights",
        dest="weights",
        type=str,
        default=None,
        help="Optional fusion weights as comma-separated values for active sources. "
        "Use 2 values for a,b (e.g. 0.4,0.6) or 3 values for a,b,c (e.g. 0.2,0.4,0.4). "
        "Defaults to equal weights.",
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
LESION_INDICES = [1, 2, 3, 4]  # EX, MA, SE, HE — same as test.py


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


def parse_weights(weights_str, n_sources):
    if n_sources not in (2, 3):
        raise ValueError(f"Expected 2 or 3 sources, got {n_sources}")
    if weights_str is None:
        return np.full((n_sources,), 1.0 / float(n_sources), dtype=np.float32)
    parts = [x.strip() for x in weights_str.split(",")]
    if len(parts) != n_sources:
        raise ValueError(
            f"--weights must contain exactly {n_sources} comma-separated values, got {len(parts)}: {weights_str}"
        )
    weights = np.array([float(x) for x in parts], dtype=np.float32)
    if np.any(weights < 0.0):
        raise ValueError(f"--weights must be non-negative, got: {weights.tolist()}")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError(f"--weights must sum to > 0, got: {weights.tolist()}")
    return weights / weight_sum


def fuse_weighted_sum(a, b, c, weights):
    fused = (weights[0] * a) + (weights[1] * b)
    if c is not None:
        fused = fused + (weights[2] * c)
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


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
    h, w, default_image_dir, default_label_dir = get_dataset_config(args.dataset_type, args.dataset_name)
    if args.test_dir is not None:
        test_root = os.path.normpath(args.test_dir)
        image_dir = os.path.join(test_root, "images/")
        label_dir = os.path.join(test_root, "masks/")
    else:
        image_dir, label_dir = default_image_dir, default_label_dir
    order_a = parse_order(args.order_a)
    order_b = parse_order(args.order_b)
    use_third_source = args.prob_dir_c is not None
    order_c = parse_order(args.order_c) if use_third_source else None
    n_sources = 3 if use_third_source else 2
    weights = parse_weights(args.weights, n_sources)
    effective_save_png = args.save_png
    weight_labels = "a,b,c" if use_third_source else "a,b"
    print(f"Using fusion weights ({weight_labels}): {weights.tolist()}")

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
        pc = os.path.join(args.prob_dir_c, f"{base}_probs.npy") if use_third_source else None
        if not os.path.isfile(pa) or not os.path.isfile(pb) or (use_third_source and not os.path.isfile(pc)):
            if use_third_source:
                missing.append((base, pa, pb, pc))
            else:
                missing.append((base, pa, pb))
            continue
        prob_a = load_prob_npy(pa)
        prob_b = load_prob_npy(pb)
        prob_c = load_prob_npy(pc) if use_third_source else None
        prob_b = align_spatial(prob_a, prob_b, pa, pb)
        if use_third_source:
            prob_c = align_spatial(prob_a, prob_c, pa, pc)
        prob_a = reorder_channels(prob_a, order_a, CLASS_NAMES)
        prob_b = reorder_channels(prob_b, order_b, CLASS_NAMES)
        if use_third_source:
            prob_c = reorder_channels(prob_c, order_c, CLASS_NAMES)
        if prob_a.shape != prob_b.shape:
            raise ValueError(f"Channel mismatch after align: {pa} {prob_a.shape} vs {pb} {prob_b.shape}")
        if use_third_source and prob_a.shape != prob_c.shape:
            raise ValueError(f"Channel mismatch after align: {pa} {prob_a.shape} vs {pc} {prob_c.shape}")
        if args.class_mode == "all":
            fused = renormalize_probs(fuse_weighted_sum(prob_a, prob_b, prob_c, weights))
            fused = align_to_target_hw(fused, h, w, base)
            fused = renormalize_probs(np.clip(fused, 0.0, 1.0))
        else:
            ci = class_to_idx[args.class_mode]
            fused_ch = fuse_weighted_sum(prob_a[:, :, ci], prob_b[:, :, ci], prob_c[:, :, ci], weights)
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
        MA = PR_AUC(probs[:, :, :, 2], new_labels[:n, :, :, 2])
        SE = PR_AUC(probs[:, :, :, 3], new_labels[:n, :, :, 3])
        HE = PR_AUC(probs[:, :, :, 4], new_labels[:n, :, :, 4])

        for name, lab in [
            ("EX", new_labels[:n, :, :, 1]),
            ("MA", new_labels[:n, :, :, 2]),
            ("SE", new_labels[:n, :, :, 3]),
            ("HE", new_labels[:n, :, :, 4])
        ]:
            if np.sum(lab) == 0:
                print(f"[INFO] No positive pixels for class {name} in evaluated set — PR-AUC may be 0.0")

        print("AUPR Results (fused, renormalized full tensor):")
        print("EX:", EX)
        print("MA:", MA)
        print("SE:", SE)
        print("HE:", HE)
        mean_aupr = np.nanmean([EX, MA, SE, HE])
        print("Mean AUPR:", mean_aupr)
        
        '''
        #Outputs AUPR for each separate test image (only use if necessary, otherwise comment this part off)
        # Per-image AUPR
        print("\nPer-image AUPR:")
        for i, name in enumerate(image_names[:n]):
            base = os.path.splitext(os.path.basename(name))[0]
            ex_i  = PR_AUC(probs[i:i+1, :, :, 1], new_labels[i:i+1, :, :, 1])
            ma_i  = PR_AUC(probs[i:i+1, :, :, 2], new_labels[i:i+1, :, :, 2])
            se_i  = PR_AUC(probs[i:i+1, :, :, 3], new_labels[i:i+1, :, :, 3])
            he_i  = PR_AUC(probs[i:i+1, :, :, 4], new_labels[i:i+1, :, :, 4])
            mean_i = np.nanmean([ex_i, ma_i, se_i, he_i])
            print(f"  {base}  EX:{ex_i:.4f}  MA:{ma_i:.4f}  SE:{se_i:.4f}  HE:{he_i:.4f}  mean:{mean_i:.4f}")
        '''

        predictions = np.argmax(probs, axis=-1)
        pred = make_label(predictions)
        
        print("\nPer-image AUPR and DICE:")
        for i, name in enumerate(image_names[:n]):
            base = os.path.splitext(os.path.basename(name))[0]
            ex_i  = PR_AUC(probs[i:i+1, :, :, 1], new_labels[i:i+1, :, :, 1])
            ma_i  = PR_AUC(probs[i:i+1, :, :, 2], new_labels[i:i+1, :, :, 2])
            se_i  = PR_AUC(probs[i:i+1, :, :, 3], new_labels[i:i+1, :, :, 3])
            he_i  = PR_AUC(probs[i:i+1, :, :, 4], new_labels[i:i+1, :, :, 4])
            mean_aupr_i = np.nanmean([ex_i, ma_i, se_i, he_i])

            # Fresh Evaluator per class per image so DICE is computed on
            # this single image only (not accumulated across the dataset).
            ev_ex_i = Evaluator()
            ev_ma_i = Evaluator()
            ev_se_i = Evaluator()
            ev_he_i = Evaluator()
            ev_ex_i.update(pred[i, :, :, 1], new_labels[i, :, :, 1])
            ev_ma_i.update(pred[i, :, :, 2], new_labels[i, :, :, 2])
            ev_se_i.update(pred[i, :, :, 3], new_labels[i, :, :, 3])
            ev_he_i.update(pred[i, :, :, 4], new_labels[i, :, :, 4])
            ex_dice_i, _ = ev_ex_i.show()
            ma_dice_i, _ = ev_ma_i.show()
            se_dice_i, _ = ev_se_i.show()
            he_dice_i, _ = ev_he_i.show()
            mean_dice_i = np.nanmean([ex_dice_i, ma_dice_i, se_dice_i, he_dice_i])

            print(f"  {base}  AUPR  EX:{ex_i:.6f}  MA:{ma_i:.6f}  SE:{se_i:.6f}  HE:{he_i:.6f}  mean:{mean_aupr_i:.6f}")
            print(f"  {' ' * len(base)}  DICE  EX:{ex_dice_i:.6f}  MA:{ma_dice_i:.6f}  SE:{se_dice_i:.6f}  HE:{he_dice_i:.6f}  mean:{mean_dice_i:.6f}\n")

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
        fused_ch = np.stack(fused_stack, axis=0)
        gt = new_labels[:n, :, :, ci]
        aupr = PR_AUC(fused_ch, gt)
        if np.sum(gt) == 0:
            print(f"[INFO] No positive pixels for class {args.class_mode.upper()} — PR-AUC set to 0.0 if undefined.")

        print(f"AUPR ({args.class_mode}, fused weighted sum):", aupr)

        pred_bin = (fused_ch >= args.dice_threshold).astype(np.float32)
        ev = Evaluator()
        for i in range(n):
            ev.update(pred_bin[i], gt[i])
        dice, iou = ev.show()
        print(f"DICE ({args.class_mode}, threshold={args.dice_threshold}):", dice)
        print(f"IoU  ({args.class_mode}, threshold={args.dice_threshold}):", iou)

        # Per-image DICE/IoU (fresh Evaluator per image, not accumulated).
        print(f"\nPer-image DICE ({args.class_mode}, threshold={args.dice_threshold}):")
        for i, name in enumerate(image_names[:n]):
            base = os.path.splitext(os.path.basename(name))[0]
            ev_i = Evaluator()
            ev_i.update(pred_bin[i], gt[i])
            dice_i, iou_i = ev_i.show()
            print(f"  {base}  DICE:{dice_i:.4f}  IoU:{iou_i:.4f}")

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
