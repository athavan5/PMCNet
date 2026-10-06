r"""
threshold_k_hard_voting.py

Threshold-k hard voting across N probability map directories.

Each model contributes one .npy file per image with shape (H, W, C).
For every (image, class) pair the per-channel maps are binarised at
`threshold` and reduced by threshold-k: a pixel is labelled lesion (1)
only if AT LEAST k models voted lesion.

Special cases:
    k = 1        equivalent to OR  (union)
    k = N        equivalent to AND (intersection)
    k = ceil(N/2) equivalent to majority voting

Expected filename convention (consistent across all model dirs):
    <prob-dir>/<image_id>_probs.npy   shape: (H, W, C)

CLI usage:
    python threshold_k_hard_voting.py \
        --prob-dir-a   <path> \
        --prob-dir-b   <path> \
        --prob-dir-c   <path> \
        --order-a      bg,ex,ma,se,he \
        --order-b      bg,ex,ma,se,he \
        --order-c      bg,ex,ma,se,he \
        --classes      ex,ma,se,he \
        --threshold    0.5 \
        --k            2 \
        --save-k-dir   <output-root>
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── Low-level helpers ──────────────────────────────────────────────────────────

def load_channel(path: Path, class_name: str, order: list[str]) -> np.ndarray:
    """
    Load a single lesion-class channel from a (H, W, C) .npy prob map.

    Args:
        path:       Path to the .npy file, shape (H, W, C).
        class_name: Lesion class to extract, e.g. 'ex'.
        order:      Channel order declared by --order-X,
                    e.g. ['bg', 'ex', 'ma', 'se', 'he'].

    Returns:
        2-D float32 array of shape (H, W) with values in [0, 1].
    """
    arr = np.load(path).astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f"{path}: expected shape (H, W, C), got {arr.shape}.")

    if class_name not in order:
        raise ValueError(f"Class '{class_name}' not found in order {order}.")

    channel_idx = order.index(class_name)

    if channel_idx >= arr.shape[2]:
        raise ValueError(
            f"{path}: channel index {channel_idx} out of range for shape {arr.shape}."
        )

    return np.clip(arr[:, :, channel_idx], 0.0, 1.0)


def resize_to(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Bilinear resize (H, W) arr to target_shape if shapes differ."""
    if arr.shape == target_shape:
        return arr
    factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    log.warning("Resizing %s -> %s", arr.shape, target_shape)
    return zoom(arr, factors, order=1).astype(np.float32)


def binarise(prob_map: np.ndarray, threshold: float) -> np.ndarray:
    return (prob_map >= threshold).astype(np.uint8)


# ── Core threshold-k voting ────────────────────────────────────────────────────

def threshold_k_hard_vote_channels(
    prob_map_paths: list[Path],
    orders: list[list[str]],
    class_name: str,
    k: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Threshold-k hard vote across N models for a single (image, class) pair.

    Each path is a (H, W, C) .npy file; the relevant channel is extracted
    per model using its declared order, binarised, then summed across models.
    A pixel is labelled lesion (1) if the sum of votes >= k.

    Args:
        prob_map_paths: One .npy path per model.
        orders:         Channel order list per model.
        class_name:     Lesion class to evaluate, e.g. 'ex'.
        k:              Minimum number of models that must vote lesion.
        threshold:      Binarisation cutoff applied per model. Default 0.5.

    Returns:
        Binary mask (uint8, 0 or 1) of shape (H, W).
    """
    if not prob_map_paths:
        raise ValueError("No probability map paths supplied.")

    n_models = len(prob_map_paths)
    if not (1 <= k <= n_models):
        raise ValueError(
            f"k={k} is out of range: must satisfy 1 <= k <= n_models ({n_models})."
        )

    binary_masks = []
    ref_shape = None

    for path, order in zip(prob_map_paths, orders):
        channel = load_channel(path, class_name, order)
        if ref_shape is None:
            ref_shape = channel.shape
        else:
            channel = resize_to(channel, ref_shape)
        binary_masks.append(binarise(channel, threshold))

    stacked = np.stack(binary_masks, axis=0)                    # (N, H, W)
    return (stacked.sum(axis=0) >= k).astype(np.uint8)          # 1 if >= k models agree


# ── Directory-level runner ─────────────────────────────────────────────────────

def collect_image_ids(prob_dirs: list[Path]) -> list[str]:
    """
    Return sorted intersection of image stems across all prob dirs.
    Stems are the full filename minus the .npy suffix, e.g. 'IDRiD_01_probs'.
    """
    id_sets = [
        {p.stem for p in d.glob("*.npy")}
        for d in prob_dirs
    ]
    common = sorted(set.intersection(*id_sets))
    if not common:
        raise RuntimeError(
            "No common .npy files found across model directories:\n"
            + "\n".join(f"  {d}: {len(s)} files" for d, s in zip(prob_dirs, id_sets))
        )
    return common


def run_threshold_k_voting(
    prob_dirs: list[Path],
    orders: list[list[str]],
    classes: list[str],
    k: int,
    threshold: float,
    save_dir: Path,
) -> None:
    """
    Iterate over every (image, class) pair and write threshold-k masks to save_dir.

    Output layout:
        <save_dir>/<class>/<image_stem>.npy
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    image_stems = collect_image_ids(prob_dirs)
    log.info(
        "Found %d common images across %d model dirs.",
        len(image_stems), len(prob_dirs),
    )

    total_written = 0

    for cls in classes:
        cls_out_dir = save_dir / cls
        cls_out_dir.mkdir(parents=True, exist_ok=True)
        log.info("Processing class '%s' -> %s", cls, cls_out_dir)

        for stem in image_stems:
            paths = [d / f"{stem}.npy" for d in prob_dirs]
            k_mask = threshold_k_hard_vote_channels(
                paths, orders, cls, k=k, threshold=threshold
            )
            np.save(cls_out_dir / f"{stem}.npy", k_mask)
            total_written += 1

        log.info("  Saved %d masks for class '%s'.", len(image_stems), cls)

    log.info("Done. Total masks written: %d", total_written)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Threshold-k hard voting across model probability map directories."
    )
    p.add_argument("--prob-dir-a", required=True, type=Path,
                   help="Prob map dir for model A (PMCNet).")
    p.add_argument("--prob-dir-b", required=True, type=Path,
                   help="Prob map dir for model B (HACDR-Net).")
    p.add_argument("--prob-dir-c", default=None, type=Path,
                   help="Prob map dir for model C (DSR-U-Net). Optional.")
    p.add_argument("--order-a", required=True,
                   help="Channel order for model A, e.g. bg,ex,ma,se,he.")
    p.add_argument("--order-b", required=True,
                   help="Channel order for model B.")
    p.add_argument("--order-c", default=None,
                   help="Channel order for model C.")
    p.add_argument("--classes", default="ex,ma,se,he",
                   help="Lesion classes to process (comma-separated). Default: ex,ma,se,he.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binarisation threshold per channel. Default: 0.5.")
    p.add_argument("--k", type=int, required=True,
                   help="Minimum number of models that must vote lesion (1 <= k <= N models).")
    p.add_argument("--save-k-dir", required=True, type=Path,
                   help="Root output directory for threshold-k hard vote masks.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prob_dirs = [args.prob_dir_a, args.prob_dir_b]
    orders    = [args.order_a.split(","), args.order_b.split(",")]

    if args.prob_dir_c is not None:
        if args.order_c is None:
            raise ValueError("--order-c must be supplied when --prob-dir-c is given.")
        prob_dirs.append(args.prob_dir_c)
        orders.append(args.order_c.split(","))

    n_models = len(prob_dirs)
    if not (1 <= args.k <= n_models):
        raise ValueError(
            f"--k={args.k} is out of range: must satisfy 1 <= k <= {n_models} (number of models)."
        )

    classes = args.classes.split(",")

    log.info(
        "Threshold-k hard voting -- k=%d/%d models, threshold=%.2f",
        args.k, n_models, args.threshold,
    )
    log.info("Classes : %s", classes)
    log.info("Output  : %s", args.save_k_dir)

    run_threshold_k_voting(
        prob_dirs=prob_dirs,
        orders=orders,
        classes=classes,
        k=args.k,
        threshold=args.threshold,
        save_dir=args.save_k_dir,
    )


if __name__ == "__main__":
    main()
