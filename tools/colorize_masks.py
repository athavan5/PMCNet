#!/usr/bin/env python3
"""
Read grayscale masks from a directory, resize to a given shape, colorize by class
(1=Red, 2=Green, 3=Blue, 4=Pink), and save as PNGs.

Usage:
  cd /scratch/st-ipor-1/rlmacdou/PMCNet
  python tools/colorize_masks.py \
    --masks-dir /scratch/st-ipor-1/rlmacdou/PMCNet/data/IDRiD_MAPLES_combined/test/masks \
    --out-dir /scratch/st-ipor-1/rlmacdou/PMCNet/visualization_output/IDRiD_MAPLES_combined/test \
    --height 1024 --width 1024

  python tools/colorize_masks.py \
    --masks-dir /scratch/st-ipor-1/rlmacdou/PMCNet/data/IDRiD_MAPLES_v2/test/masks \
    --out-dir /scratch/st-ipor-1/rlmacdou/PMCNet/visualization_output/IDRiD_MAPLES_v2/test \
    --height 960 --width 1440

  python tools/colorize_masks.py \
    --masks-dir /arc/project/st-ipor-1/rlmacdou/input_data/MAPLES-DR-MESSIDOR/DR_Lesions_Split/train/masks \
    --out-dir /scratch/st-ipor-1/rlmacdou/PMCNet/visualization_output/MAPLES-DR/train \
    --height 512 --width 512
"""

import argparse
import os
import numpy as np
from PIL import Image

# Grayscale value -> (R, G, B)
VALUE_TO_COLOR = {
    1: (255, 0, 0),      # Red
    2: (0, 255, 0),      # Green
    3: (0, 0, 255),      # Blue
    4: (255, 192, 203),  # Pink
}

MASK_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def colorize_mask(gray: np.ndarray, value_to_color: dict) -> np.ndarray:
    """Convert grayscale mask to RGB using value_to_color mapping."""
    h, w = gray.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for value, (r, g, b) in value_to_color.items():
        mask = gray == value
        out[mask, 0] = r
        out[mask, 1] = g
        out[mask, 2] = b
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Resize and colorize grayscale masks (1=Red, 2=Green, 3=Blue, 4=Pink)."
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        required=True,
        help="Directory containing grayscale mask images.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to write colored PNGs.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output height (default: 512).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output width (default: 512).",
    )
    parser.add_argument(
        "--resample",
        type=str,
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="nearest",
        help="Resampling for resize (default: nearest, use for label masks).",
    )
    args = parser.parse_args()

    resample_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }
    resample = resample_map[args.resample]

    os.makedirs(args.out_dir, exist_ok=True)

    names = sorted(
        f
        for f in os.listdir(args.masks_dir)
        if os.path.splitext(f)[1].lower() in MASK_EXTENSIONS
    )

    for name in names:
        path = os.path.join(args.masks_dir, name)
        stem, ext = os.path.splitext(name)
        out_path = os.path.join(args.out_dir, stem + ".png")

        img = Image.open(path)
        gray = np.asarray(img)
        if gray.ndim > 2:
            gray = gray[:, :, 0]

        pil_gray = Image.fromarray(gray, mode="L")
        resized = pil_gray.resize((args.width, args.height), resample=resample)
        resized_np = np.asarray(resized)

        colored = colorize_mask(resized_np, VALUE_TO_COLOR)
        Image.fromarray(colored).save(out_path)
        print(out_path)

    print(f"Done: {len(names)} masks -> {args.out_dir}")


if __name__ == "__main__":
    main()
