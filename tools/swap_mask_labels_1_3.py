#!/usr/bin/env python3
"""
Swap grayscale label values 1 <-> 3 in every mask image under a folder.

Typical use: fix MA/EX channel order in IDRiD-style single-channel label PNGs/TIFs.

Usage:
  python tools/swap_mask_labels_1_3.py /path/to/masks
  python tools/swap_mask_labels_1_3.py /path/to/masks --output-dir /path/to/out
  python tools/swap_mask_labels_1_3.py /path/to/masks --recursive --output-dir /path/to/out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp"}


def swap_1_and_3(arr: np.ndarray) -> np.ndarray:
    """Pixels equal to 1 become 3 and pixels equal to 3 become 1; all others unchanged."""
    out = np.array(arr, copy=True)
    is_one = out == 1
    is_three = out == 3
    out[is_one] = 3
    out[is_three] = 1
    return out


def process_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return swap_1_and_3(arr)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return swap_1_and_3(arr[:, :, 0])[..., np.newaxis]
    raise ValueError(
        f"Expected a single-channel (H, W) mask or (H, W, 1); got shape {arr.shape}"
    )


def process_image(in_path: Path, out_path: Path) -> None:
    img = Image.open(in_path)
    arr = np.asarray(img)
    new_arr = process_array(arr)
    new_arr = np.asarray(new_arr, dtype=arr.dtype)
    out_img = Image.fromarray(new_arr, mode=img.mode) if new_arr.ndim == 2 else Image.fromarray(new_arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)


def iter_mask_files(root: Path, recursive: bool):
    if recursive:
        paths = sorted(root.rglob("*"))
    else:
        paths = sorted(root.iterdir())
    for path in paths:
        if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swap grayscale values 1 and 3 in all mask images in a folder."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing mask images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If set, write here (mirrors subpaths when --recursive). "
        "If omitted, files are overwritten in place.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images in subfolders as well.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without writing.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    out_root = Path(args.output_dir).resolve() if args.output_dir else None
    if out_root is not None and out_root == input_dir and not args.dry_run:
        raise SystemExit("--output-dir must differ from input_dir when both are the same path.")

    count = 0
    for path in iter_mask_files(input_dir, args.recursive):
        if out_root is None:
            out_path = path
        else:
            out_path = out_root / path.relative_to(input_dir)

        if args.dry_run:
            print(f"would process: {path} -> {out_path}")
        else:
            process_image(path, out_path)
            print(out_path)
        count += 1

    if count == 0:
        print("No matching mask files found.")
    elif args.dry_run:
        print(f"Would process {count} file(s).")


if __name__ == "__main__":
    main()
