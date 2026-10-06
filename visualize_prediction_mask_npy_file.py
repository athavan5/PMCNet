"""
save_masks.py
-------------
Convert .npy retinal prediction masks to clean PNG segmentation images
suitable for direct comparison with ground truth masks.

Output PNGs use the exact RGBA colour map below — black background,
with each lesion class rendered in its designated colour.

Usage
-----
    python save_masks.py --input  /path/to/npy/folder \
                         --output /path/to/output/folder

Arguments
---------
  --input      Directory containing .npy prediction mask files (required)
  --output     Directory where PNG masks will be saved (required)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# RGBA colour map — edit values here to change class colours.
# Format: class_index: (R, G, B, A)  — all values 0-255.
COLOR_MAP = {
    0: (0,   0,   0,   255),    # Background      - black (change to 0,0,0,0 for transparent background)
    1: (255, 230, 0,   230),  # Hard Exudates   - yellow
#    2: (255, 0,   0,   230),  # Microaneurysms  - red
#    3: (30,  120, 255, 230),  # Soft Exudates   - blue
#    4: (0,   220, 0,   230),  # Hemorrhages     - green
}


def save_mask(arr: np.ndarray, out_path: Path) -> None:
    """Save an RGBA segmentation PNG using COLOR_MAP."""
    
    #uncomment this out if you're geting just the hard exudate mask
    masked = arr.copy()
    masked[:, :, [2, 3, 4]] = 0   # zero out all classes except background (0) and Hard Exudates (1)
    seg = masked.argmax(axis=-1)
    
    #uncomment this if you're geting all lesion masks
    #seg = arr.argmax(axis=-1)          # (H, W) int
    
    h, w = seg.shape

    # Build RGBA output array — default to black/transparent background
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        rgba[seg == class_idx] = color

    Image.fromarray(rgba, mode="RGBA").save(out_path)


def process_folder(input_dir: Path, output_dir: Path) -> None:
    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(npy_files)} .npy file(s) in '{input_dir}'")
    print(f"Saving masks to '{output_dir}'\n")

    for i, npy_path in enumerate(npy_files, 1):
        print(f"[{i}/{len(npy_files)}] {npy_path.name}")
        try:
            arr = np.load(str(npy_path))
        except Exception as e:
            print(f"  ✗ Could not load: {e}")
            continue

        if arr.ndim != 3:
            print(f"  ✗ Unexpected shape {arr.shape} — expected (H, W, C). Skipping.")
            continue
        
        #uncomment this if you're geting all lesion masks
        #out_path = output_dir / f"{npy_path.stem}_mask.png"
        
        #uncomment this if you're geting just the exudate masks
        out_path = output_dir / f"{npy_path.stem}_EX_mask.png"
        
        try:
            save_mask(arr, out_path)
            print(f"  ✓ {out_path.name}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save RGBA prediction mask PNGs from .npy files.")
    p.add_argument("--input",  required=True, type=Path, help="Folder containing .npy files")
    p.add_argument("--output", required=True, type=Path, help="Folder to save mask PNGs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input.is_dir():
        print(f"Error: not a directory: {args.input}")
        sys.exit(1)
    process_folder(args.input, args.output)
