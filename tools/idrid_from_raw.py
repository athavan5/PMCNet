import argparse
import os
import numpy as np
from PIL import Image

# Run instructions:
# python tools/idrid_from_raw.py \
#   --images-dir "/arc/project/st-ipor-1/athavang/A. Segmentation/1. Original Images" \
#   --groundtruths-dir "/arc/project/st-ipor-1/athavang/A. Segmentation/2. All Segmentation Groundtruths" \
#   --out-root /scratch/st-ipor-1/rlmacdou/PMCNet/data/IDRiD_from_raw
#   --val-frac 0.5  # optional; split Testing Set into val/test (deterministic)
#
# Input structure:
#   images-dir/
#     a. Training Set/   -> images (*.jpg, *.png, etc.)
#     b. Testing Set/    -> images
#   groundtruths-dir/
#     a. Training Set/
#       1. Microaneurysms/   -> per-image masks (ma -> grayscale 3)
#       2. Haemorrhages/    -> (he -> grayscale 2)
#       3. Hard Exudates/  -> (ex -> grayscale 1)
#       4. Soft Exudates/   -> (se -> grayscale 4)
#     b. Testing Set/
#       (same subfolders)
#
# Splits are preserved: Training Set -> train, Testing Set -> test by default (no reshuffling).
# Optionally, the Testing Set can be split into validation and test subsets.
# Output:
#   out_root/train/images, out_root/train/masks
#   out_root/test/images,  out_root/test/masks
#   out_root/val/images,   out_root/val/masks   (optional)

# Class folder names under each split (as they exist on disk),
# and their grayscale value in the combined mask.
# IDRiD uses numbered class folders, e.g. "1. Microaneurysms".
# Order matches idrid_from_athavan.py (ex, he, ma, se) so overlaps resolve the same way.
GT_SUBFOLDERS = [
    ("3. Hard Exudates", 1),
    ("2. Haemorrhages", 2),
    ("1. Microaneurysms", 3),
    ("4. Soft Exudates", 4),
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
MASK_EXTENSIONS = (".tif", ".tiff", ".png")


def find_mask_path(folder, stem):
    """
    Return path to mask for a given image stem if it exists, else None.

    IDRiD groundtruth masks are commonly named like:
      IDRiD_01_EX.tif, IDRiD_01_HE.tif, IDRiD_01_MA.tif, IDRiD_01_SE.tif
    within their respective class folders.

    We therefore:
    - Prefer an exact match: <stem>.<ext>
    - Otherwise accept any file in the folder whose basename contains <stem>
      (e.g. <stem>_HE.tif), restricted to known mask extensions.
    """
    # 1) Exact match (fast path)
    for ext in MASK_EXTENSIONS:
        path = os.path.join(folder, stem + ext)
        if os.path.exists(path):
            return path

    # 2) Fuzzy match within folder (deterministic)
    try:
        candidates = []
        for fname in os.listdir(folder):
            base, ext = os.path.splitext(fname)
            if ext.lower() not in MASK_EXTENSIONS:
                continue
            if stem not in base:
                continue
            candidates.append(fname)
        if not candidates:
            return None
        candidates.sort()
        return os.path.join(folder, candidates[0])
    except FileNotFoundError:
        return None


def find_image_path(images_dir, stem):
    """Return path to image for given stem if it exists, else None."""
    for ext in IMAGE_EXTENSIONS:
        path = os.path.join(images_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def prepare_split(images_dir, groundtruths_dir, out_images_dir, out_masks_dir, stems=None):
    """
    Process one split (e.g. Training Set): copy images and build combined labelmaps.
    Images come from images_dir; per-class masks from groundtruths_dir subfolders.
    If stems is provided, only those image stems are processed (deterministic subset).
    """
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    if stems is None:
        stems = [
            os.path.splitext(f)[0]
            for f in sorted(os.listdir(images_dir))
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

    gt_dirs = {
        name: os.path.join(groundtruths_dir, name)
        for name, _ in GT_SUBFOLDERS
    }

    for stem in stems:
        img_path = find_image_path(images_dir, stem)
        if img_path is None:
            continue

        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        label = np.zeros((height, width), dtype=np.uint8)
        for folder_name, value in GT_SUBFOLDERS:
            folder = gt_dirs.get(folder_name)
            if folder is None or not os.path.isdir(folder):
                print(f"[WARNING] Folder {folder_name} not found in {groundtruths_dir}")
                continue
            path = find_mask_path(folder, stem)
            if path is None:
                print(f"[WARNING] Mask for {stem} not found in {folder}")
                continue
            m = Image.open(path).convert("L")
            arr = np.array(m, dtype=np.uint8)
            label[arr > 0] = value

        img.save(os.path.join(out_images_dir, stem + ".png"))
        Image.fromarray(label).save(os.path.join(out_masks_dir, stem + ".png"))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare IDRiD dataset from images + groundtruths dirs; "
            "preserve train split and optionally split Testing Set into val + test."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Root folder containing 'a. Training Set' and 'b. Testing Set' image subfolders.",
    )
    parser.add_argument(
        "--groundtruths-dir",
        type=str,
        required=True,
        help="Root folder containing 'a. Training Set' and 'b. Testing Set' with Hard Exudates, Haemorrhages, Microaneurysms, Soft Exudates subfolders.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Output root folder where train/, (optional) val/, and test/ will be created.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.0,
        help=(
            "Fraction of the Testing Set to put into `val/` (deterministic by sorted stem order). "
            "Set to 0.0 to keep all Testing Set in `test/` (default)."
        ),
    )
    args = parser.parse_args()

    train_images_src = os.path.join(args.images_dir, "a. Training Set")
    test_images_src = os.path.join(args.images_dir, "b. Testing Set")
    train_gt_src = os.path.join(args.groundtruths_dir, "a. Training Set")
    test_gt_src = os.path.join(args.groundtruths_dir, "b. Testing Set")

    train_img_dir = os.path.join(args.out_root, "train", "images")
    train_mask_dir = os.path.join(args.out_root, "train", "masks")
    val_img_dir = os.path.join(args.out_root, "val", "images")
    val_mask_dir = os.path.join(args.out_root, "val", "masks")
    test_img_dir = os.path.join(args.out_root, "test", "images")
    test_mask_dir = os.path.join(args.out_root, "test", "masks")

    print(f"[INFO] Preparing train split from {train_images_src} and {train_gt_src}")
    if os.path.isdir(train_images_src) and os.path.isdir(train_gt_src):
        prepare_split(train_images_src, train_gt_src, train_img_dir, train_mask_dir)

    if args.val_frac < 0.0 or args.val_frac >= 1.0:
        raise ValueError("--val-frac must be in the range [0.0, 1.0).")

    # Deterministically split Testing Set into val/test by stem order (no reshuffling).
    test_stems = [
        os.path.splitext(f)[0]
        for f in sorted(os.listdir(test_images_src))
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if args.val_frac == 0.0:
        print(f"[INFO] Preparing test split (all Testing Set) from {test_images_src} and {test_gt_src}")
        if os.path.isdir(test_images_src) and os.path.isdir(test_gt_src):
            prepare_split(test_images_src, test_gt_src, test_img_dir, test_mask_dir, stems=test_stems)
    else:
        val_count = int(len(test_stems) * args.val_frac)
        if val_count <= 0:
            print(
                f"[WARNING] --val-frac={args.val_frac} produced 0 val samples for {len(test_stems)} testing images; "
                "keeping all samples in `test/`."
            )
            if os.path.isdir(test_images_src) and os.path.isdir(test_gt_src):
                prepare_split(test_images_src, test_gt_src, test_img_dir, test_mask_dir, stems=test_stems)
        else:
            val_stems = test_stems[:val_count]
            test_stems_subset = test_stems[val_count:]

            print(
                f"[INFO] Preparing val split ({len(val_stems)} samples) from {test_images_src} and {test_gt_src}"
            )
            if os.path.isdir(test_images_src) and os.path.isdir(test_gt_src):
                prepare_split(
                    test_images_src,
                    test_gt_src,
                    val_img_dir,
                    val_mask_dir,
                    stems=val_stems,
                )

            print(
                f"[INFO] Preparing test split ({len(test_stems_subset)} samples) from {test_images_src} and {test_gt_src}"
            )
            if os.path.isdir(test_images_src) and os.path.isdir(test_gt_src):
                prepare_split(
                    test_images_src,
                    test_gt_src,
                    test_img_dir,
                    test_mask_dir,
                    stems=test_stems_subset,
                )

    print(f"[INFO] Done preparing splits")

if __name__ == "__main__":
    main()
