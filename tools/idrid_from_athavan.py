import argparse
import os
import random
import shutil
import numpy as np
from PIL import Image

# Run instructions:
# python tools/idrid.py \
#  --out-root /scratch/st-ipor-1/rlmacdou/PMCNet/data/IDRiD \
#  --idrid-root "/arc/project/st-ipor-1/athavang/A. Segmentation/Cropped_Full_Images" \
#  --split-ratio 0.8


def prepare_idrid(idrid_root, work_root):
    """
    Create resized images and labelmaps from the original IDRiD data.

    Output (temporary working structure, original resolution preserved):
      work_root/
        images/
          <id>.png
        labelmaps/
          <id>.png
    """
    images_dir = os.path.join(idrid_root, "images")
    ex_dir = os.path.join(idrid_root, "masks", "Hard Exudates")
    he_dir = os.path.join(idrid_root, "masks", "Haemorrhages")
    ma_dir = os.path.join(idrid_root, "masks", "Microaneurysms")
    se_dir = os.path.join(idrid_root, "masks", "Soft Exudates")

    os.makedirs(work_root, exist_ok=True)
    out_img_dir = os.path.join(work_root, "images")
    out_label_dir = os.path.join(work_root, "labelmaps")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        stem = os.path.splitext(fname)[0]

        # load RGB image (keep original resolution)
        img = Image.open(os.path.join(images_dir, fname)).convert("RGB")
        width, height = img.size

        # load binary masks (0/255), keep original resolution
        def load_mask(folder):
            path = os.path.join(folder, stem + ".tif")  # adjust extension if needed
            if not os.path.exists(path):
                return None
            m = Image.open(path).convert("L")
            return np.array(m, dtype=np.uint8)

        ex = load_mask(ex_dir)
        he = load_mask(he_dir)
        ma = load_mask(ma_dir)
        se = load_mask(se_dir)

        label = np.zeros((height, width), dtype=np.uint8)  # (H, W)

        if ex is not None:
            label[ex > 0] = 1
        if he is not None:
            label[he > 0] = 2
        if ma is not None:
            label[ma > 0] = 3
        if se is not None:
            label[se > 0] = 4

        img.save(os.path.join(out_img_dir, stem + ".png"))
        Image.fromarray(label).save(os.path.join(out_label_dir, stem + ".png"))


def write_splits(out_root, work_root, split_ratio=0.8, seed=0):
    """
    Create train/test directory splits with images and masks:

      out_root/train/images
      out_root/train/masks
      out_root/test/images
      out_root/test/masks

    Source images/labelmaps are read from work_root and are not kept in the final layout.
    """
    src_img_dir = os.path.join(work_root, "images")
    src_label_dir = os.path.join(work_root, "labelmaps")

    all_ids = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(src_img_dir)
            if f.endswith(".png")
        ]
    )

    random.seed(seed)
    random.shuffle(all_ids)

    split_idx = int(len(all_ids) * split_ratio)
    train_ids = all_ids[:split_idx]
    test_ids = all_ids[split_idx:]

    # Create folder structure: train/images, train/masks, test/images, test/masks
    train_img_dir = os.path.join(out_root, "train", "images")
    train_mask_dir = os.path.join(out_root, "train", "masks")
    test_img_dir = os.path.join(out_root, "test", "images")
    test_mask_dir = os.path.join(out_root, "test", "masks")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)

    def copy_split(ids, img_target_dir, mask_target_dir):
        for sid in ids:
            src_img = os.path.join(src_img_dir, f"{sid}.png")
            src_mask = os.path.join(src_label_dir, f"{sid}.png")
            dst_img = os.path.join(img_target_dir, f"{sid}.png")
            dst_mask = os.path.join(mask_target_dir, f"{sid}.png")
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)

    copy_split(train_ids, train_img_dir, train_mask_dir)
    copy_split(test_ids, test_img_dir, test_mask_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare IDRiD dataset: resize images/masks and create train/test splits."
    )
    parser.add_argument(
        "--idrid-root",
        type=str,
        default="/arc/project/st-ipor-1/athavang/A. Segmentation/Cropped_Full_Images",
        help="Root folder of the original IDRiD dataset (with images/ and masks/).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Output root folder where train/ and test/ splits will be created.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Fraction of samples to use for training (rest go to test).",
    )
    args = parser.parse_args()

    # Temporary working directory to hold full images/labelmaps before splitting
    work_root = os.path.join(args.out_root, "_tmp_full")

    prepare_idrid(args.idrid_root, work_root)
    write_splits(args.out_root, work_root, split_ratio=args.split_ratio)

    # Remove temporary working data so final layout only has train/ and test/
    if os.path.exists(work_root):
        shutil.rmtree(work_root)


if __name__ == "__main__":
    main()
