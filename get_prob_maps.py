import argparse
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from utils import get_full_test_data
from efficientnet import *  # noqa: F401,F403  # needed for loading model custom layers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PMCNet inference and save per-pixel class probability maps."
    )
    parser.add_argument("--ckpt_epoch", type=int, default=60)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, required=True)
    parser.add_argument(
        "--dataset-type",
        dest="dataset_type",
        type=str,
        default="IDRiD",
        choices=["DDR", "IDRiD", "IDRiD_w_PBDA", "MAPLES-DR", "IDRiD_MAPLES_combined", "epoch"],
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Output root. Default: ./result/<dataset>/epoch_<epoch>/prob_maps/<test-dir>_test",
    )
    parser.add_argument(
        "--test-dir",
        dest="test_dir",
        type=str,
        default=None,
        help="Override test data root: uses <test-dir>/images/ and <test-dir>/masks/. "
        "If omitted, image and mask paths come from get_dataset_config(dataset_type, dataset_name).",
    )
    parser.add_argument(
        "--save-png",
        dest="save_png",
        type=str,
        default=None,
        choices=["all", "bg", "ex", "ma", "se", "he"],
        help="Save grayscale PNG heatmaps for one class (e.g. ex) or all classes.",
    )
    return parser.parse_args()


def get_dataset_config(dataset_type, dataset_name):
    if dataset_type == "DDR":
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "IDRiD":
        return 960, 1440, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "IDRiD_w_PBDA":
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "MAPLES-DR":
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "IDRiD_MAPLES_combined":
        return 1024, 1024, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    if dataset_type == "epoch":
        return 960, 1440, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    raise ValueError(f"Unsupported dataset: {dataset_type}")


def main():
    args = parse_args()
    h, w, default_image_dir, default_label_dir = get_dataset_config(
        args.dataset_type, args.dataset_name
    )
    if args.test_dir is not None:
        test_root = os.path.normpath(args.test_dir)
        image_dir = os.path.join(test_root, "images/")
        label_dir = os.path.join(test_root, "masks/")
    else:
        image_dir, label_dir = default_image_dir, default_label_dir

    if args.output_dir is None:
        output_dir = (
            f"./result/{args.dataset_name}/epoch_{args.ckpt_epoch}/prob_maps/"
            f'{label_dir.split("/")[-4]}_test'
        )
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_path = f"./weights/{args.dataset_name}/weights/PMCNet_{args.dataset_name}_{args.ckpt_epoch}.h5"
    print(f"[INFO] Loading model from: {model_path}")
    model = load_model(model_path, compile=False)

    images, image_names, _ = get_full_test_data(image_dir, label_dir, h, w)
    print(f"[INFO] Running inference on {len(image_names)} images from {label_dir.split('/')[-4]}/test")
    probs = model.predict(images, batch_size=1, verbose=1)
    print(f"[INFO] Prediction shape: {probs.shape}")

    # OLD version: class_names = ["bg", "ex", "he", "ma", "se"]
    # NEW version: class_names = ["bg", "ex", "ma", "se", "he"]
    class_names = ["bg", "ex", "ma", "se", "he"]
    num_classes = probs.shape[-1]

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for i, image_name in enumerate(image_names):
        base = os.path.splitext(os.path.basename(image_name))[0]
        prob_map = probs[i].astype(np.float32)  # H x W x C
        np.save(os.path.join(output_dir, f"{base}_probs.npy"), prob_map)

        if args.save_png is not None:
            if args.save_png == "all":
                class_indices = list(range(num_classes))
            else:
                class_indices = [class_to_idx[args.save_png]]

            for c in class_indices:
                ch = np.clip(prob_map[:, :, c], 0.0, 1.0)
                ch_u8 = (ch * 255.0).astype(np.uint8)
                if c < len(class_names):
                    out_name = f"{base}_prob_{class_names[c]}.png"
                else:
                    out_name = f"{base}_prob_class{c}.png"
                cv2.imwrite(os.path.join(output_dir, out_name), ch_u8)

    print(f"[INFO] Saved probability maps to: {output_dir}")
    print("[INFO] .npy format: H x W x C float32 softmax probabilities")


if __name__ == "__main__":
    main()
