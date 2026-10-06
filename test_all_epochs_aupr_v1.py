import argparse
import csv
import os
import re

import numpy as np
from tensorflow.keras.models import load_model

from efficientnet import *  # noqa: F401,F403 (needed for model loading)
from utils import PR_AUC, get_full_test_data, make_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoint epochs and save per-class AUPR CSV."
    )
    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        type=str,
        required=True,
        help="Dataset name used in checkpoint filenames (e.g. IDRiD_MAPLES_combined).",
    )
    parser.add_argument(
        "--dataset-type",
        dest="dataset_type",
        type=str,
        default="IDRiD",
        choices=["DDR", "IDRiD", "IDRiD_w_PBDA", "MAPLES-DR", "epoch"],
    )
    parser.add_argument(
        "--weights-dir",
        dest="weights_dir",
        type=str,
        default=None,
        help="Directory containing .h5 checkpoints. Defaults to ./weights/<dataset>/weights/",
    )
    parser.add_argument(
        "--output-csv",
        dest="output_csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to ./result/<dataset>/all_epochs_aupr.csv",
    )
    parser.add_argument(
        "--test-dir",
        dest="test_dir",
        type=str,
        default=None,
        help="Optional test set root directory containing images/ and masks/.",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1)
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
    if dataset_type == "epoch":
        return 960, 1440, f"./data/{dataset_name}/test/images/", f"./data/{dataset_name}/test/masks/"
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def parse_epoch_from_filename(filename):
    # Expected: PMCNet_<dataset>_<epoch>.h5
    match = re.search(r"_(\d+)\.h5$", filename)
    return int(match.group(1)) if match else None


def list_checkpoints(weights_dir, dataset_name):
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory does not exist: {weights_dir}")

    prefix = f"PMCNet_{dataset_name}_"
    ckpts = []
    for name in os.listdir(weights_dir):
        if not (name.startswith(prefix) and name.endswith(".h5")):
            continue
        epoch = parse_epoch_from_filename(name)
        if epoch is None:
            continue
        ckpts.append((epoch, os.path.join(weights_dir, name)))

    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found in {weights_dir} matching {prefix}<epoch>.h5"
        )
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def evaluate_aupr(model, images, new_labels, batch_size):
    probs = model.predict(images, batch_size=batch_size, verbose=0)
    ex = PR_AUC(probs[:, :, :, 1], new_labels[:, :, :, 1])
    ma = PR_AUC(probs[:, :, :, 2], new_labels[:, :, :, 2])
    se = PR_AUC(probs[:, :, :, 3], new_labels[:, :, :, 3])
    he = PR_AUC(probs[:, :, :, 4], new_labels[:, :, :, 4])
    mean_aupr = float(np.nanmean([ex, ma, se, he]))
    return ex, ma, se, he, mean_aupr


def main():
    args = parse_args()
    h, w, image_dir, label_dir = get_dataset_config(args.dataset_type, args.dataset_name)
    if args.test_dir is not None:
        image_dir = os.path.join(args.test_dir, "images") + "/"
        label_dir = os.path.join(args.test_dir, "masks") + "/"

    weights_dir = args.weights_dir or f"./weights/{args.dataset_name}/weights/"
    output_csv = args.output_csv or f"./result/{args.dataset_name}/{label_dir.split('/')[-3]}/all_epochs_aupr.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"[INFO] Loading test set from: {image_dir}")
    images, _, labels = get_full_test_data(image_dir, label_dir, h, w)
    new_labels = make_label(labels)

    checkpoints = list_checkpoints(weights_dir, args.dataset_name)
    print(f"[INFO] Found {len(checkpoints)} checkpoints in: {weights_dir}")

    rows = []
    for epoch, ckpt_path in checkpoints:
        print(f"[INFO] Evaluating epoch {epoch}: {ckpt_path}")
        model = load_model(ckpt_path, compile=False)
        ex, ma, se, he, mean_aupr = evaluate_aupr(model, images, new_labels, args.batch_size)
        rows.append(
            {
                "epoch": epoch,
                "EX": float(ex),
                "MA": float(ma),
                "SE": float(se),
                "HE": float(he),
                "Mean_AUPR": mean_aupr,
            }
        )
        print(
            "[INFO] AUPR - "
            f"EX: {ex:.6f}, MA: {ma:.6f}, SE: {se:.6f}, HE: {he:.6f}, Mean: {mean_aupr:.6f}"
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "EX", "MA", "SE", "HE", "Mean_AUPR"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved per-epoch AUPR CSV to: {output_csv}")


if __name__ == "__main__":
    main()
