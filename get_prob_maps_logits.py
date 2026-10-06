import argparse
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D

from utils import get_full_test_data
from efficientnet import *  # noqa: F401,F403  # needed for loading model custom layers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PMCNet inference and save per-pixel class logit maps (pre-softmax)."
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
        help="Output root. Default: ./result/<dataset>/epoch_<epoch>/logits/<test-dir>_test",
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
        help="Save grayscale PNG heatmaps of logits for one class (e.g. ex) or all classes.",
    )
    parser.add_argument(
        "--also-save-probs",
        dest="also_save_probs",
        action="store_true",
        default=False,
        help="If set, also save softmax probabilities alongside logits as {base}_probs.npy.",
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


def build_logit_model(model):
    """
    Reconstruct the final Conv2D layer with activation='linear' (i.e. no softmax),
    copy its weights from the original model, and return a new Model that outputs
    raw logits instead of softmax probabilities.

    This is necessary because PMCNet's softmax is fused directly into the final
    Conv2D layer (activation='softmax'), so it cannot be removed by simply
    cutting off a separate Softmax layer.
    """
    last_layer = model.layers[-1]

    # Sanity check: confirm the last layer is the expected fused softmax Conv2D
    last_cfg = last_layer.get_config()
    if not (isinstance(last_layer, Conv2D) and last_cfg.get("activation") == "softmax"):
        raise ValueError(
            f"Expected final layer to be Conv2D with activation='softmax', "
            f"but got: {last_layer.__class__.__name__} with config: {last_cfg}"
        )

    # Build a new Conv2D layer identical to the last layer but with linear activation
    new_cfg = last_cfg.copy()
    new_cfg["activation"] = "linear"
    new_cfg["name"] = last_cfg["name"] + "_logits"
    logit_layer = Conv2D.from_config(new_cfg)

    # Wire the new layer onto the second-to-last layer's output
    second_to_last_output = model.layers[-2].output
    logit_output = logit_layer(second_to_last_output)

    # Build the logit model
    logit_model = Model(inputs=model.input, outputs=logit_output)

    # Copy weights from the original final Conv2D into the new linear Conv2D
    # Weights are kernel and bias — identical regardless of activation function
    logit_layer.set_weights(last_layer.get_weights())

    print(f"[INFO] Logit model built. Final layer: '{logit_layer.name}' (activation=linear)")
    print(f"[INFO] Output shape: {logit_output.shape}")

    return logit_model


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
            f"./result/{args.dataset_name}/epoch_{args.ckpt_epoch}/logits/"
            f'{label_dir.split("/")[-4]}_test_new'
        )
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_path = f"./weights/{args.dataset_name}/weights/PMCNet_{args.dataset_name}_{args.ckpt_epoch}.h5"
    print(f"[INFO] Loading model from: {model_path}")
    model = load_model(model_path, compile=False)

    print(f"[INFO] Original final layer: '{model.layers[-1].name}' "
          f"activation='{model.layers[-1].get_config().get('activation')}'")

    # Build the logit model (pre-softmax outputs)
    logit_model = build_logit_model(model)

    images, image_names, _ = get_full_test_data(image_dir, label_dir, h, w)
    test_set_name = label_dir.split('/')[-4]
    print(f"[INFO] Running inference on {len(image_names)} images from {test_set_name}/test")

    logits = logit_model.predict(images, batch_size=1, verbose=1)
    print(f"[INFO] Logit output shape: {logits.shape}")  # N x H x W x 5

    # Optionally also compute softmax probabilities from logits
    if args.also_save_probs:
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        print(f"[INFO] Also saving softmax probabilities alongside logits.")

    # class_names order matches the new updated ordering (bg, ex, ma, se, he)
    class_names = ["bg", "ex", "ma", "se", "he"]
    num_classes = logits.shape[-1]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for i, image_name in enumerate(image_names):
        base = os.path.splitext(os.path.basename(image_name))[0]

        # Save raw logits
        logit_map = logits[i].astype(np.float32)  # H x W x C
        np.save(os.path.join(output_dir, f"{base}_logits.npy"), logit_map)

        # Optionally save softmax probabilities
        if args.also_save_probs:
            prob_map = probs[i].astype(np.float32)
            np.save(os.path.join(output_dir, f"{base}_probs.npy"), prob_map)

        # Optionally save PNG heatmaps of the logits
        if args.save_png is not None:
            if args.save_png == "all":
                class_indices = list(range(num_classes))
            else:
                class_indices = [class_to_idx[args.save_png]]

            for c in class_indices:
                ch = logit_map[:, :, c]
                # Normalize logits to [0, 1] for PNG visualisation
                ch_min, ch_max = ch.min(), ch.max()
                if ch_max > ch_min:
                    ch_norm = (ch - ch_min) / (ch_max - ch_min)
                else:
                    ch_norm = np.zeros_like(ch)
                ch_u8 = (ch_norm * 255.0).astype(np.uint8)
                if c < len(class_names):
                    out_name = f"{base}_logit_{class_names[c]}.png"
                else:
                    out_name = f"{base}_logit_class{c}.png"
                cv2.imwrite(os.path.join(output_dir, out_name), ch_u8)

    print(f"[INFO] Saved logit maps to: {output_dir}")
    print("[INFO] .npy format: H x W x C float32 raw logits (pre-softmax)")
    if args.also_save_probs:
        print("[INFO] Also saved softmax probability maps as {base}_probs.npy")


if __name__ == "__main__":
    main()
