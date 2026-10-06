"""
Per-image AUPR (PR-AUC): one row per test image, one column per lesion class (MA, HE, EX, SE).
Uses the same data layout and PR_AUC helper as test.py.
"""
import csv
import os
import sys

import numpy as np
from tensorflow.keras.models import load_model
from efficientnet import *
from utils import PR_AUC, get_full_test_data, make_label

dataset = "IDRiD_from_raw"  # IDRiD
if dataset == "DDR":
    h, w = 1024, 1024
    image_dir = "./data/DDR/test/images/"
    label_dir = "./data/DDR/test/masks/"

if dataset == "IDRiD_from_raw":
    h, w = 960, 1440
    image_dir = "./data/IDRiD_from_raw/val/images/"
    label_dir = "./data/IDRiD_from_raw/val/masks/"

if dataset == "epoch":
    h, w = 960, 1440
    image_dir = "./data/epoch/test/images/"
    label_dir = "./data/epoch/test/masks/"

save_dir = f"./result/{dataset}/val/" # TODO: Change to val folder if necessary
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "per_image_aupr.csv")
log_path = os.path.join(save_dir, "test_aupr_per_image_output.txt")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


log_file = open(log_path, "w")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = TeeStream(original_stdout, log_file)
sys.stderr = TeeStream(original_stderr, log_file)

CLASS_CHANNELS = [
    (1, "AUPR_MA"),
    (2, "AUPR_HE"),
    (3, "AUPR_EX"),
    (4, "AUPR_SE"),
]

try:
    images, image_names, labels = get_full_test_data(image_dir, label_dir, h, w)
    new_labels = make_label(labels)

    model_name = "./weights/" + dataset + "/PMCNet_" + dataset + "_60.h5"
    print(f"[INFO] Loading model from: {model_name}\n")
    model = load_model(model_name, compile=False)
    probs = model.predict(images, batch_size=1, verbose=1)

    header = ["image"] + [col for _, col in CLASS_CHANNELS]
    rows = []
    for i, name in enumerate(image_names):
        basename = os.path.basename(name)
        row = [basename]
        for ch, _ in CLASS_CHANNELS:
            aupr = PR_AUC(probs[i, :, :, ch], new_labels[i, :, :, ch])
            row.append(aupr)
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[INFO] Per-image AUPR CSV written to: {csv_path}")
    print(f"[INFO] Rows: {len(rows)}, columns: {', '.join(header)}")
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
