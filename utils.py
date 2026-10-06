from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from matplotlib import pyplot as plt
import cv2
from sklearn.metrics import precision_recall_curve, auc
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------------------------------
# Evaluator — mirrors metrics.py intersect_and_union / total_area_to_metrics
#
# Instead of computing Dice/IoU per image and averaging (the old approach),
# this accumulates raw pixel counts (intersection, union, pred area, label
# area) across ALL images, then divides once at the end.  This is the same
# "global accumulation" strategy used in HACDR-Net's metrics.py and avoids
# inflated scores on images that contain no positive pixels for a class.
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Accumulate per-image intersection / union counts and compute
    dataset-level Dice and IoU at the end, matching the approach in
    metrics.py (intersect_and_union → total_area_to_metrics).

    Usage (binary, one class at a time — same call-site as before):
        ev = Evaluator()
        for pred_bin, gt_bin in zip(preds, gts):
            ev.update(pred_bin, gt_bin)
        dice, iou = ev.show()   # returns percentages, rounded to 2 d.p.
    """

    def __init__(self):
        self.total_intersect = 0.0
        self.total_union = 0.0
        self.total_pred = 0.0
        self.total_label = 0.0

    def _intersect_and_union(self, pred_bin, gt_bin):
        """
        Compute raw pixel counts for one image.

        pred_bin : 2-D binary array (0/1 float or bool) — hard prediction
        gt_bin   : 2-D binary array (0/1 float or bool) — ground truth

        Mirrors metrics.py intersect_and_union but operates directly on
        pre-binarised numpy arrays instead of going through torch.histc,
        since we are in the binary (single-class) case.
        """
        pred = pred_bin.astype(bool)
        gt   = gt_bin.astype(bool)

        intersect      = np.logical_and(pred, gt).sum()
        area_pred      = pred.sum()
        area_label     = gt.sum()
        area_union     = area_pred + area_label - intersect
        return float(intersect), float(area_union), float(area_pred), float(area_label)

    def update(self, pred_bin, gt_bin):
        """Accumulate counts for one image."""
        i, u, p, l = self._intersect_and_union(pred_bin, gt_bin)
        self.total_intersect += i
        self.total_union     += u
        self.total_pred      += p
        self.total_label     += l

    def show(self):
        """
        Return (dice %, iou %) computed over the full accumulated dataset,
        matching metrics.py total_area_to_metrics with metric='mDice'.

        dice = 2 * total_intersect / (total_pred + total_label)
        iou  = total_intersect / total_union
        """
        eps = 1e-6
        dice = (2.0 * self.total_intersect) / (self.total_pred + self.total_label + eps)
        iou  = self.total_intersect / (self.total_union + eps)
        return round(dice * 100, 2), round(iou * 100, 2)


# ---------------------------------------------------------------------------
# Utility helpers (unchanged from original utils.py)
# ---------------------------------------------------------------------------

def resize_label(masks, w, h):
    new_mask = []
    for i in range(len(masks)):
        label = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
        new_mask.append(label)
    return np.array(new_mask)


def make_label(label, num_class=5):
    # NOTE: changed from 3 to num_class; default is 5 for IDRiD
    new = to_categorical(label, num_class)
    return new


def dataGenerator(batch_size, target_size, train_path,
                  image_folder='images', mask_folder='masks',
                  seed=100, image_color_mode="rgb", mask_color_mode="grayscale"):

    image_datagen = ImageDataGenerator(fill_mode='nearest')
    mask_datagen  = ImageDataGenerator(fill_mode='nearest')

    image_generator = image_datagen.flow_from_directory(
        train_path, classes=[image_folder], class_mode=None,
        color_mode=image_color_mode, target_size=target_size,
        batch_size=batch_size, seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path, classes=[mask_folder], class_mode=None,
        color_mode=mask_color_mode, target_size=target_size,
        batch_size=batch_size, seed=seed)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        labels = make_label(mask[:, :, :, 0])
        yield (img, labels)


def get_full_test_data(imgs_dir, label_dir, h, w):
    iter_tot    = 0
    image_names = sorted(os.listdir(imgs_dir))
    images_num  = len(image_names)
    images  = np.empty((images_num, h, w, 3))
    labels  = np.empty((images_num, h, w))

    for name in image_names:
        image = Image.open(imgs_dir + name)
        image = np.asarray(image)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)

        label = Image.open(label_dir + name[:-4] + '.png')
        label = np.asarray(label)
        if label.ndim == 3:
            if not np.array_equal(label[:, :, 0], label[:, :, 1]) or \
               not np.array_equal(label[:, :, 0], label[:, :, 2]):
                raise ValueError(
                    f"Non-identical RGB channels in label mask: "
                    f"{label_dir + name[:-4] + '.png'}"
                )
            label = label[:, :, 0]
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        images[iter_tot] = image
        labels[iter_tot] = label
        iter_tot += 1

    return images, image_names, labels


def get_scores(pred, label):
    y_pred = pred.flatten()
    y_true = label.flatten()
    return y_pred, y_true


def PR_AUC(pred, label):
    y_pred, y_true = get_scores(pred, label)
    if np.sum(y_true) == 0:
        # No positive samples: PR curve is undefined, return 0 to avoid NaN
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    return pr_auc if np.isfinite(pr_auc) else 0.0


def save_results(images, image_names, probs, lables, save_dir, h, w):
    os.makedirs(save_dir, exist_ok=True)
    predictions = np.argmax(probs, axis=-1)
    for i, name in enumerate(image_names):
        pred  = predictions[i]
        label = lables[i]

        label_vis = np.zeros((h, w, 3), np.uint8)
        label_vis[label == 1] = [255,   0,   0]   # Red
        label_vis[label == 2] = [  0, 255,   0]   # Green
        label_vis[label == 3] = [  0,   0, 255]   # Blue
        label_vis[label == 4] = [255,   0, 255]   # Magenta

        pred_vis = np.zeros((h, w, 3), np.uint8)
        pred_vis[pred == 1] = [255,   0,   0]
        pred_vis[pred == 2] = [  0, 255,   0]
        pred_vis[pred == 3] = [  0,   0, 255]
        pred_vis[pred == 4] = [255,   0, 255]

        cv2.imwrite(
            os.path.join(save_dir,
                         os.path.splitext(os.path.basename(name))[0] + '_pred.png'),
            pred_vis[:, :, ::-1]
        )
    print(f"[INFO] Results saved to: {save_dir}")


def Multi_dataGenerator(batch_size, target_size, train_path,
                        image_folder='images', mask_folder='masks',
                        seed=1000, image_color_mode="rgb",
                        mask_color_mode="grayscale"):
    w, h = 1440, 960
    image_datagen = ImageDataGenerator(fill_mode='nearest')
    mask_datagen  = ImageDataGenerator(fill_mode='nearest')

    image_generator = image_datagen.flow_from_directory(
        train_path, classes=[image_folder], class_mode=None,
        color_mode=image_color_mode, target_size=target_size,
        batch_size=batch_size, seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path, classes=[mask_folder], class_mode=None,
        color_mode=mask_color_mode, target_size=target_size,
        batch_size=batch_size, seed=seed)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        labels   = make_label(mask[:, :, :, 0])
        label_2  = resize_label(mask[:, :, :, 0], w // 2,  h // 2)
        label_4  = resize_label(mask[:, :, :, 0], w // 4,  h // 4)
        label_8  = resize_label(mask[:, :, :, 0], w // 8,  h // 8)
        label_16 = resize_label(mask[:, :, :, 0], w // 16, h // 16)
        label_2  = make_label(label_2)
        label_4  = make_label(label_4)
        label_8  = make_label(label_8)
        label_16 = make_label(label_16)
        yield (img, [labels, label_2, label_4, label_8])
