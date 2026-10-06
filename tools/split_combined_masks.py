"""
Split a flat MAPLES-DR export into train / val / test with combined lesion masks.

Expects ``source_dir`` to match ``export_masks_and_fundus.py`` output: subfolders
``fundus``, ``exudates``, ``microaneurysms``, ``hemorrhages``, ``cottonWoolSpots``.

Each split gets::

    <split>/images/   — fundus files (same basename and extension as source)
    <split>/masks/    — single-channel PNG, grayscale label per lesion type:
        1 microaneurysms, 2 hemorrhages, 3 exudates, 4 cottonWoolSpots

Where masks overlap, later types in that order overwrite earlier pixels.

Default split (after shuffling with ``--seed``)::

    - 104 images with any exudate foreground: 70 train, 17 val, 17 test
    - 94 images without exudates: 63 train; remaining 31 split 15 val / 16 test

Example::

    python split_combined_masks.py --source-dir ./MAPLES_DR --output-dir ./splits
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# (grayscale value, export subfolder name) — applied in order; later wins on overlap.
COMBINED_LABEL_FOLDERS: tuple[tuple[int, str], ...] = (
    (1, "microaneurysms"),
    (2, "hemorrhages"),
    (3, "exudates"),
    (4, "cottonWoolSpots"),
)


def _find_by_stem(folder: Path, stem: str) -> Path | None:
    if not folder.is_dir():
        return None
    for ext in IMAGE_EXTENSIONS:
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _list_fundus_stems(fundus_dir: Path) -> list[str]:
    stems: set[str] = set()
    for ext in IMAGE_EXTENSIONS:
        for p in fundus_dir.glob(f"*{ext}"):
            if p.is_file():
                stems.add(p.stem)
    return sorted(stems)


def _mask_has_foreground(path: Path) -> bool:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return bool(np.any(arr > 0))


def _has_exudate(source: Path, stem: str) -> bool:
    p = _find_by_stem(source / "exudates", stem)
    if p is None:
        return False
    return _mask_has_foreground(p)


def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 0


def build_combined_mask(source: Path, stem: str) -> np.ndarray:
    """Single-channel uint8 mask with labels 0–4."""
    combined: np.ndarray | None = None
    for label, sub in COMBINED_LABEL_FOLDERS:
        p = _find_by_stem(source / sub, stem)
        if p is None:
            continue
        m = _load_binary_mask(p)
        if combined is None:
            combined = np.zeros(m.shape, dtype=np.uint8)
        elif m.shape != combined.shape:
            raise ValueError(
                f"Shape mismatch for {stem}: {sub} {m.shape} vs combined {combined.shape}"
            )
        combined[m] = label
    if combined is None:
        fp = _find_by_stem(source / "fundus", stem)
        if fp is None:
            raise FileNotFoundError(f"No fundus or lesion masks for stem {stem!r}")
        with Image.open(fp) as im:
            w, h = im.size
        combined = np.zeros((h, w), dtype=np.uint8)
    return combined


def _copy_fundus(source_fundus: Path, stem: str, dest_images: Path) -> None:
    src = _find_by_stem(source_fundus, stem)
    if src is None:
        raise FileNotFoundError(f"No fundus image for stem {stem!r} under {source_fundus}")
    dest_images.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_images / src.name)


def _write_mask_png(mask: np.ndarray, dest_masks: Path, stem: str) -> None:
    dest_masks.mkdir(parents=True, exist_ok=True)
    out = dest_masks / f"{stem}.png"
    Image.fromarray(mask, mode="L").save(out)


def split_stems(
    stems_with_exudate: list[str],
    stems_without_exudate: list[str],
    *,
    n_train_ex: int,
    n_val_ex: int,
    n_test_ex: int,
    n_train_no: int,
    seed: int,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    ex = np.array(sorted(stems_with_exudate))
    rng.shuffle(ex)
    no = np.array(sorted(stems_without_exudate))
    rng.shuffle(no)

    need_ex = n_train_ex + n_val_ex + n_test_ex
    if len(ex) != need_ex:
        raise ValueError(
            f"Expected {need_ex} stems with exudate foreground, found {len(ex)}. "
            "Adjust counts or use --no-strict-expect (see script help)."
        )

    rest_no = len(no) - n_train_no
    if rest_no < 0:
        raise ValueError(
            f"Not enough non-exudate stems: have {len(no)}, need at least {n_train_no} for train."
        )
    n_val_no = rest_no // 2
    n_test_no = rest_no - n_val_no
    need_no = n_train_no + n_val_no + n_test_no
    if len(no) != need_no:
        raise ValueError(
            f"Expected {need_no} stems without exudate, found {len(no)}. "
            "Adjust --n-train-no-exudate or expectations."
        )

    ex_list = ex.tolist()
    no_list = no.tolist()
    i = 0
    train_ex = ex_list[i : i + n_train_ex]
    i += n_train_ex
    val_ex = ex_list[i : i + n_val_ex]
    i += n_val_ex
    test_ex = ex_list[i : i + n_test_ex]

    j = 0
    train_no = no_list[j : j + n_train_no]
    j += n_train_no
    val_no = no_list[j : j + n_val_no]
    j += n_val_no
    test_no = no_list[j : j + n_test_no]

    return {
        "train": train_ex + train_no,
        "val": val_ex + val_no,
        "test": test_ex + test_no,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True, help="Flat export root (fundus/, exudates/, …).")
    p.add_argument("--output-dir", type=Path, required=True, help="Root for train/, val/, test/.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling before splitting.")
    p.add_argument(
        "--n-train-exudate",
        type=int,
        default=70,
        help="Train size among images with exudate foreground (default: 70).",
    )
    p.add_argument("--n-val-exudate", type=int, default=17, help="Val size for exudate set (default: 17).")
    p.add_argument("--n-test-exudate", type=int, default=17, help="Test size for exudate set (default: 17).")
    p.add_argument(
        "--n-train-no-exudate",
        type=int,
        default=63,
        help="Train size among images without exudate foreground (default: 63).",
    )
    p.add_argument(
        "--no-strict-expect",
        action="store_true",
        help="Do not require exudate pool = train+val+test exudate counts or "
        "non-exudate pool = train + equal val/test remainder; use min available per bucket instead.",
    )
    args = p.parse_args()

    source = args.source_dir.resolve()
    fundus_dir = source / "fundus"
    if not fundus_dir.is_dir():
        raise SystemExit(f"Missing fundus folder: {fundus_dir}")

    stems = _list_fundus_stems(fundus_dir)
    if not stems:
        raise SystemExit(f"No fundus images found under {fundus_dir}")

    with_ex: list[str] = []
    without_ex: list[str] = []
    for stem in stems:
        if _has_exudate(source, stem):
            with_ex.append(stem)
        else:
            without_ex.append(stem)

    n_train_ex, n_val_ex, n_test_ex = args.n_train_exudate, args.n_val_exudate, args.n_test_exudate
    n_train_no = args.n_train_no_exudate

    if args.no_strict_expect:
        rng = np.random.default_rng(args.seed)
        ex = np.array(sorted(with_ex))
        rng.shuffle(ex)
        no = np.array(sorted(without_ex))
        rng.shuffle(no)

        def take(arr: np.ndarray, start: int, n: int) -> list[str]:
            return arr[start : start + n].tolist()

        e0 = 0
        train_ex = take(ex, e0, min(n_train_ex, len(ex) - e0))
        e0 += len(train_ex)
        val_ex = take(ex, e0, min(n_val_ex, len(ex) - e0))
        e0 += len(val_ex)
        test_ex = take(ex, e0, min(n_test_ex, len(ex) - e0))

        n0 = 0
        train_no = take(no, n0, min(n_train_no, len(no) - n0))
        n0 += len(train_no)
        rest = len(no) - n0
        n_val_no = rest // 2
        n_test_no = rest - n_val_no
        val_no = take(no, n0, n_val_no)
        n0 += len(val_no)
        test_no = take(no, n0, n_test_no)

        by_split = {
            "train": train_ex + train_no,
            "val": val_ex + val_no,
            "test": test_ex + test_no,
        }
    else:
        rest_no = len(without_ex) - n_train_no
        n_val_no = rest_no // 2
        n_test_no = rest_no - n_val_no
        by_split = split_stems(
            with_ex,
            without_ex,
            n_train_ex=n_train_ex,
            n_val_ex=n_val_ex,
            n_test_ex=n_test_ex,
            n_train_no=n_train_no,
            seed=args.seed,
        )

    out_root = args.output_dir.resolve()
    for split_name, stem_list in by_split.items():
        img_dir = out_root / split_name / "images"
        msk_dir = out_root / split_name / "masks"
        for stem in stem_list:
            _copy_fundus(source / "fundus", stem, img_dir)
            mask = build_combined_mask(source, stem)
            _write_mask_png(mask, msk_dir, stem)

    print(f"Source: {source}")
    print(f"Output: {out_root}")
    print(f"With exudate: {len(with_ex)}, without: {len(without_ex)}")
    for name in ("train", "val", "test"):
        print(f"  {name}: {len(by_split[name])} samples")


if __name__ == "__main__":
    main()
