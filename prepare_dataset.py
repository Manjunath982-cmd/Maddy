#!/usr/bin/env python
"""Automatically assemble a 1-k image food-detection dataset.

The script does three things:
1. Downloads four lightweight classification datasets from Kaggle (≈300 MB total).
2. Picks ~250 images per target class (indian_food, foreign_food, fruit, vegetable).
3. Generates YOLO-format labels that span the *full image* (object occupies most area).

The labels are an acceptable proxy for small experiments because classification
images are typically centered and contain a single object on a plain background.
For production-grade detection you should hand-annotate tighter boxes.

Prerequisites:
• Create a Kaggle account → *Account* → *API* → “Create New Token”.  
  Place resulting kaggle.json in ~/.kaggle/ (or set the KAGGLE_USERNAME / KAGGLE_KEY env vars).
• `pip install kaggle opencv-python tqdm` (already included in requirements.txt)

Usage:
    python prepare_dataset.py --dest data
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path
from random import sample, seed

import cv2  # type: ignore
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASETS = {
    # (kaggle slug, folder_inside_archive, list_of_subfolders_to_use, class_id)
    "indian_food": (
        "gargmanas/indian-food-images-dataset",  # 310 images across 35 dishes
        "indian_food_images",  # root inside zip
        [],  # use all
        0,
    ),
    "foreign_food": (
        "kmader/food41",  # Food-101 subset, 41 k images, choose non-Indian dishes
        "food41/images",  # images/<class_name>/..
        [
            "pizza", "hamburger", "spaghetti_bolognese", "paella", "takoyaki", "ramen",
            "sushi", "falafel", "poutine", "hot_dog"
        ],
        1,
    ),
    "fruit": (
        "moltean/fruits",  # Fruits-360, 90 k images cropped
        "fruits-360/Training",  # Training/<fruit_name>/..
        [],
        2,
    ),
    "vegetable": (
        "rahmakopen/vegetables-image-dataset",  # 2 k veg images
        "Vegetable Images",  # Vegetable Images/<veg_name>/..
        [],
        3,
    ),
}

IMAGES_PER_CLASS = 250  # 250 * 4 = 1000 total
VAL_SPLIT = 0.2  # 80/20 split
RNG_SEED = 42

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def download_and_extract(api: KaggleApi, slug: str, dest: Path) -> Path:
    """Download a Kaggle dataset and extract it to *dest/slug/*."""
    dest.mkdir(parents=True, exist_ok=True)
    archive_path = dest / f"{slug.split('/')[-1]}.zip"
    if not archive_path.exists():
        print(f"⏬ Downloading {slug}… (~this may take a while)")
        api.dataset_download_files(slug, path=dest, quiet=False)
    else:
        print(f"✔ Archive already present: {archive_path.name}")

    print("📦 Extracting…")
    extract_dir = dest / slug.replace("/", "_")
    if extract_dir.exists():
        print("✔ Already extracted, skipping")
        return extract_dir

    # Kaggle always delivers zip archives for datasets
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(path=extract_dir)
    return extract_dir


def list_images(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def write_yolo_label(label_path: Path, class_id: int) -> None:
    """Write a full-image bounding box (class_id  x=0.5 y=0.5 w=1 h=1)."""
    label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    seed(RNG_SEED)
    api = KaggleApi()
    api.authenticate()

    dest_root = Path(args.dest).resolve()
    img_train = dest_root / "images" / "train"
    img_val = dest_root / "images" / "val"
    lbl_train = dest_root / "labels" / "train"
    lbl_val = dest_root / "labels" / "val"
    for p in (img_train, img_val, lbl_train, lbl_val):
        p.mkdir(parents=True, exist_ok=True)

    for class_name, (slug, root_inside, subfolders, class_id) in DATASETS.items():
        print(f"========== Processing class {class_name!r} ==========")
        extract_dir = download_and_extract(api, slug, Path("kaggle_datasets"))
        src_root = extract_dir / root_inside
        candidates: list[Path] = []
        if subfolders:
            for sf in subfolders:
                candidates.extend(list_images(src_root / sf))
        else:
            candidates = list_images(src_root)

        if len(candidates) < IMAGES_PER_CLASS:
            print(
                f"⚠ Not enough images for '{class_name}' ({len(candidates)}) – will sample with replacement"
            )
        chosen = sample(candidates, IMAGES_PER_CLASS) if len(candidates) >= IMAGES_PER_CLASS else candidates

        # Split
        split_idx = int(len(chosen) * (1 - VAL_SPLIT))
        train_files = chosen[:split_idx]
        val_files = chosen[split_idx:]

        for split, files, img_dir, lbl_dir in [
            ("train", train_files, img_train, lbl_train),
            ("val", val_files, img_val, lbl_val),
        ]:
            for src in tqdm(files, desc=f"{class_name}-{split}"):
                dst_img = img_dir / f"{class_name}_{src.stem}{src.suffix.lower()}"
                shutil.copy2(src, dst_img)
                write_yolo_label(lbl_dir / (dst_img.stem + ".txt"), class_id)

    print("✅ Dataset assembly complete! Edit data.yaml if you used a non-default destination.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare 1000-image food dataset for YOLOv8")
    parser.add_argument("--dest", type=str, default="data", help="Destination root directory")
    main(parser.parse_args())