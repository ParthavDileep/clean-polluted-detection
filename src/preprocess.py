"""
Data Preprocessing for Clean vs Polluted Detection
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
DATASET_DIR  = BASE_DIR / "dataset"
TRAIN_DIR    = DATASET_DIR / "train"
VAL_DIR      = DATASET_DIR / "val"
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
CLASSES      = ["clean", "polluted"]

# ─────────────────────────────────────────
# TRAINING GENERATOR
# ─────────────────────────────────────────
def get_train_generator():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASSES,
        shuffle=True
    )
    return train_generator

# ─────────────────────────────────────────
# VALIDATION GENERATOR
# ─────────────────────────────────────────
def get_val_generator():
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASSES,
        shuffle=False
    )
    return val_generator

# ─────────────────────────────────────────
# DATASET STATISTICS
# ─────────────────────────────────────────
def dataset_statistics():
    print("\n Dataset Statistics:")
    print("=" * 50)
    stats = {}
    for split in ["train", "val"]:
        for class_name in CLASSES:
            path  = DATASET_DIR / split / class_name
            count = (
                len(list(path.glob("*.jpg")))
                + len(list(path.glob("*.jpeg")))
                + len(list(path.glob("*.png")))
            )
            key        = f"{split}/{class_name}"
            stats[key] = count
            print(f"  {key:25s} : {count} images")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dataset Distribution", fontsize=14, fontweight="bold")

    train_counts = [stats["train/clean"], stats["train/polluted"]]
    axes[0].bar(CLASSES, train_counts, color=["green", "red"],
                edgecolor="black", alpha=0.7)
    axes[0].set_title("Training Set")
    axes[0].set_ylabel("Number of Images")
    for i, v in enumerate(train_counts):
        axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

    val_counts = [stats["val/clean"], stats["val/polluted"]]
    axes[1].bar(CLASSES, val_counts, color=["green", "red"],
                edgecolor="black", alpha=0.7)
    axes[1].set_title("Validation Set")
    axes[1].set_ylabel("Number of Images")
    for i, v in enumerate(val_counts):
        axes[1].text(i, v + 1, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(BASE_DIR / "dataset_distribution.png"), dpi=150)
    plt.show()
    print("✅ Distribution chart saved!")

# ─────────────────────────────────────────
# VERIFY IMAGES
# ─────────────────────────────────────────
def verify_images():
    print("\n Verifying Images...")
    print("=" * 50)
    corrupt = []
    valid   = 0
    for split in ["train", "val"]:
        for class_name in CLASSES:
            path   = DATASET_DIR / split / class_name
            images = (
                list(path.glob("*.jpg"))
                + list(path.glob("*.jpeg"))
                + list(path.glob("*.png"))
            )
            for img_path in images:
                try:
                    img = Image.open(img_path)
                    img.verify()
                    valid += 1
                except Exception:
                    corrupt.append(str(img_path))
                    print(f"  Corrupt: {img_path.name}")

    print(f"\n  Valid Images   : {valid}")
    print(f"  Corrupt Images : {len(corrupt)}")
    if not corrupt:
        print("\n  All images are valid!")
    return corrupt

# ─────────────────────────────────────────
# VISUALIZE SAMPLES
# ─────────────────────────────────────────
def visualize_samples():
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Sample Dataset Images", fontsize=16, fontweight="bold")
    for idx, class_name in enumerate(CLASSES):
        class_dir = TRAIN_DIR / class_name
        images    = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )
        for col, img_path in enumerate(images[:5]):
            img = Image.open(img_path).resize(IMAGE_SIZE)
            axes[idx, col].imshow(img)
            axes[idx, col].axis("off")
            if col == 0:
                axes[idx, col].set_title(
                    class_name.upper(),
                    fontsize=12,
                    fontweight="bold",
                    color="green" if class_name == "clean" else "red"
                )
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / "sample_images.png"), dpi=150)
    plt.show()
    print("✅ Sample images saved!")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Data Preprocessing Started")
    print("=" * 50)

    dataset_statistics()
    verify_images()

    print("\n Generating sample visualization...")
    visualize_samples()

    print("\n Testing Data Generators...")
    train_gen = get_train_generator()
    val_gen   = get_val_generator()

    print(f"\n  Train Samples  : {train_gen.samples}")
    print(f"  Val Samples    : {val_gen.samples}")
    print(f"  Classes        : {train_gen.class_indices}")

    print("\n" + "=" * 50)
    print("✅ Preprocessing Complete! Ready for Training")
    print("=" * 50)