"""
Prediction System for Clean vs Polluted Detection
- Predict single image
- Predict batch of images
- Visualize predictions with confidence
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
MODEL_DIR       = BASE_DIR / "model"
BEST_MODEL_PATH = MODEL_DIR / "best_model.h5"
TEST_DIR        = BASE_DIR / "test_images"
RESULTS_DIR     = BASE_DIR / "predictions"

TEST_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
CLASSES    = ["clean", "polluted"]

# ─────────────────────────────────────────
# LOAD MODEL (singleton)
# ─────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"📦 Loading model...")
        _model = load_model(str(BEST_MODEL_PATH))
        print("✅ Model loaded!")
    return _model

# ─────────────────────────────────────────
# PREPROCESS IMAGE
# ─────────────────────────────────────────
def preprocess_image(img_path):
    img       = Image.open(img_path).convert("RGB")
    img       = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# ─────────────────────────────────────────
# PREDICT SINGLE IMAGE
# ─────────────────────────────────────────
def predict_image(img_path, show_plot=True, save_result=True):
    """
    Predict if an image is clean or polluted
    """
    img_path = Path(img_path)
    
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return None
    
    # Load model
    model = get_model()
    
    # Preprocess
    img_array, img = preprocess_image(img_path)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Interpret result
    if prediction > 0.5:
        label      = "POLLUTED"
        confidence = prediction * 100
        color      = "red"
        emoji      = "🚫"
    else:
        label      = "CLEAN"
        confidence = (1 - prediction) * 100
        color      = "green"
        emoji      = "✅"
    
    # Print result
    print("\n" + "=" * 50)
    print(f"   {emoji} PREDICTION RESULT")
    print("=" * 50)
    print(f"  📁 Image      : {img_path.name}")
    print(f"  🏷️  Prediction : {label}")
    print(f"  📊 Confidence : {confidence:.2f}%")
    print(f"  🔢 Raw Score  : {prediction:.4f}")
    print("=" * 50)
    
    # Visualize
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title(f"Input Image\n{img_path.name}",
                          fontsize=12, fontweight="bold")
        axes[0].axis("off")
        
        # Prediction bar chart
        clean_prob    = (1 - prediction) * 100
        polluted_prob = prediction * 100
        
        bars = axes[1].barh(
            ["Clean", "Polluted"],
            [clean_prob, polluted_prob],
            color=["green", "red"],
            edgecolor="black", alpha=0.7
        )
        axes[1].set_xlim([0, 100])
        axes[1].set_xlabel("Probability (%)", fontsize=12)
        axes[1].set_title(
            f"Prediction: {label} ({confidence:.1f}%)",
            fontsize=14, fontweight="bold", color=color
        )
        axes[1].grid(True, alpha=0.3, axis="x")
        
        # Add labels
        for bar, value in zip(bars, [clean_prob, polluted_prob]):
            axes[1].text(
                value + 2, bar.get_y() + bar.get_height()/2,
                f"{value:.1f}%", va="center", fontweight="bold"
            )
        
        plt.tight_layout()
        
        # Save result
        if save_result:
            result_path = RESULTS_DIR / f"prediction_{img_path.stem}.png"
            plt.savefig(str(result_path), dpi=150)
            print(f"💾 Result saved: {result_path}")
        
        plt.show()
    
    return {
        "image"      : img_path.name,
        "label"      : label,
        "confidence" : confidence,
        "raw_score"  : float(prediction)
    }

# ─────────────────────────────────────────
# PREDICT BATCH
# ─────────────────────────────────────────
def predict_batch(folder_path=None):
    """
    Predict all images in a folder
    """
    if folder_path is None:
        folder_path = TEST_DIR
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        print(f"👉 Create folder and add images: {folder_path}")
        return
    
    # Get all images
    images = list(folder_path.glob("*.jpg")) + \
             list(folder_path.glob("*.jpeg")) + \
             list(folder_path.glob("*.png"))
    
    if not images:
        print(f"❌ No images found in: {folder_path}")
        print("👉 Add some test images (.jpg, .jpeg, .png)")
        return
    
    print("\n" + "=" * 60)
    print(f"   🔮 BATCH PREDICTION ({len(images)} images)")
    print("=" * 60)
    
    results = []
    for img_path in images:
        result = predict_image(img_path, show_plot=False, save_result=False)
        if result:
            results.append(result)
    
    # Show summary
    print("\n" + "=" * 60)
    print("   📊 BATCH PREDICTION SUMMARY")
    print("=" * 60)
    
    clean_count    = sum(1 for r in results if r["label"] == "CLEAN")
    polluted_count = sum(1 for r in results if r["label"] == "POLLUTED")
    
    print(f"  ✅ Clean    : {clean_count}")
    print(f"  🚫 Polluted : {polluted_count}")
    print(f"  📁 Total    : {len(results)}")
    print("=" * 60)
    
    # Visualize all predictions
    visualize_batch_results(results, images)
    
    return results

# ─────────────────────────────────────────
# VISUALIZE BATCH RESULTS
# ─────────────────────────────────────────
def visualize_batch_results(results, image_paths):
    n     = len(results)
    cols  = min(4, n)
    rows  = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle("Batch Predictions", fontsize=16, fontweight="bold")
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (result, img_path) in enumerate(zip(results, image_paths)):
        row = i // cols
        col = i % cols
        
        img   = Image.open(img_path)
        color = "green" if result["label"] == "CLEAN" else "red"
        
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
        axes[row, col].set_title(
            f"{result['label']}\n{result['confidence']:.1f}%",
            fontsize=11, fontweight="bold", color=color
        )
    
    # Hide unused
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "batch_predictions.png"
    plt.savefig(str(save_path), dpi=150)
    plt.show()
    print(f"💾 Batch visualization saved: {save_path}")

# ─────────────────────────────────────────
# MAIN (Interactive)
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("   🔮 CLEAN vs POLLUTED - PREDICTION SYSTEM")
    print("=" * 60)
    
    while True:
        print("\n📋 Choose an option:")
        print("  1. Predict single image (enter path)")
        print("  2. Predict batch (from test_images folder)")
        print("  3. Predict from validation set (random sample)")
        print("  4. Exit")
        
        choice = input("\n👉 Enter choice (1-4): ").strip()
        
        if choice == "1":
            img_path = input("📁 Enter image path: ").strip().strip('"')
            predict_image(img_path)
        
        elif choice == "2":
            predict_batch()
        
        elif choice == "3":
            # Random sample from validation set
            val_dir = BASE_DIR / "dataset" / "val"
            import random
            
            all_images = []
            for class_name in CLASSES:
                imgs = list((val_dir / class_name).glob("*.jpg")) + \
                       list((val_dir / class_name).glob("*.jpeg")) + \
                       list((val_dir / class_name).glob("*.png"))
                all_images.extend(imgs)
            
            if all_images:
                sample = random.choice(all_images)
                print(f"\n🎲 Random sample: {sample.name}")
                predict_image(sample)
            else:
                print("❌ No validation images found")
        
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()