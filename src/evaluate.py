"""
Model Evaluation for Clean vs Polluted Detection
- Confusion Matrix
- Classification Report
- ROC Curve
- Per-class metrics
- Misclassified samples
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import warnings
warnings.filterwarnings("ignore")

# Add src folder to path
sys.path.append(str(Path(__file__).parent))

from preprocess import get_val_generator

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
MODEL_DIR       = BASE_DIR / "model"
BEST_MODEL_PATH = MODEL_DIR / "best_model.h5"
CLASSES         = ["clean", "polluted"]

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
def load_trained_model():
    print(f"📦 Loading model from: {BEST_MODEL_PATH}")
    model = load_model(str(BEST_MODEL_PATH))
    print("✅ Model loaded successfully!")
    return model

# ─────────────────────────────────────────
# GET PREDICTIONS
# ─────────────────────────────────────────
def get_predictions(model, val_gen):
    print("\n🔮 Generating predictions...")
    
    # Reset generator
    val_gen.reset()
    
    # Get predictions
    predictions = model.predict(val_gen, verbose=1)
    y_pred_prob = predictions.flatten()
    y_pred      = (y_pred_prob > 0.5).astype(int)
    
    # Get true labels
    y_true      = val_gen.classes
    filenames   = val_gen.filenames
    
    return y_true, y_pred, y_pred_prob, filenames

# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 16, "weight": "bold"}
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / "confusion_matrix.png"), dpi=150)
    plt.show()
    print("✅ Confusion matrix saved!")
    
    return cm

# ─────────────────────────────────────────
# CLASSIFICATION REPORT
# ─────────────────────────────────────────
def print_classification_report(y_true, y_pred):
    print("\n" + "=" * 60)
    print("           CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        digits=4
    )
    print(report)
    
    # Save to file
    with open(BASE_DIR / "classification_report.txt", "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print("✅ Report saved to classification_report.txt")

# ─────────────────────────────────────────
# ROC CURVE
# ─────────────────────────────────────────
def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc     = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--",
             label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / "roc_curve.png"), dpi=150)
    plt.show()
    print(f"✅ ROC curve saved! AUC = {roc_auc:.4f}")
    
    return roc_auc

# ─────────────────────────────────────────
# PRECISION-RECALL CURVE
# ─────────────────────────────────────────
def plot_precision_recall_curve(y_true, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="green", lw=2)
    plt.fill_between(recall, precision, alpha=0.2, color="green")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / "precision_recall_curve.png"), dpi=150)
    plt.show()
    print("✅ Precision-Recall curve saved!")

# ─────────────────────────────────────────
# PREDICTION CONFIDENCE DISTRIBUTION
# ─────────────────────────────────────────
def plot_confidence_distribution(y_true, y_pred_prob):
    plt.figure(figsize=(10, 6))
    
    # Clean predictions
    clean_probs    = y_pred_prob[y_true == 0]
    polluted_probs = y_pred_prob[y_true == 1]
    
    plt.hist(clean_probs, bins=20, alpha=0.6, label="Clean (True)",
             color="green", edgecolor="black")
    plt.hist(polluted_probs, bins=20, alpha=0.6, label="Polluted (True)",
             color="red", edgecolor="black")
    
    plt.axvline(x=0.5, color="blue", linestyle="--", linewidth=2,
                label="Decision Threshold (0.5)")
    plt.xlabel("Predicted Probability (Polluted)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Prediction Confidence Distribution",
              fontsize=16, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / "confidence_distribution.png"), dpi=150)
    plt.show()
    print("✅ Confidence distribution saved!")

# ─────────────────────────────────────────
# MISCLASSIFIED SAMPLES
# ─────────────────────────────────────────
def show_misclassified(y_true, y_pred, y_pred_prob, filenames, val_gen):
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    print(f"\n🔍 Misclassified Samples: {len(misclassified_idx)}/{len(y_true)}")
    
    if len(misclassified_idx) == 0:
        print("  🎉 No misclassified images! Perfect predictions!")
        return
    
    # Show up to 10 misclassified
    n_show = min(10, len(misclassified_idx))
    cols   = 5
    rows   = (n_show + cols - 1) // cols
    
    from PIL import Image
    val_dir = BASE_DIR / "dataset" / "val"
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.suptitle("Misclassified Samples", fontsize=16, fontweight="bold")
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(misclassified_idx[:n_show]):
        row = i // cols
        col = i % cols
        
        img_path = val_dir / filenames[idx]
        img      = Image.open(img_path)
        
        true_label = CLASSES[y_true[idx]]
        pred_label = CLASSES[y_pred[idx]]
        confidence = y_pred_prob[idx]
        
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
        axes[row, col].set_title(
            f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}",
            fontsize=10, color="red"
        )
    
    # Hide unused subplots
    for i in range(n_show, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / "misclassified_samples.png"), dpi=150)
    plt.show()
    print("✅ Misclassified samples saved!")

# ─────────────────────────────────────────
# FINAL METRICS
# ─────────────────────────────────────────
def print_final_metrics(y_true, y_pred, y_pred_prob):
    from sklearn.metrics import (
        accuracy_score, precision_score,
        recall_score, f1_score
    )
    
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("           FINAL EVALUATION METRICS")
    print("=" * 60)
    print(f"  ✅ Accuracy   : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ✅ Precision  : {precision:.4f} ({precision*100:.2f}%)")
    print(f"  ✅ Recall     : {recall:.4f} ({recall*100:.2f}%)")
    print(f"  ✅ F1-Score   : {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 60)

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def evaluate():
    print("=" * 60)
    print("   📊 MODEL EVALUATION")
    print("=" * 60)
    
    # Load model and data
    model   = load_trained_model()
    val_gen = get_val_generator()
    
    # Get predictions
    y_true, y_pred, y_pred_prob, filenames = get_predictions(model, val_gen)
    
    # Print final metrics
    print_final_metrics(y_true, y_pred, y_pred_prob)
    
    # Classification report
    print_classification_report(y_true, y_pred)
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_pred_prob)
    plot_precision_recall_curve(y_true, y_pred_prob)
    plot_confidence_distribution(y_true, y_pred_prob)
    
    # Show misclassified
    show_misclassified(y_true, y_pred, y_pred_prob, filenames, val_gen)
    
    print("\n" + "=" * 60)
    print("   ✅ EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()