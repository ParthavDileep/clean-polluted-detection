"""
Model Training for Clean vs Polluted Detection
- Initial training with frozen base
- Fine-tuning with unfrozen layers
- Save best model
- Plot training history
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, CSVLogger
)
import warnings
warnings.filterwarnings("ignore")

# Add src folder to path
sys.path.append(str(Path(__file__).parent))

from preprocess import get_train_generator, get_val_generator
from model_builder import build_model, fine_tune_model, print_model_info

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
MODEL_DIR       = BASE_DIR / "model"
LOGS_DIR        = BASE_DIR / "logs"

MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

MODEL_PATH      = MODEL_DIR / "clean_polluted_model.h5"
BEST_MODEL_PATH = MODEL_DIR / "best_model.h5"
HISTORY_PATH    = MODEL_DIR / "training_history.json"

EPOCHS_INITIAL  = 20
EPOCHS_FINETUNE = 10

# ─────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────
def get_callbacks(phase="initial"):
    """
    Get training callbacks for early stopping, checkpointing, etc.
    """
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        
        # Stop early if no improvement
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # CSV Logger
        CSVLogger(
            filename=str(LOGS_DIR / f"training_log_{phase}.csv"),
            append=False
        )
    ]
    
    return callbacks

# ─────────────────────────────────────────
# PLOT TRAINING HISTORY
# ─────────────────────────────────────────
def plot_history(history, phase="initial"):
    """
    Plot training and validation accuracy/loss curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History - {phase.upper()}",
                 fontsize=16, fontweight="bold")
    
    # Accuracy plot
    axes[0].plot(history.history["accuracy"],
                 label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"],
                 label="Val Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history["loss"],
                 label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"],
                 label="Val Loss", linewidth=2)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = BASE_DIR / f"training_history_{phase}.png"
    plt.savefig(str(save_path), dpi=150)
    plt.show()
    print(f"✅ Training history saved as training_history_{phase}.png")

# ─────────────────────────────────────────
# SAVE HISTORY TO JSON
# ─────────────────────────────────────────
def save_history(history_initial, history_finetune):
    """
    Combine and save training history to JSON
    """
    combined = {
        "initial": {
            "accuracy"     : history_initial.history["accuracy"],
            "val_accuracy" : history_initial.history["val_accuracy"],
            "loss"         : history_initial.history["loss"],
            "val_loss"     : history_initial.history["val_loss"]
        },
        "finetune": {
            "accuracy"     : history_finetune.history["accuracy"],
            "val_accuracy" : history_finetune.history["val_accuracy"],
            "loss"         : history_finetune.history["loss"],
            "val_loss"     : history_finetune.history["val_loss"]
        }
    }
    
    with open(HISTORY_PATH, "w") as f:
        json.dump(combined, f, indent=2)
    
    print(f"✅ Training history saved to {HISTORY_PATH}")

# ─────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────
def train():
    print("=" * 60)
    print("   🚀 CLEAN vs POLLUTED MODEL TRAINING")
    print("=" * 60)
    
    # ─────────────────────────────
    # STEP 1: Load Data Generators
    # ─────────────────────────────
    print("\n📦 Loading Data Generators...")
    train_gen = get_train_generator()
    val_gen   = get_val_generator()
    
    print(f"  ✅ Train samples : {train_gen.samples}")
    print(f"  ✅ Val   samples : {val_gen.samples}")
    print(f"  ✅ Classes       : {train_gen.class_indices}")
    
    # ─────────────────────────────
    # STEP 2: Build Model
    # ─────────────────────────────
    print("\n🏗️  Building Model...")
    model, base_model = build_model()
    print_model_info(model)
    
    # ─────────────────────────────
    # STEP 3: Initial Training
    # ─────────────────────────────
    print("\n" + "=" * 60)
    print(f"   📚 PHASE 1: INITIAL TRAINING ({EPOCHS_INITIAL} epochs)")
    print("=" * 60)
    
    history_initial = model.fit(
        train_gen,
        epochs=EPOCHS_INITIAL,
        validation_data=val_gen,
        callbacks=get_callbacks(phase="initial"),
        verbose=1
    )
    
    print("\n✅ Initial training complete!")
    plot_history(history_initial, phase="initial")
    
    # ─────────────────────────────
    # STEP 4: Fine-tuning
    # ─────────────────────────────
    print("\n" + "=" * 60)
    print(f"   🎯 PHASE 2: FINE-TUNING ({EPOCHS_FINETUNE} epochs)")
    print("=" * 60)
    
    model = fine_tune_model(model, base_model)
    
    history_finetune = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=get_callbacks(phase="finetune"),
        verbose=1
    )
    
    print("\n✅ Fine-tuning complete!")
    plot_history(history_finetune, phase="finetune")
    
    # ─────────────────────────────
    # STEP 5: Save Final Model
    # ─────────────────────────────
    print("\n💾 Saving Final Model...")
    model.save(str(MODEL_PATH))
    print(f"  ✅ Model saved to: {MODEL_PATH}")
    
    # Save history
    save_history(history_initial, history_finetune)
    
    # ─────────────────────────────
    # STEP 6: Final Results
    # ─────────────────────────────
    print("\n" + "=" * 60)
    print("   📊 FINAL TRAINING RESULTS")
    print("=" * 60)
    
    final_train_acc = history_finetune.history["accuracy"][-1]
    final_val_acc   = history_finetune.history["val_accuracy"][-1]
    final_train_loss = history_finetune.history["loss"][-1]
    final_val_loss  = history_finetune.history["val_loss"][-1]
    
    print(f"  Final Train Accuracy : {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"  Final Val   Accuracy : {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"  Final Train Loss     : {final_train_loss:.4f}")
    print(f"  Final Val   Loss     : {final_val_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("   ✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  📁 Model: {MODEL_PATH}")
    print(f"  📁 Best Model: {BEST_MODEL_PATH}")
    print(f"  📁 History: {HISTORY_PATH}")
    print("=" * 60)


# ─────────────────────────────────────────
# RUN TRAINING
# ─────────────────────────────────────────
if __name__ == "__main__":
    train()