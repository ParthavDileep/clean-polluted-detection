"""
Model Builder for Clean vs Polluted Detection
Using MobileNetV2 with Transfer Learning
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D,
    BatchNormalization, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
IMAGE_SIZE     = (224, 224)
INPUT_SHAPE    = (224, 224, 3)
LEARNING_RATE  = 0.0001
DROPOUT_RATE   = 0.5

# ─────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────
def build_model(input_shape=INPUT_SHAPE, learning_rate=LEARNING_RATE):
    """
    Build a CNN model using MobileNetV2 transfer learning
    """
    
    # Load pre-trained MobileNetV2 (without top layer)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build custom model on top
    inputs = Input(shape=input_shape)
    x      = base_model(inputs, training=False)
    
    # Custom classification head
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Output layer (binary classification)
    outputs = Dense(1, activation="sigmoid")(x)
    
    # Create model
    model = Model(inputs, outputs, name="Clean_Polluted_Detector")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", 
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    
    return model, base_model


# ─────────────────────────────────────────
# FINE-TUNE MODEL (Unfreeze top layers)
# ─────────────────────────────────────────
def fine_tune_model(model, base_model, learning_rate=0.00001, unfreeze_layers=30):
    """
    Unfreeze top layers of base model for fine-tuning
    """
    # Unfreeze base model
    base_model.trainable = True
    
    # Freeze all layers except last 'unfreeze_layers'
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    
    return model


# ─────────────────────────────────────────
# MODEL SUMMARY
# ─────────────────────────────────────────
def print_model_info(model):
    """
    Print detailed model information
    """
    print("\n" + "=" * 60)
    print("           MODEL ARCHITECTURE")
    print("=" * 60)
    
    model.summary()
    
    # Count parameters
    total_params     = model.count_params()
    trainable_params = sum([
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    ])
    non_trainable    = total_params - trainable_params
    
    print("\n" + "=" * 60)
    print("           PARAMETER SUMMARY")
    print("=" * 60)
    print(f"  Total Parameters       : {total_params:,}")
    print(f"  Trainable Parameters   : {trainable_params:,}")
    print(f"  Non-trainable Params   : {non_trainable:,}")
    print("=" * 60)


# ─────────────────────────────────────────
# MAIN - TEST MODEL CREATION
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Building Model...")
    print("=" * 60)
    
    # Build model
    model, base_model = build_model()
    
    # Print model info
    print_model_info(model)
    
    print("\n✅ Model built successfully!")
    print("👉 Ready to train using train.py")