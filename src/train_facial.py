"""
Training script for Facial Emotion Recognition CNN model.
Trains on FER-style grayscale facial expression image datasets with real-time
data augmentation (rotations, zooms, flips, translation) using Keras 3.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import keras
from keras import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    Input,
    Rescaling,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

from src.config import FACIAL_MODEL_PATH, EMOTIONS, PROJECT_ROOT


def build_facial_cnn(input_shape=(48, 48, 1), num_classes=7) -> Sequential:
    """Build the CNN architecture matching the production emotion model."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        Conv2D(256, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    return model


def train_facial_model(
    data_dir: str,
    output_path: str = str(FACIAL_MODEL_PATH),
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.0001,
    fine_tune: bool = True,
    augment: bool = True,
):
    """Train or fine-tune the CNN facial emotion classifier."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data directory not found at: {train_dir}")

    print("=" * 60)
    print("Starting Facial Emotion Recognition Model Training")
    print("=" * 60)
    print(f"Dataset root      : {data_dir}")
    print(f"Target model file : {output_path}")
    print(f"Epochs            : {epochs}")
    print(f"Batch size        : {batch_size}")
    print(f"Learning rate     : {lr}")
    print(f"Fine-tune weights : {fine_tune}")
    print(f"Data augmentation : {augment}")

    # Load image datasets natively with Keras 3 / tf.data
    raw_train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        color_mode="grayscale",
        image_size=(48, 48),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
    )

    has_val = os.path.exists(val_dir)
    raw_val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        color_mode="grayscale",
        image_size=(48, 48),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    ) if has_val else None

    # Normalization & Augmentation pipeline
    rescaling = Rescaling(1.0 / 255.0)

    if augment:
        data_aug = Sequential([
            RandomFlip("horizontal"),
            RandomRotation(0.08),
            RandomZoom(0.08),
            RandomTranslation(0.05, 0.05),
        ])
        train_ds = raw_train_ds.map(
            lambda x, y: (data_aug(rescaling(x), training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
    else:
        train_ds = raw_train_ds.map(
            lambda x, y: (rescaling(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

    val_ds = raw_val_ds.map(
        lambda x, y: (rescaling(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE) if raw_val_ds else None

    # Model initialization
    if fine_tune and os.path.exists(output_path):
        print(f"\nLoading existing model weights from: {output_path}")
        try:
            model = keras.models.load_model(output_path, compile=False)
            print("Successfully loaded existing model for fine-tuning.")
        except Exception as err:
            print(f"Could not load existing model ({err}). Building fresh architecture.")
            model = build_facial_cnn(input_shape=(48, 48, 1), num_classes=len(EMOTIONS))
    else:
        print("\nBuilding fresh CNN architecture.")
        model = build_facial_cnn(input_shape=(48, 48, 1), num_classes=len(EMOTIONS))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = []
    if val_ds:
        callbacks.extend([
            ModelCheckpoint(output_path, monitor="val_accuracy", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor="val_loss", patience=4, verbose=1, restore_best_weights=True),
        ])

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nTraining completed successfully. Model saved to: {output_path}")

    if val_ds:
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        print(f"Final Validation Accuracy: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}\n")

    return history


def main():
    default_dataset = os.path.join(
        PROJECT_ROOT.parent, "Facial-Emotion-Recognition", "data"
    )
    if not os.path.exists(default_dataset):
        default_dataset = str(PROJECT_ROOT / "data")

    parser = argparse.ArgumentParser(description="Train Facial Emotion CNN")
    parser.add_argument("--data_dir", type=str, default=default_dataset, help="Root data folder")
    parser.add_argument("--output_model", type=str, default=str(FACIAL_MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch without loading weights")
    parser.add_argument("--no_augment", action="store_true", help="Disable real-time data augmentation")
    args = parser.parse_args()

    train_facial_model(
        data_dir=args.data_dir,
        output_path=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        fine_tune=not args.from_scratch,
        augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
