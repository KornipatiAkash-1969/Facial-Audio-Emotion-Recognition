"""
Training script for Facial Emotion Recognition CNN model.
Trains on FER-style grayscale facial expression image datasets.
"""

import argparse
import os
from pathlib import Path
import keras
from keras import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    Input,
)
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from src.config import FACIAL_MODEL_PATH, EMOTIONS


def build_facial_cnn(input_shape=(48, 48, 1), num_classes=7) -> Sequential:
    """Build the 4-block CNN architecture for facial emotion recognition."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
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
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.0001,
):
    """Train the CNN facial emotion classifier."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data directory not found at: {train_dir}")

    print("=" * 60)
    print("Starting Facial Emotion Recognition Model Training")
    print("=" * 60)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        color_mode="grayscale",
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        color_mode="grayscale",
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    ) if os.path.exists(val_dir) else None

    model = build_facial_cnn(input_shape=(48, 48, 1), num_classes=len(EMOTIONS))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(output_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ] if val_gen else []

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nTraining complete. Model saved to: {output_path}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Facial Emotion CNN")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data folder")
    parser.add_argument("--output_model", type=str, default=str(FACIAL_MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()

    train_facial_model(
        data_dir=args.data_dir,
        output_path=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
