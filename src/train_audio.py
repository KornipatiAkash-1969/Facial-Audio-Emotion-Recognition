"""
Training script for Speech Emotion Recognition model using the TESS dataset.
Supports multi-threaded feature extraction and scikit-learn MLP/RandomForest training.
"""

import argparse
import glob
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import joblib
import librosa
import numpy as np
import soundfile as sf
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.config import (
    AUDIO_MODEL_PATH,
    PROJECT_ROOT,
    TESS_FOLDER_MAP,
    normalize_emotion,
)


def extract_features_single(item):
    """Worker function to extract MFCCs from a single audio file."""
    wav_path, label, n_mfcc = item
    try:
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        # Normalize amplitude
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
        return mfcc, label
    except Exception as err:
        print(f"Warning: Failed to process {wav_path}: {err}")
        return None


def train_audio_model(
    data_dir: str,
    output_path: str = str(AUDIO_MODEL_PATH),
    model_type: str = "mlp",
    n_mfcc: int = 40,
    test_size: float = 0.2,
    max_workers: int = 8,
):
    """Main training routine for Audio Emotion Recognition."""
    print("=" * 60)
    print("Starting Speech Emotion Recognition Model Training")
    print("=" * 60)
    print(f"Dataset directory : {data_dir}")
    print(f"Output model path : {output_path}")
    print(f"Model architecture: {model_type.upper()}")
    print(f"MFCC coefficients : {n_mfcc}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    # Gather audio files matching TESS emotion folders
    file_items = []
    for folder, raw_label in TESS_FOLDER_MAP.items():
        folder_path = os.path.join(data_dir, folder)
        canonical_label = normalize_emotion(raw_label)
        if os.path.exists(folder_path):
            wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
            for w in wav_files:
                file_items.append((w, canonical_label, n_mfcc))

    # Also search directly in subfolders if nested
    if len(file_items) == 0:
        nested_wavs = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
        for w in nested_wavs:
            parent = os.path.basename(os.path.dirname(w))
            for key, val in TESS_FOLDER_MAP.items():
                if key.lower() in parent.lower():
                    file_items.append((w, normalize_emotion(val), n_mfcc))
                    break

    if len(file_items) == 0:
        raise ValueError(f"No valid TESS audio files found in {data_dir}")

    print(f"Found {len(file_items)} audio samples to process.")
    print("Extracting acoustic features in parallel...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(extract_features_single, file_items))

    valid = [r for r in results if r is not None]
    X = np.array([r[0] for r in valid])
    y = np.array([r[1] for r in valid])
    elapsed = time.time() - t0
    print(f"Feature extraction completed in {elapsed:.2f} seconds ({len(X)} valid samples).")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Dataset split: {len(X_train)} train samples, {len(X_test)} test samples.")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Instantiate model
    if model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )
    else:
        clf = RandomForestClassifier(n_estimators=150, random_state=42)

    print(f"Training {model_type.upper()} classifier...")
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 60)
    print(f"Evaluation Results - Accuracy: {acc * 100:.2f}%")
    print("=" * 60)
    print(classification_report(y_test, y_pred))

    # Save artifact
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": clf,
        "scaler": scaler,
        "classes": list(clf.classes_),
        "n_mfcc": n_mfcc,
        "accuracy": acc,
    }
    joblib.dump(payload, output_path)
    print(f"Trained model saved to: {output_path}\n")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Train Audio Emotion Recognition Model")
    default_dataset = os.path.join(
        PROJECT_ROOT.parent,
        "Audio-Emotion-Recognition",
        "TESS Toronto emotional speech set data",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_dataset,
        help="Path to TESS dataset root directory",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=str(AUDIO_MODEL_PATH),
        help="Destination path for .joblib model",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["mlp", "rf"],
        default="mlp",
        help="Classifier type: mlp (Neural Net) or rf (Random Forest)",
    )
    parser.add_argument(
        "--n_mfcc",
        type=int,
        default=40,
        help="Number of MFCC coefficients to extract",
    )
    args = parser.parse_args()
    train_audio_model(
        data_dir=args.data_dir,
        output_path=args.output_model,
        model_type=args.model_type,
        n_mfcc=args.n_mfcc,
    )


if __name__ == "__main__":
    main()
