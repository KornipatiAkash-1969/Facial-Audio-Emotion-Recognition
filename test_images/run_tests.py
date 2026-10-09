"""
Facial Emotion Recognition - Test Runner Script
Tests all sample images in the test_images folder using the trained CNN model.
"""

import os
import sys
from pathlib import Path
import cv2

# Add parent directory to path so src modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.facial_detector import FacialEmotionDetector

def main():
    print("=" * 65)
    print("  EMOTION RECOGNITION SYSTEM - BATCH TEST RUNNER")
    print("=" * 65)

    detector = FacialEmotionDetector()
    test_dir = Path(__file__).resolve().parent

    image_files = sorted([f for f in test_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    if not image_files:
        print(f"[!] No test images found in {test_dir}")
        return

    print(f"\n[+] Found {len(image_files)} test images in: {test_dir.name}/\n")

    results = []
    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[ERROR] Could not load {img_path.name}")
            continue

        pred = detector.predict(frame)
        num_faces = pred.get("num_faces", 0)
        top_emotion = pred.get("primary_emotion", "Unknown")
        confidence = pred.get("primary_confidence", 0.0)

        results.append({
            "filename": img_path.name,
            "num_faces": num_faces,
            "emotion": top_emotion,
            "confidence": confidence,
        })

        conf_pct = f"{confidence * 100:.1f}%"
        print(f"  • {img_path.name:<20} | Faces: {num_faces} | Predicted: {top_emotion:<10} | Confidence: {conf_pct}")

    print("\n" + "=" * 65)
    print(f"  BATCH COMPLETED: Successfully tested {len(results)} images.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
