"""
Acoustic Emotion Recognition - Batch Test Runner Script
Tests all sample audio clips in the test_audio folder using the trained MLP classifier and 40 MFCC extraction.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio_detector import AudioEmotionDetector

def main():
    print("=" * 68)
    print("   ACOUSTIC SPEECH EMOTION RECOGNITION - BATCH TEST RUNNER")
    print("=" * 68)

    detector = AudioEmotionDetector()
    test_dir = Path(__file__).resolve().parent

    audio_files = sorted([f for f in test_dir.iterdir() if f.suffix.lower() == ".wav"])

    if not audio_files:
        print(f"[!] No .wav test audio files found in {test_dir}")
        return

    print(f"\n[+] Found {len(audio_files)} test audio clips in: {test_dir.name}/\n")

    results = []
    for audio_path in audio_files:
        try:
            pred = detector.predict(str(audio_path))
            top_emotion = pred.get("emotion", "Unknown")
            confidence = pred.get("confidence", 0.0)
            duration = pred.get("duration_seconds", 0.0)

            results.append({
                "filename": audio_path.name,
                "duration": duration,
                "emotion": top_emotion,
                "confidence": confidence,
            })

            conf_pct = f"{confidence * 100:.1f}%"
            dur_str = f"{duration:.2f}s"
            print(f"  • {audio_path.name:<20} | Duration: {dur_str:>6} | Predicted: {top_emotion:<10} | Confidence: {conf_pct:>6}")
        except Exception as err:
            print(f"  [ERROR] {audio_path.name}: {err}")

    print("\n" + "=" * 68)
    print(f"  BATCH COMPLETED: Successfully evaluated {len(results)} audio clips.")
    print("=" * 68 + "\n")

if __name__ == "__main__":
    main()
