#!/usr/bin/env python3
"""
Command-Line Interface (CLI) for Multimodal Emotion Recognition System.
Enables headless, batch, and scriptable emotion classification from face, audio,
or multimodal inputs.

Examples:
    python cli.py --face assets/samples/faces/happy_PrivateTest_10077120.jpg
    python cli.py --audio assets/samples/audio/tess_happy.wav
    python cli.py --multimodal --face assets/samples/faces/happy_PrivateTest_10077120.jpg --audio assets/samples/audio/tess_happy.wav
    python cli.py --webcam
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.facial_detector import FacialEmotionDetector
from src.audio_detector import AudioEmotionDetector
from src.multimodal_fusion import MultimodalEmotionFusion


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_webcam_mode():
    """Launch real-time OpenCV window for webcam facial emotion recognition."""
    print("Starting Live Webcam Facial Emotion Recognition. Press 'q' to exit.")
    detector = FacialEmotionDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam at index 0.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = detector.predict(frame, annotate=True)
        cv2.imshow("Facial Emotion Recognition (Press 'q' to exit)", res["annotated_frame"])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def print_table(probs: dict, top_emotion: str):
    """Print an ASCII bar chart of emotion probabilities."""
    print("\nEmotion Distribution:")
    print("-" * 45)
    for em, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(p * 25)
        bars = "=" * bar_len
        marker = " <--" if em == top_emotion else ""
        print(f"  {em:10s} : {p*100:5.1f}% | {bars:<25}{marker}")
    print("-" * 45)


def main():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition CLI")
    parser.add_argument("--face", type=str, help="Path to face image file")
    parser.add_argument("--audio", type=str, help="Path to speech audio file (.wav)")
    parser.add_argument("--multimodal", action="store_true", help="Run multimodal fusion (requires both --face and --audio)")
    parser.add_argument("--face_weight", type=float, default=0.5, help="Weight for facial modality (default: 0.5)")
    parser.add_argument("--audio_weight", type=float, default=0.5, help="Weight for audio modality (default: 0.5)")
    parser.add_argument("--webcam", action="store_true", help="Launch live OpenCV webcam detection")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if args.webcam:
        run_webcam_mode()
        return

    if not args.face and not args.audio:
        parser.print_help()
        sys.exit(1)

    face_detector = None
    audio_detector = None

    face_res = None
    audio_res = None

    # Process Face
    if args.face:
        if not os.path.exists(args.face):
            print(f"Error: Face image not found: {args.face}", file=sys.stderr)
            sys.exit(1)
        face_detector = FacialEmotionDetector()
        face_res = face_detector.predict(args.face)

    # Process Audio
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
            sys.exit(1)
        audio_detector = AudioEmotionDetector()
        audio_res = audio_detector.predict(args.audio)

    # Multimodal Fusion
    if args.multimodal:
        if not face_res or not audio_res:
            print("Error: Both --face and --audio are required for --multimodal mode.", file=sys.stderr)
            sys.exit(1)

        fusion_res = MultimodalEmotionFusion.fuse(
            face_result=face_res,
            audio_result=audio_res,
            face_weight=args.face_weight,
            audio_weight=args.audio_weight,
        )

        if args.json:
            # Strip non-serializable fields if any
            print(json.dumps(fusion_res, indent=2))
        else:
            print("\n" + "=" * 55)
            print("         MULTIMODAL EMOTION RECOGNITION REPORT         ")
            print("=" * 55)
            print(f"Top Fused Emotion : {fusion_res['fused_emotion'].upper()} ({fusion_res['fused_confidence']*100:.1f}%)")
            status = "CONGRUENT (Synergistic Match)" if fusion_res["is_congruent"] else "INCONGRUENT (Affective Conflict)"
            print(f"Congruency Status : {status}")
            print(f"Congruency Score  : {fusion_res['congruency_score']}")
            print("-" * 55)
            print(f"Facial Modality   : {fusion_res['face']['emotion']} ({fusion_res['face']['confidence']*100:.1f}%) [Weight: {fusion_res['face']['weight']*100:.0f}%]")
            print(f"Audio Modality    : {fusion_res['audio']['emotion']} ({fusion_res['audio']['confidence']*100:.1f}%) [Weight: {fusion_res['audio']['weight']*100:.0f}%]")
            print_table(fusion_res["combined_probabilities"], fusion_res["fused_emotion"])
            print(f"Affective Insight : {fusion_res['insight']}")
            print("=" * 55 + "\n")
        return

    # Single Face Output
    if face_res and not args.audio:
        if args.json:
            clean = {k: v for k, v in face_res.items() if k not in ["original_frame", "annotated_frame"]}
            print(json.dumps(clean, indent=2))
        else:
            print("\n" + "=" * 50)
            print("         FACIAL EMOTION RECOGNITION REPORT        ")
            print("=" * 50)
            print(f"Input Image       : {args.face}")
            print(f"Faces Detected    : {face_res['num_faces']}")
            print(f"Primary Emotion   : {face_res['primary_emotion'].upper()} ({face_res['primary_confidence']*100:.1f}%)")
            print_table(face_res["primary_probabilities"], face_res["primary_emotion"])
            print("=" * 50 + "\n")

    # Single Audio Output
    if audio_res and not args.face:
        if args.json:
            print(json.dumps(audio_res, indent=2))
        else:
            print("\n" + "=" * 50)
            print("          AUDIO EMOTION RECOGNITION REPORT        ")
            print("=" * 50)
            print(f"Input Audio       : {args.audio}")
            print(f"Duration          : {audio_res['duration_seconds']} seconds")
            print(f"Predicted Emotion : {audio_res['emotion'].upper()} ({audio_res['confidence']*100:.1f}%)")
            print_table(audio_res["probabilities"], audio_res["emotion"])
            print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
