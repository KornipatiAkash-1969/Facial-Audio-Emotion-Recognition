"""
Comprehensive Automated Test Suite for Multimodal Emotion Recognition.
Tests configuration, facial detection, audio emotion detection, and multimodal fusion.
"""

import os
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMOTIONS,
    FACIAL_MODEL_PATH,
    HAAR_CASCADE_PATH,
    AUDIO_MODEL_PATH,
    FACE_SAMPLES_DIR,
    AUDIO_SAMPLES_DIR,
    get_emoji_path,
)
from src.facial_detector import FacialEmotionDetector
from src.audio_detector import AudioEmotionDetector
from src.multimodal_fusion import MultimodalEmotionFusion


class TestMultimodalEmotionRecognition(unittest.TestCase):
    """Test suite covering all components."""

    def test_01_config_and_assets(self):
        """Verify model files, cascades, and asset paths exist."""
        self.assertTrue(FACIAL_MODEL_PATH.exists(), f"Missing facial model: {FACIAL_MODEL_PATH}")
        self.assertTrue(HAAR_CASCADE_PATH.exists(), f"Missing Haar cascade: {HAAR_CASCADE_PATH}")
        self.assertTrue(AUDIO_MODEL_PATH.exists(), f"Missing audio model: {AUDIO_MODEL_PATH}")
        self.assertEqual(len(EMOTIONS), 7)

        for em in EMOTIONS:
            path = get_emoji_path(em)
            self.assertIsNotNone(path, f"Emoji for {em} should not be None")
            self.assertTrue(path.exists(), f"Emoji file for {em} does not exist: {path}")

    def test_02_facial_detector(self):
        """Test facial emotion detection on a sample image."""
        detector = FacialEmotionDetector()
        self.assertTrue(detector.is_ready)

        sample_faces = list(FACE_SAMPLES_DIR.glob("*.*"))
        self.assertGreater(len(sample_faces), 0, "No sample faces found")

        # Test inference on first sample
        res = detector.predict(sample_faces[0], annotate=True)
        self.assertTrue(res["success"])
        self.assertIn(res["primary_emotion"], EMOTIONS)
        self.assertGreaterEqual(res["primary_confidence"], 0.0)
        self.assertLessEqual(res["primary_confidence"], 1.0)
        self.assertEqual(len(res["primary_probabilities"]), 7)
        self.assertIsNotNone(res["annotated_frame"])

    def test_03_audio_detector(self):
        """Test audio emotion detection on a sample audio file."""
        detector = AudioEmotionDetector()
        self.assertTrue(detector.is_ready)

        sample_audios = list(AUDIO_SAMPLES_DIR.glob("*.wav"))
        self.assertGreater(len(sample_audios), 0, "No sample audios found")

        # Test inference on first sample
        res = detector.predict(sample_audios[0])
        self.assertTrue(res["success"])
        self.assertIn(res["emotion"], EMOTIONS)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertEqual(len(res["probabilities"]), 7)
        self.assertGreater(res["duration_seconds"], 0.0)

    def test_04_multimodal_fusion(self):
        """Test decision-level fusion and congruency metrics."""
        face_mock = {
            "primary_emotion": "Happy",
            "primary_confidence": 0.90,
            "primary_probabilities": {
                "Angry": 0.01, "Disgust": 0.01, "Fear": 0.01,
                "Happy": 0.90, "Neutral": 0.05, "Sad": 0.01, "Surprise": 0.01
            }
        }
        audio_congruent = {
            "emotion": "Happy",
            "confidence": 0.95,
            "probabilities": {
                "Angry": 0.01, "Disgust": 0.01, "Fear": 0.01,
                "Happy": 0.95, "Neutral": 0.01, "Sad": 0.00, "Surprise": 0.01
            }
        }

        # Congruent test
        res_cong = MultimodalEmotionFusion.fuse(face_mock, audio_congruent, face_weight=0.5, audio_weight=0.5)
        self.assertEqual(res_cong["fused_emotion"], "Happy")
        self.assertTrue(res_cong["is_congruent"])
        self.assertGreaterEqual(res_cong["congruency_score"], 0.8)

        # Incongruent test (Face Happy vs Audio Angry)
        audio_incongruent = {
            "emotion": "Angry",
            "confidence": 0.95,
            "probabilities": {
                "Angry": 0.95, "Disgust": 0.01, "Fear": 0.01,
                "Happy": 0.01, "Neutral": 0.01, "Sad": 0.00, "Surprise": 0.01
            }
        }
        res_incong = MultimodalEmotionFusion.fuse(face_mock, audio_incongruent, face_weight=0.5, audio_weight=0.5)
        self.assertFalse(res_incong["is_congruent"])
        self.assertIn("Dissonance", res_incong["insight"])


if __name__ == "__main__":
    unittest.main()
