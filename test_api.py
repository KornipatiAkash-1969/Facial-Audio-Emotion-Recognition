"""
Integration test suite for FastAPI REST API backend.
Tests health endpoints, sample listings, facial inference, audio inference, and multimodal fusion.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient
from server import app


class TestFastAPIEndpoints(unittest.TestCase):
    """Test all FastAPI routes."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_status(self):
        """Test GET /api/status."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["models"]["face_detector_ready"])
        self.assertTrue(data["models"]["audio_detector_ready"])

    def test_02_samples(self):
        """Test GET /api/samples."""
        res = self.client.get("/api/samples")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreater(len(data["faces"]), 0)
        self.assertGreater(len(data["audios"]), 0)

    def test_03_predict_face_sample(self):
        """Test POST /api/predict/face with sample_name form data."""
        res = self.client.post("/api/predict/face", data={"sample_name": "happy_PrivateTest_10077120.jpg"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["primary_emotion"], "Happy")
        self.assertIn("annotated_image", data)
        self.assertTrue(data["annotated_image"].startswith("data:image/jpeg;base64,"))

    def test_04_predict_audio_sample(self):
        """Test POST /api/predict/audio with sample_name form data."""
        res = self.client.post("/api/predict/audio", data={"sample_name": "tess_happy.wav"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["emotion"], "Happy")
        self.assertGreater(data["duration_seconds"], 0)

    def test_05_predict_multimodal(self):
        """Test POST /api/predict/multimodal with face and audio samples."""
        res = self.client.post(
            "/api/predict/multimodal",
            data={
                "face_sample": "happy_PrivateTest_10077120.jpg",
                "audio_sample": "tess_happy.wav",
                "face_weight": 0.5,
                "audio_weight": 0.5,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["fused_emotion"], "Happy")
        self.assertTrue(data["is_congruent"])
        self.assertIn("insight", data)

    def test_06_frontend_static_serving(self):
        """Test GET / serves React index.html."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("<html", res.text.lower())


if __name__ == "__main__":
    unittest.main()
