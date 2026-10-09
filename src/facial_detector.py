"""
Facial Emotion Recognition Detector.
Provides face detection using OpenCV Haar Cascades and emotion classification
using a deep convolutional neural network.
"""

from pathlib import Path
from typing import Union, List, Dict, Any, Tuple, Optional
import os
import cv2
import numpy as np

from src.config import (
    FACIAL_MODEL_PATH,
    HAAR_CASCADE_PATH,
    EMOTIONS,
    EMOTION_BGR_COLORS,
    normalize_emotion,
)


class FacialEmotionDetector:
    """Detects faces in images/frames and classifies facial emotion."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        cascade_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_path) if model_path else FACIAL_MODEL_PATH
        self.cascade_path = Path(cascade_path) if cascade_path else HAAR_CASCADE_PATH

        self.model = None
        self.face_cascade = None
        self.labels = EMOTIONS

        self._load_cascade()
        self._load_model()

    def _load_cascade(self) -> None:
        """Load Haar Cascade classifier."""
        if not self.cascade_path.exists():
            raise FileNotFoundError(
                f"Haar cascade XML not found at {self.cascade_path}"
            )
        self.face_cascade = cv2.CascadeClassifier(str(self.cascade_path))
        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from {self.cascade_path}"
            )

    def _load_model(self) -> None:
        """Load Keras CNN emotion classification model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Facial emotion model not found at {self.model_path}"
            )
        try:
            import keras
            self.model = keras.models.load_model(str(self.model_path), compile=False)
        except Exception as err:
            raise RuntimeError(
                f"Error loading Keras facial model from {self.model_path}: {err}"
            ) from err

    @property
    def is_ready(self) -> bool:
        """Check if both cascade and model are loaded and ready."""
        return self.face_cascade is not None and self.model is not None

    def read_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Convert image input (filepath or array) to BGR OpenCV image."""
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image file not found: {path_str}")
            # Use imdecode with numpy to safely handle Unicode/Windows paths
            img_bytes = np.fromfile(path_str, dtype=np.uint8)
            frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not decode image at {path_str}")
            return frame
        elif isinstance(image_input, np.ndarray):
            return image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect face bounding boxes (x, y, w, h) in a BGR frame."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def predict_face_crop(self, sub_face: np.ndarray) -> Dict[str, Any]:
        """Predict emotion probabilities for an already cropped face region."""
        if sub_face.ndim == 3:
            gray = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
        else:
            gray = sub_face

        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        preds = self.model.predict(reshaped, verbose=0)[0]
        prob_dict = {
            self.labels[i]: float(preds[i]) for i in range(len(self.labels))
        }

        top_idx = int(np.argmax(preds))
        top_emotion = self.labels[top_idx]
        top_confidence = float(preds[top_idx])

        return {
            "emotion": top_emotion,
            "confidence": top_confidence,
            "probabilities": prob_dict,
        }

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        annotate: bool = False,
    ) -> Dict[str, Any]:
        """
        Run end-to-end facial emotion detection on an image or frame.
        Returns face detections, emotion probabilities, and optional annotated image.
        """
        frame = self.read_image(image_input)
        faces = self.detect_faces(frame)

        face_results = []
        if len(faces) == 0:
            # Fallback: if no face cascade hit (e.g. tight crop or stylized photo),
            # test inference on the central crop of the image
            h, w = frame.shape[:2]
            crop_size = min(h, w)
            cy, cx = h // 2, w // 2
            sub = frame[
                max(0, cy - crop_size // 2) : min(h, cy + crop_size // 2),
                max(0, cx - crop_size // 2) : min(w, cx + crop_size // 2),
            ]
            crop_pred = self.predict_face_crop(sub)
            face_results.append({
                "box": (0, 0, w, h),
                "emotion": crop_pred["emotion"],
                "confidence": crop_pred["confidence"],
                "probabilities": crop_pred["probabilities"],
                "is_fallback": True,
            })
            num_faces_detected = 0
        else:
            num_faces_detected = len(faces)
            for (x, y, w, h) in faces:
                sub = frame[y : y + h, x : x + w]
                pred = self.predict_face_crop(sub)
                face_results.append({
                    "box": (x, y, w, h),
                    "emotion": pred["emotion"],
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"],
                    "is_fallback": False,
                })

        # Primary face: largest bounding box area
        primary_face = max(face_results, key=lambda f: f["box"][2] * f["box"][3])

        annotated_frame = None
        if annotate:
            annotated_frame = self.annotate_frame(frame, face_results)

        return {
            "success": True,
            "num_faces": num_faces_detected,
            "faces": face_results,
            "primary_emotion": primary_face["emotion"],
            "primary_confidence": primary_face["confidence"],
            "primary_probabilities": primary_face["probabilities"],
            "annotated_frame": annotated_frame,
            "original_frame": frame,
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        face_results: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw styled bounding boxes and emotion banners on a BGR frame."""
        annotated = frame.copy()
        for item in face_results:
            if item.get("is_fallback") and len(face_results) == 1:
                # Don't draw giant full-screen box if fallback
                continue

            x, y, w, h = item["box"]
            emotion = item["emotion"]
            conf = item["confidence"]
            color = EMOTION_BGR_COLORS.get(emotion, (0, 255, 0))

            # Outer bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Top label badge background
            label_text = f"{emotion} ({conf * 100:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )

            tag_y1 = max(0, y - th - 12)
            tag_y2 = y
            tag_x2 = min(annotated.shape[1], x + tw + 10)

            # Draw badge background
            cv2.rectangle(
                annotated,
                (x, tag_y1),
                (tag_x2, tag_y2),
                color,
                -1,
            )
            # Text inside badge
            cv2.putText(
                annotated,
                label_text,
                (x + 5, tag_y2 - 6),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return annotated
