"""
Audio Emotion Recognition Detector.
Extracts acoustic features (MFCCs) from speech signals and classifies emotional state.
"""

from pathlib import Path
from typing import Union, Dict, Any, Optional
import os
import numpy as np
import soundfile as sf
import librosa
import joblib

from src.config import (
    AUDIO_MODEL_PATH,
    AUDIO_KERAS_MODEL_PATH,
    EMOTIONS,
    normalize_emotion,
)


class AudioEmotionDetector:
    """Classifies speech audio signals into emotion classes."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else AUDIO_MODEL_PATH
        self.model = None
        self.scaler = None
        self.classes = EMOTIONS
        self.model_type = "joblib"

        self._load_model()

    def _load_model(self) -> None:
        """Load trained audio model and preprocessor."""
        if self.model_path.exists() and self.model_path.suffix == ".joblib":
            data = joblib.load(str(self.model_path))
            self.model = data["model"]
            self.scaler = data.get("scaler")
            raw_classes = data.get("classes", EMOTIONS)
            self.classes = [normalize_emotion(c) for c in raw_classes]
            self.model_type = "joblib"
        elif AUDIO_KERAS_MODEL_PATH.exists():
            import keras
            self.model = keras.models.load_model(str(AUDIO_KERAS_MODEL_PATH), compile=False)
            self.model_type = "keras"
        else:
            raise FileNotFoundError(
                f"Audio emotion model not found at {self.model_path} or {AUDIO_KERAS_MODEL_PATH}. "
                "Please run `python src/train_audio.py` to train an audio model."
            )

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None

    def load_audio(
        self, audio_input: Union[str, Path, np.ndarray], target_sr: int = 22050
    ) -> tuple[np.ndarray, int]:
        """Load audio input into a 1D mono float numpy array and sample rate."""
        if isinstance(audio_input, (str, Path)):
            path_str = str(audio_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Audio file not found: {path_str}")
            try:
                data, sr = sf.read(path_str)
            except Exception:
                data, sr = librosa.load(path_str, sr=target_sr)
        elif isinstance(audio_input, np.ndarray):
            data = audio_input
            sr = target_sr
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        # Convert to mono if multi-channel
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Normalize amplitude to prevent clipping distortion
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

        return data.astype(np.float32), sr

    def extract_features(
        self,
        audio_input: Union[str, Path, np.ndarray],
        n_mfcc: int = 40,
        sr: int = 22050,
    ) -> tuple[np.ndarray, float]:
        """Extract mean MFCC feature vector and audio duration in seconds."""
        y, sample_rate = self.load_audio(audio_input, target_sr=sr)
        duration = len(y) / float(sample_rate)

        # Extract Mel-Frequency Cepstral Coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        return mfcc_mean, duration

    def predict(
        self,
        audio_input: Union[str, Path, np.ndarray],
    ) -> Dict[str, Any]:
        """Predict emotion probabilities for an audio file or signal."""
        features, duration = self.extract_features(audio_input)

        if self.model_type == "joblib":
            if self.scaler is not None:
                feat_scaled = self.scaler.transform([features])
            else:
                feat_scaled = [features]
            probs_raw = self.model.predict_proba(feat_scaled)[0]
        else:
            # Keras model prediction
            feat_in = features[np.newaxis, np.newaxis, :]
            probs_raw = self.model.predict(feat_in, verbose=0)[0]

        # Map to canonical emotions
        prob_dict = {}
        for emotion in EMOTIONS:
            prob_dict[emotion] = 0.0

        for idx, cls_name in enumerate(self.classes):
            canon = normalize_emotion(cls_name)
            prob_dict[canon] = float(probs_raw[idx])

        # Find top predicted emotion
        top_emotion = max(prob_dict, key=prob_dict.get)
        top_confidence = prob_dict[top_emotion]

        return {
            "success": True,
            "emotion": top_emotion,
            "confidence": top_confidence,
            "probabilities": prob_dict,
            "duration_seconds": round(duration, 2),
        }

    @staticmethod
    def record_microphone(
        duration: float = 3.0,
        sample_rate: int = 22050,
        save_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Record audio from the microphone for a specified duration."""
        try:
            import sounddevice as sd
        except ImportError as err:
            raise ImportError(
                "Microphone recording requires `sounddevice`. Install it with `pip install sounddevice`."
            ) from err

        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio_data = np.squeeze(recording)

        if save_path:
            sf.write(str(save_path), audio_data, sample_rate)

        return audio_data

    @staticmethod
    def play_audio(audio_path: Union[str, Path]) -> None:
        """Play audio file using native OS audio or sounddevice."""
        path_str = str(audio_path)
        try:
            # On Windows, winsound provides reliable native playback for WAV without extra processes
            import winsound
            winsound.PlaySound(path_str, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            try:
                import sounddevice as sd
                data, sr = sf.read(path_str)
                sd.play(data, sr)
            except Exception as err:
                print(f"Could not play audio: {err}")
