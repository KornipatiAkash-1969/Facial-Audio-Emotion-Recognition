"""
Configuration module for Multimodal Emotion Recognition System.
Provides centralized path management, canonical emotion mappings, color palettes,
and model metadata.
"""

from pathlib import Path
import os

# Project Directory Roots
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
EMOJIS_DIR = ASSETS_DIR / "emojis"
BANNERS_DIR = ASSETS_DIR / "banners"
SAMPLES_DIR = ASSETS_DIR / "samples"
AUDIO_SAMPLES_DIR = SAMPLES_DIR / "audio"
FACE_SAMPLES_DIR = SAMPLES_DIR / "faces"
DATA_DIR = PROJECT_ROOT / "data"

# Model File Paths
FACIAL_MODEL_PATH = MODELS_DIR / "facial_emotion_model.h5"
HAAR_CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
AUDIO_MODEL_PATH = MODELS_DIR / "audio_emotion_model.joblib"
AUDIO_KERAS_MODEL_PATH = MODELS_DIR / "audio_emotion_model.keras"
AUDIO_LABELS_PATH = MODELS_DIR / "audio_label_encoder.json"

# Canonical 7 Universal Emotions
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Canonical Casing / Normalization Dictionary
NORMALIZE_EMOTION_MAP = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise",
    "surprised": "Surprise",
    "pleasant_surprised": "Surprise",
    "pleasant_surprise": "Surprise",
    "yaf_angry": "Angry",
    "yaf_disgust": "Disgust",
    "yaf_fear": "Fear",
    "yaf_happy": "Happy",
    "yaf_neutral": "Neutral",
    "yaf_pleasant_surprised": "Surprise",
    "yaf_sad": "Sad",
    "oaf_angry": "Angry",
    "oaf_disgust": "Disgust",
    "oaf_fear": "Fear",
    "oaf_happy": "Happy",
    "oaf_neutral": "Neutral",
    "oaf_pleasant_surprise": "Surprise",
    "oaf_sad": "Sad",
}

# TESS Folder to Standard Emotion
TESS_FOLDER_MAP = {
    "OAF_angry": "Angry",
    "YAF_angry": "Angry",
    "OAF_disgust": "Disgust",
    "YAF_disgust": "Disgust",
    "OAF_Fear": "Fear",
    "YAF_fear": "Fear",
    "OAF_happy": "Happy",
    "YAF_happy": "Happy",
    "OAF_neutral": "Neutral",
    "YAF_neutral": "Neutral",
    "OAF_Pleasant_surprise": "Surprise",
    "YAF_pleasant_surprised": "Surprise",
    "OAF_Sad": "Sad",
    "YAF_sad": "Sad",
}

# Emotion Emoji Filenames
EMOJI_FILES = {
    "Angry": "angry.png",
    "Disgust": "disgust.png",
    "Fear": "fear.png",
    "Happy": "happy.png",
    "Neutral": "neutral.png",
    "Sad": "sad.png",
    "Surprise": "surprised.png",
}

# Color Palettes for GUI and OpenCV visualization
# Hex colors for Tkinter / Web
EMOTION_HEX_COLORS = {
    "Angry": "#E74C3C",     # Crimson Red
    "Disgust": "#27AE60",   # Emerald Green
    "Fear": "#8E44AD",      # Purple
    "Happy": "#F39C12",     # Amber/Gold
    "Neutral": "#7F8C8D",   # Slate Gray
    "Sad": "#2980B9",       # Deep Blue
    "Surprise": "#E67E22",  # Vivid Orange
}

# BGR colors for OpenCV drawing (B, G, R)
EMOTION_BGR_COLORS = {
    "Angry": (60, 76, 231),
    "Disgust": (96, 174, 39),
    "Fear": (173, 68, 142),
    "Happy": (18, 156, 243),
    "Neutral": (141, 140, 127),
    "Sad": (185, 128, 41),
    "Surprise": (34, 126, 230),
}


def normalize_emotion(name: str) -> str:
    """Normalize any emotion label string to canonical Capitalized format."""
    clean = str(name).strip().lower()
    return NORMALIZE_EMOTION_MAP.get(clean, str(name).capitalize())


def get_emoji_path(emotion: str) -> Path | None:
    """Return the absolute Path to the emoji icon for a given emotion."""
    canonical = normalize_emotion(emotion)
    fname = EMOJI_FILES.get(canonical, "neutral.png")
    path = EMOJIS_DIR / fname
    return path if path.exists() else None
