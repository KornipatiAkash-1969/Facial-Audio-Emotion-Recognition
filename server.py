"""
FastAPI REST API Server for Multimodal Emotion Recognition.
Provides high-performance endpoints for Facial, Audio, and Multimodal Emotion Analysis.
"""

import base64
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import (
    ASSETS_DIR,
    FACE_SAMPLES_DIR,
    AUDIO_SAMPLES_DIR,
    EMOTIONS,
    EMOTION_HEX_COLORS,
    EMOJI_FILES,
    normalize_emotion,
)
from src.facial_detector import FacialEmotionDetector
from src.audio_detector import AudioEmotionDetector
from src.multimodal_fusion import MultimodalEmotionFusion

# Initialize FastAPI Application
app = FastAPI(
    title="Multimodal Emotion Recognition API",
    description="REST API for Facial Expression Recognition, Audio Emotion Recognition, and Multimodal Affective Fusion.",
    version="1.0.0",
)

# Enable CORS for React Frontend (Vite default: 5173, Create-React-App: 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Project Assets Directory for static files (emojis, banners, sample audios)
if ASSETS_DIR.exists():
    app.mount("/media", StaticFiles(directory=str(ASSETS_DIR)), name="media")

# Global Detector Instances (Lazy or Startup Initialized)
face_detector: Optional[FacialEmotionDetector] = None
audio_detector: Optional[AudioEmotionDetector] = None
fusion_engine = MultimodalEmotionFusion()


def get_face_detector() -> FacialEmotionDetector:
    """Get or lazily initialize the FacialEmotionDetector."""
    global face_detector
    if face_detector is None:
        face_detector = FacialEmotionDetector()
    return face_detector


def get_audio_detector() -> AudioEmotionDetector:
    """Get or lazily initialize the AudioEmotionDetector."""
    global audio_detector
    if audio_detector is None:
        audio_detector = AudioEmotionDetector()
    return audio_detector


@app.on_event("startup")
def startup_event():
    """Initialize ML models at server startup."""
    try:
        get_face_detector()
        print("[OK] FacialEmotionDetector initialized successfully.")
    except Exception as err:
        print(f"[WARN] Facial detector initialization warning: {err}")

    try:
        get_audio_detector()
        print("[OK] AudioEmotionDetector initialized successfully.")
    except Exception as err:
        print(f"[WARN] Audio detector initialization warning: {err}")


# =============================================================================
# Helper Utilities
# =============================================================================
def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR numpy array."""
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image from provided data.")
    return frame


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 data URI (e.g. from browser webcam canvas) into OpenCV BGR."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    return decode_image_bytes(img_data)


def encode_image_base64(frame: np.ndarray) -> str:
    """Encode an OpenCV BGR image into a base64 JPEG data URI."""
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


# =============================================================================
# Pydantic Request Models
# =============================================================================
class Base64FaceRequest(BaseModel):
    image: str  # Base64 data URI from webcam or canvas


# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/api/status")
def get_system_status():
    """Return backend health and model readiness status."""
    fd = None
    ad = None
    try:
        fd = get_face_detector()
    except Exception:
        pass
    try:
        ad = get_audio_detector()
    except Exception:
        pass

    return {
        "status": "online",
        "models": {
            "face_detector_ready": fd.is_ready if fd else False,
            "audio_detector_ready": ad.is_ready if ad else False,
        },
        "emotions": EMOTIONS,
        "emotion_colors": EMOTION_HEX_COLORS,
    }


@app.get("/api/samples")
def get_sample_files():
    """Return available sample faces and audio clips for quick testing."""
    sample_faces = []
    if FACE_SAMPLES_DIR.exists():
        for f in FACE_SAMPLES_DIR.glob("*.*"):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                sample_faces.append({
                    "name": f.name,
                    "url": f"/media/samples/faces/{f.name}",
                })

    sample_audios = []
    if AUDIO_SAMPLES_DIR.exists():
        for f in AUDIO_SAMPLES_DIR.glob("*.wav"):
            sample_audios.append({
                "name": f.name,
                "url": f"/media/samples/audio/{f.name}",
            })

    return {
        "faces": sample_faces,
        "audios": sample_audios,
    }


@app.post("/api/predict/face")
async def predict_face(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
):
    """
    Predict facial emotion from an uploaded image file or sample name.
    Returns detected emotion, confidence distribution, and annotated bounding box image.
    """
    try:
        detector = get_face_detector()
    except Exception:
        raise HTTPException(status_code=503, detail="Facial emotion model is not loaded.")

    try:
        if sample_name:
            target = FACE_SAMPLES_DIR / sample_name
            if not target.exists():
                raise HTTPException(status_code=404, detail=f"Sample face not found: {sample_name}")
            frame = detector.read_image(target)
        elif file:
            content = await file.read()
            frame = decode_image_bytes(content)
        else:
            raise HTTPException(status_code=400, detail="Must provide an image file or sample_name.")

        res = detector.predict(frame, annotate=True)
        annotated_b64 = encode_image_base64(res["annotated_frame"])

        return {
            "success": True,
            "num_faces": res["num_faces"],
            "primary_emotion": res["primary_emotion"],
            "primary_confidence": res["primary_confidence"],
            "probabilities": res["primary_probabilities"],
            "annotated_image": annotated_b64,
            "faces": [
                {
                    "box": f["box"],
                    "emotion": f["emotion"],
                    "confidence": f["confidence"],
                }
                for f in res["faces"]
            ],
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Facial analysis failed: {err}")


@app.post("/api/predict/face-base64")
async def predict_face_base64(payload: Base64FaceRequest):
    """
    Predict facial emotion from a base64-encoded webcam frame.
    Ideal for real-time browser camera snapshots.
    """
    try:
        detector = get_face_detector()
    except Exception:
        raise HTTPException(status_code=503, detail="Facial emotion model is not loaded.")

    try:
        frame = decode_base64_image(payload.image)
        res = detector.predict(frame, annotate=True)
        annotated_b64 = encode_image_base64(res["annotated_frame"])

        return {
            "success": True,
            "num_faces": res["num_faces"],
            "primary_emotion": res["primary_emotion"],
            "primary_confidence": res["primary_confidence"],
            "probabilities": res["primary_probabilities"],
            "annotated_image": annotated_b64,
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Webcam frame prediction failed: {err}")


@app.post("/api/predict/audio")
async def predict_audio(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
):
    """
    Predict emotion from an uploaded audio file (.wav) or preloaded sample audio.
    """
    try:
        detector = get_audio_detector()
    except Exception:
        raise HTTPException(status_code=503, detail="Audio emotion model is not loaded.")

    try:
        if sample_name:
            target = AUDIO_SAMPLES_DIR / sample_name
            if not target.exists():
                raise HTTPException(status_code=404, detail=f"Sample audio not found: {sample_name}")
            res = detector.predict(target)
        elif file:
            suffix = Path(file.filename).suffix if file.filename else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                res = detector.predict(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Must provide an audio file or sample_name.")

        return {
            "success": True,
            "emotion": res["emotion"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "duration_seconds": res["duration_seconds"],
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {err}")


@app.post("/api/predict/multimodal")
async def predict_multimodal(
    face_file: Optional[UploadFile] = File(None),
    face_sample: Optional[str] = Form(None),
    face_base64: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    audio_sample: Optional[str] = Form(None),
    face_weight: float = Form(0.5),
    audio_weight: float = Form(0.5),
):
    """
    Run end-to-end multimodal decision fusion combining facial and acoustic cues.
    """
    try:
        f_det = get_face_detector()
    except Exception:
        raise HTTPException(status_code=503, detail="Facial emotion model is not loaded.")
    try:
        a_det = get_audio_detector()
    except Exception:
        raise HTTPException(status_code=503, detail="Audio emotion model is not loaded.")

    # 1. Process Face
    try:
        if face_base64:
            face_frame = decode_base64_image(face_base64)
        elif face_sample:
            target_face = FACE_SAMPLES_DIR / face_sample
            face_frame = f_det.read_image(target_face)
        elif face_file:
            content = await face_file.read()
            face_frame = decode_image_bytes(content)
        else:
            raise HTTPException(status_code=400, detail="Must provide face_file, face_sample, or face_base64.")

        face_res = f_det.predict(face_frame, annotate=True)
        annotated_b64 = encode_image_base64(face_res["annotated_frame"])
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Facial preprocessing failed: {err}")

    # 2. Process Audio
    try:
        if audio_sample:
            target_audio = AUDIO_SAMPLES_DIR / audio_sample
            audio_res = a_det.predict(target_audio)
        elif audio_file:
            suffix = Path(audio_file.filename).suffix if audio_file.filename else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await audio_file.read()
                tmp.write(content)
                tmp_path = tmp.name
            try:
                audio_res = a_det.predict(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Must provide audio_file or audio_sample.")
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Audio preprocessing failed: {err}")

    # 3. Decision-Level Fusion
    try:
        fusion_res = fusion_engine.fuse(
            face_result=face_res,
            audio_result=audio_res,
            face_weight=face_weight,
            audio_weight=audio_weight,
        )

        return {
            "success": True,
            "fused_emotion": fusion_res["fused_emotion"],
            "fused_confidence": fusion_res["fused_confidence"],
            "is_congruent": fusion_res["is_congruent"],
            "congruency_score": fusion_res["congruency_score"],
            "face": fusion_res["face"],
            "audio": fusion_res["audio"],
            "combined_probabilities": fusion_res["combined_probabilities"],
            "insight": fusion_res["insight"],
            "annotated_face_image": annotated_b64,
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Multimodal fusion calculation failed: {err}")


# Serve compiled React Frontend if available
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Multimodal Emotion Recognition FastAPI Server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)

