/**
 * API Client for interacting with the FastAPI Emotion Recognition Backend
 */

const API_BASE = "";

export async function fetchSystemStatus() {
  const res = await fetch(`${API_BASE}/api/status`);
  if (!res.ok) throw new Error("Failed to fetch system status");
  return res.json();
}

export async function fetchSampleFiles() {
  const res = await fetch(`${API_BASE}/api/samples`);
  if (!res.ok) throw new Error("Failed to fetch sample files");
  return res.json();
}

export async function predictFace(fileOrSample) {
  const formData = new FormData();
  if (typeof fileOrSample === "string") {
    formData.append("sample_name", fileOrSample);
  } else {
    formData.append("file", fileOrSample);
  }

  const res = await fetch(`${API_BASE}/api/predict/face`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Facial prediction failed" }));
    throw new Error(err.detail || "Facial prediction failed");
  }
  return res.json();
}

export async function predictFaceBase64(base64Image) {
  const res = await fetch(`${API_BASE}/api/predict/face-base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: base64Image }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Webcam prediction failed" }));
    throw new Error(err.detail || "Webcam prediction failed");
  }
  return res.json();
}

export async function predictAudio(fileOrSample) {
  const formData = new FormData();
  if (typeof fileOrSample === "string") {
    formData.append("sample_name", fileOrSample);
  } else {
    formData.append("file", fileOrSample);
  }

  const res = await fetch(`${API_BASE}/api/predict/audio`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Audio prediction failed" }));
    throw new Error(err.detail || "Audio prediction failed");
  }
  return res.json();
}

export async function predictMultimodal({
  faceFile,
  faceSample,
  faceBase64,
  audioFile,
  audioSample,
  faceWeight = 0.5,
  audioWeight = 0.5,
}) {
  const formData = new FormData();
  if (faceBase64) formData.append("face_base64", faceBase64);
  else if (faceFile) formData.append("face_file", faceFile);
  else if (faceSample) formData.append("face_sample", faceSample);

  if (audioFile) formData.append("audio_file", audioFile);
  else if (audioSample) formData.append("audio_sample", audioSample);

  formData.append("face_weight", faceWeight);
  formData.append("audio_weight", audioWeight);

  const res = await fetch(`${API_BASE}/api/predict/multimodal`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Multimodal fusion failed" }));
    throw new Error(err.detail || "Multimodal fusion failed");
  }
  return res.json();
}
