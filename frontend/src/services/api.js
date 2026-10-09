/**
 * API Client for interacting with the FastAPI Emotion Recognition Backend
 */

export function getApiBase() {
  const custom = typeof window !== 'undefined' ? localStorage.getItem('custom_api_url') : null;
  if (custom && custom.trim()) {
    return custom.trim().replace(/\/+$/, '');
  }
  const envUrl = import.meta.env.VITE_API_URL;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, '');
  }
  if (
    typeof window !== 'undefined' &&
    (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ) {
    return 'http://localhost:8000';
  }
  return 'https://facial-audio-emotion-recognition.onrender.com';
}

export function setCustomApiUrl(url) {
  if (typeof window === 'undefined') return;
  if (!url || !url.trim()) {
    localStorage.removeItem('custom_api_url');
  } else {
    localStorage.setItem('custom_api_url', url.trim().replace(/\/+$/, ''));
  }
}

export function getCustomApiUrl() {
  if (typeof window === 'undefined') return '';
  return localStorage.getItem('custom_api_url') || '';
}

const DEFAULT_SAMPLES = {
  faces: [
    { name: "happy_test.jpg", url: "/media/samples/faces/happy_test.jpg" },
    { name: "sad_test.jpg", url: "/media/samples/faces/sad_test.jpg" },
    { name: "angry_test.jpg", url: "/media/samples/faces/angry_test.jpg" },
    { name: "surprise_test.jpg", url: "/media/samples/faces/surprise_test.jpg" },
    { name: "neutral_test.jpg", url: "/media/samples/faces/neutral_test.jpg" },
    { name: "fear_test.jpg", url: "/media/samples/faces/fear_test.jpg" },
    { name: "disgust_test.jpg", url: "/media/samples/faces/disgust_test.jpg" },
    { name: "multiface_test.jpg", url: "/media/samples/faces/multiface_test.jpg" },
    { name: "happy_PrivateTest_10077120.jpg", url: "/media/samples/faces/happy_PrivateTest_10077120.jpg" },
    { name: "angry_PrivateTest_10131363.jpg", url: "/media/samples/faces/angry_PrivateTest_10131363.jpg" },
    { name: "sad_PrivateTest_10247676.jpg", url: "/media/samples/faces/sad_PrivateTest_10247676.jpg" },
    { name: "surprise_PrivateTest_10072988.jpg", url: "/media/samples/faces/surprise_PrivateTest_10072988.jpg" },
    { name: "fear_PrivateTest_10153550.jpg", url: "/media/samples/faces/fear_PrivateTest_10153550.jpg" },
    { name: "disgust_PrivateTest_11895083.jpg", url: "/media/samples/faces/disgust_PrivateTest_11895083.jpg" },
    { name: "neutral_PrivateTest_10086748.jpg", url: "/media/samples/faces/neutral_PrivateTest_10086748.jpg" },
    { name: "Test 2.jpg", url: "/media/samples/faces/Test 2.jpg" },
    { name: "Test 3.jpg", url: "/media/samples/faces/Test 3.jpg" },
  ],
  audios: [
    { name: "happy_test.wav", url: "/media/samples/audio/happy_test.wav" },
    { name: "sad_test.wav", url: "/media/samples/audio/sad_test.wav" },
    { name: "angry_test.wav", url: "/media/samples/audio/angry_test.wav" },
    { name: "surprise_test.wav", url: "/media/samples/audio/surprise_test.wav" },
    { name: "neutral_test.wav", url: "/media/samples/audio/neutral_test.wav" },
    { name: "fear_test.wav", url: "/media/samples/audio/fear_test.wav" },
    { name: "disgust_test.wav", url: "/media/samples/audio/disgust_test.wav" },
    { name: "speech_test_1.wav", url: "/media/samples/audio/speech_test_1.wav" },
    { name: "speech_test_2.wav", url: "/media/samples/audio/speech_test_2.wav" },
    { name: "speech_test_3.wav", url: "/media/samples/audio/speech_test_3.wav" },
    { name: "tess_happy.wav", url: "/media/samples/audio/tess_happy.wav" },
    { name: "tess_angry.wav", url: "/media/samples/audio/tess_angry.wav" },
    { name: "tess_sad.wav", url: "/media/samples/audio/tess_sad.wav" },
    { name: "tess_surprise.wav", url: "/media/samples/audio/tess_surprise.wav" },
    { name: "tess_fear.wav", url: "/media/samples/audio/tess_fear.wav" },
    { name: "tess_disgust.wav", url: "/media/samples/audio/tess_disgust.wav" },
    { name: "tess_neutral.wav", url: "/media/samples/audio/tess_neutral.wav" },
    { name: "sample 1.wav", url: "/media/samples/audio/sample 1.wav" },
    { name: "sample 2.wav", url: "/media/samples/audio/sample 2.wav" },
  ],
};

export async function fetchSystemStatus() {
  const res = await fetch(`${getApiBase()}/api/status`);
  if (!res.ok) throw new Error("Failed to fetch system status");
  return res.json();
}

export async function fetchSampleFiles() {
  try {
    const res = await fetch(`${getApiBase()}/api/samples`);
    if (res.ok) return await res.json();
  } catch (err) {
    // Fall back to bundled samples if backend is starting or offline
  }
  return DEFAULT_SAMPLES;
}

export async function predictFace(fileOrSample) {
  const formData = new FormData();
  if (typeof fileOrSample === "string") {
    formData.append("sample_name", fileOrSample);
  } else {
    formData.append("file", fileOrSample);
  }

  const res = await fetch(`${getApiBase()}/api/predict/face`, {
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
  const res = await fetch(`${getApiBase()}/api/predict/face-base64`, {
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

  const res = await fetch(`${getApiBase()}/api/predict/audio`, {
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

  const res = await fetch(`${getApiBase()}/api/predict/multimodal`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Multimodal fusion failed" }));
    throw new Error(err.detail || "Multimodal fusion failed");
  }
  return res.json();
}
