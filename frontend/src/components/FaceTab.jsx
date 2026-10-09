import React, { useState, useRef, useEffect } from 'react';
import { predictFace, predictFaceBase64 } from '../services/api';
import EmotionBarChart from './EmotionBarChart';

export default function FaceTab({ samples, onAddHistory, setMultimodalFace }) {
  const [mode, setMode] = useState('upload'); // 'upload' | 'webcam' | 'sample'
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedSample, setSelectedSample] = useState('');
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Webcam state
  const videoRef = useRef(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [autoPredict, setAutoPredict] = useState(false);
  const streamRef = useRef(null);
  const autoPredictInterval = useRef(null);

  // Stop camera when unmounting
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  // Handle webcam activation
  const startWebcam = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsWebcamActive(true);
    } catch (err) {
      setError(`Camera access error: ${err.message}. Please check browser permissions.`);
    }
  };

  const stopWebcam = () => {
    if (autoPredictInterval.current) {
      clearInterval(autoPredictInterval.current);
      autoPredictInterval.current = null;
    }
    setAutoPredict(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
  };

  // Capture frame from webcam canvas
  const captureFrame = () => {
    if (!videoRef.current) return null;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
  };

  const handlePredictWebcam = async () => {
    const dataUri = captureFrame();
    if (!dataUri) return;

    setLoading(true);
    setError(null);
    try {
      const res = await predictFaceBase64(dataUri);
      setResult(res);
      setPreviewUrl(res.annotated_image || dataUri);
      if (setMultimodalFace) {
        setMultimodalFace({ base64: dataUri, name: 'Webcam Capture' });
      }
      if (onAddHistory) {
        onAddHistory({
          modality: 'Face (Webcam)',
          emotion: res.primary_emotion,
          confidence: res.primary_confidence,
          source: 'Webcam Stream',
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Toggle periodic auto prediction
  useEffect(() => {
    if (autoPredict && isWebcamActive) {
      autoPredictInterval.current = setInterval(() => {
        handlePredictWebcam();
      }, 1200);
    } else if (autoPredictInterval.current) {
      clearInterval(autoPredictInterval.current);
      autoPredictInterval.current = null;
    }
    return () => {
      if (autoPredictInterval.current) clearInterval(autoPredictInterval.current);
    };
  }, [autoPredict, isWebcamActive]);

  // Handle file upload
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePredictFile = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictFace(selectedFile);
      setResult(res);
      if (res.annotated_image) {
        setPreviewUrl(res.annotated_image);
      }
      if (setMultimodalFace) {
        setMultimodalFace({ file: selectedFile, name: selectedFile.name });
      }
      if (onAddHistory) {
        onAddHistory({
          modality: 'Face (Image)',
          emotion: res.primary_emotion,
          confidence: res.primary_confidence,
          source: selectedFile.name,
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle sample selection
  const handleSelectSample = async (sampleName) => {
    setSelectedSample(sampleName);
    const sampleObj = samples?.faces?.find((f) => f.name === sampleName);
    if (sampleObj) {
      setPreviewUrl(sampleObj.url);
    }
    setLoading(true);
    setError(null);
    try {
      const res = await predictFace(sampleName);
      setResult(res);
      if (res.annotated_image) {
        setPreviewUrl(res.annotated_image);
      }
      if (setMultimodalFace) {
        setMultimodalFace({ sample: sampleName, name: sampleName });
      }
      if (onAddHistory) {
        onAddHistory({
          modality: 'Face (Sample)',
          emotion: res.primary_emotion,
          confidence: res.primary_confidence,
          source: sampleName,
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tab-container">
      {/* Mode Selector Toolbar */}
      <div className="toolbar-card">
        <div className="button-group">
          <button
            className={`btn-mode ${mode === 'upload' ? 'btn-active' : ''}`}
            onClick={() => {
              stopWebcam();
              setMode('upload');
            }}
          >
            📁 Upload Photo
          </button>
          <button
            className={`btn-mode ${mode === 'webcam' ? 'btn-active' : ''}`}
            onClick={() => {
              setMode('webcam');
              startWebcam();
            }}
          >
            📷 Live Webcam
          </button>
          <button
            className={`btn-mode ${mode === 'sample' ? 'btn-active' : ''}`}
            onClick={() => {
              stopWebcam();
              setMode('sample');
            }}
          >
            🖼️ Preloaded Samples
          </button>
        </div>

        {mode === 'upload' && (
          <div className="input-group">
            <input
              type="file"
              id="face-file-input"
              accept="image/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <label htmlFor="face-file-input" className="btn btn-secondary">
              Choose Image File
            </label>
            {selectedFile && <span className="file-name">{selectedFile.name}</span>}
            <button
              className="btn btn-primary"
              onClick={handlePredictFile}
              disabled={!selectedFile || loading}
            >
              {loading ? 'Analyzing...' : 'Predict Facial Emotion'}
            </button>
          </div>
        )}

        {mode === 'sample' && (
          <div className="input-group">
            <label className="input-label">Select Sample Face:</label>
            <select
              className="select-dropdown"
              value={selectedSample}
              onChange={(e) => handleSelectSample(e.target.value)}
            >
              <option value="">-- Choose Sample Face --</option>
              {samples?.faces?.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
        )}

        {mode === 'webcam' && (
          <div className="input-group">
            {!isWebcamActive ? (
              <button className="btn btn-success" onClick={startWebcam}>
                Start Webcam
              </button>
            ) : (
              <>
                <button
                  className="btn btn-primary"
                  onClick={handlePredictWebcam}
                  disabled={loading}
                >
                  {loading ? 'Analyzing...' : 'Capture & Predict'}
                </button>
                <button
                  className={`btn ${autoPredict ? 'btn-warning' : 'btn-outline'}`}
                  onClick={() => setAutoPredict(!autoPredict)}
                >
                  {autoPredict ? '⏹ Stop Auto Stream' : '▶ Auto Predict (1.2s)'}
                </button>
                <button className="btn btn-danger" onClick={stopWebcam}>
                  Stop Webcam
                </button>
              </>
            )}
          </div>
        )}
      </div>

      {error && <div className="alert-error">{error}</div>}

      {/* Main Split Grid */}
      <div className="grid-2col">
        {/* Left Column: Visual Display Card */}
        <div className="card display-card">
          <h3 className="card-heading">Visual Feed & Detection Preview</h3>
          <div className="media-viewer">
            {mode === 'webcam' && isWebcamActive && (
              <div className="webcam-wrapper">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="video-element"
                />
                {autoPredict && <span className="badge-live">● LIVE STREAMING</span>}
              </div>
            )}

            {(!isWebcamActive || mode !== 'webcam') && previewUrl && (
              <img src={previewUrl} alt="Face Preview" className="image-preview" />
            )}

            {!isWebcamActive && !previewUrl && (
              <div className="placeholder-box">
                <span className="placeholder-icon">👤</span>
                <p>Upload a photo, choose a sample, or start the webcam to begin facial recognition.</p>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Emotion Metrics Card */}
        <div className="card metrics-card">
          <h3 className="card-heading">Facial Emotion Analysis</h3>

          {result ? (
            <div className="metrics-content">
              <div className="top-result-badge">
                <span className="emotion-title">{result.primary_emotion}</span>
                <span className="confidence-pill">
                  {(result.primary_confidence * 100).toFixed(1)}% Confidence
                </span>
              </div>

              <div className="info-chip">
                <span>Faces Detected: {result.num_faces}</span>
              </div>

              <EmotionBarChart
                probabilities={result.probabilities}
                topEmotion={result.primary_emotion}
              />
            </div>
          ) : (
            <div className="placeholder-box">
              <span className="placeholder-icon">📊</span>
              <p>Awaiting facial image input to generate emotion probability metrics.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
