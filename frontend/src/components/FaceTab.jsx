import React, { useState, useRef, useEffect, useCallback } from 'react';
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
  const [isDragOver, setIsDragOver] = useState(false);

  const fileInputRef = useRef(null);

  // Webcam state
  const videoRef = useRef(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [autoPredict, setAutoPredict] = useState(false);
  const [webcamView, setWebcamView] = useState('live'); // 'live' | 'snapshot'
  const streamRef = useRef(null);
  const autoPredictInterval = useRef(null);
  const isPredictingRef = useRef(false);

  // Safely stop all webcam media tracks and release camera hardware
  const stopWebcamTracks = useCallback(() => {
    if (autoPredictInterval.current) {
      clearInterval(autoPredictInterval.current);
      autoPredictInterval.current = null;
    }
    setAutoPredict(false);

    // Stop all tracks on streamRef
    if (streamRef.current) {
      try {
        streamRef.current.getTracks().forEach((track) => {
          track.enabled = false;
          track.stop();
        });
      } catch (e) {
        console.error('Error stopping stream track:', e);
      }
      streamRef.current = null;
    }

    // Stop all tracks on video element srcObject
    if (videoRef.current && videoRef.current.srcObject) {
      try {
        const stream = videoRef.current.srcObject;
        if (stream && stream.getTracks) {
          stream.getTracks().forEach((track) => {
            track.enabled = false;
            track.stop();
          });
        }
      } catch (e) {
        console.error('Error stopping video srcObject track:', e);
      }
      videoRef.current.srcObject = null;
    }
  }, []);

  const stopWebcam = useCallback(() => {
    stopWebcamTracks();
    setIsWebcamActive(false);
    setWebcamView('live');
  }, [stopWebcamTracks]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      stopWebcamTracks();
    };
  }, [stopWebcamTracks]);

  // Handle webcam activation
  const startWebcam = async () => {
    setError(null);
    stopWebcamTracks();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError('Webcam access is not supported by your browser or environment.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;
      setIsWebcamActive(true);
      setWebcamView('live');

      // If videoRef is already mounted, attach stream immediately
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(() => {});
      }
    } catch (err) {
      setIsWebcamActive(false);
      setError(`Camera access error: ${err.message}. Please check browser camera permissions.`);
    }
  };

  // Synchronize stream with videoRef whenever isWebcamActive changes
  useEffect(() => {
    if (isWebcamActive && streamRef.current && videoRef.current) {
      if (videoRef.current.srcObject !== streamRef.current) {
        videoRef.current.srcObject = streamRef.current;
      }
      videoRef.current.play().catch((err) => {
        console.warn('Video play interrupted:', err);
      });
    }
  }, [isWebcamActive]);

  // Capture frame from webcam canvas
  const captureFrame = () => {
    const v = videoRef.current;
    if (!v) return null;
    if (v.readyState < 2) return null; // HAVE_CURRENT_DATA
    const canvas = document.createElement('canvas');
    canvas.width = v.videoWidth || 640;
    canvas.height = v.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
  };

  const handlePredictWebcam = async (isBackground = false) => {
    if (isPredictingRef.current) return;
    const dataUri = captureFrame();
    if (!dataUri) return;

    isPredictingRef.current = true;
    if (!isBackground) setLoading(true);
    setError(null);
    try {
      const res = await predictFaceBase64(dataUri);
      setResult(res);
      if (res.annotated_image) {
        setPreviewUrl(res.annotated_image);
      }
      if (setMultimodalFace) {
        setMultimodalFace({ base64: dataUri, name: 'Webcam Capture' });
      }
      if (onAddHistory && !isBackground) {
        onAddHistory({
          modality: 'Face (Webcam)',
          emotion: res.primary_emotion,
          confidence: res.primary_confidence,
          source: 'Webcam Capture',
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      isPredictingRef.current = false;
      if (!isBackground) setLoading(false);
    }
  };

  // Auto-predict interval for webcam
  useEffect(() => {
    if (autoPredict && isWebcamActive) {
      autoPredictInterval.current = setInterval(() => {
        handlePredictWebcam(true);
      }, 1200);
    } else if (autoPredictInterval.current) {
      clearInterval(autoPredictInterval.current);
      autoPredictInterval.current = null;
    }
    return () => {
      if (autoPredictInterval.current) clearInterval(autoPredictInterval.current);
    };
  }, [autoPredict, isWebcamActive]);

  // Handle file analysis (called automatically upon file selection or drop)
  const processImageFile = async (file) => {
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    try {
      const res = await predictFace(file);
      setResult(res);
      if (res.annotated_image) {
        setPreviewUrl(res.annotated_image);
      }
      if (setMultimodalFace) {
        setMultimodalFace({ file, name: file.name });
      }
      if (onAddHistory) {
        onAddHistory({
          modality: 'Face (Image)',
          emotion: res.primary_emotion,
          confidence: res.primary_confidence,
          source: file.name,
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      processImageFile(file);
    }
    // Reset file input value so re-selecting the exact same file triggers onChange
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && (file.type.startsWith('image/') || /\.(jpg|jpeg|png|webp|bmp)$/i.test(file.name))) {
      processImageFile(file);
    } else if (file) {
      setError('Please drop a valid image file (.jpg, .png, .jpeg, .webp, .bmp).');
    }
  };

  // Sample selection
  const handleSelectSample = async (sampleName) => {
    if (!sampleName) return;
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
              ref={fileInputRef}
              accept="image/*,.jpg,.jpeg,.png,.webp,.bmp"
              onChange={handleFileInputChange}
              style={{ display: 'none' }}
            />
            <button
              className="btn btn-primary"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
            >
              {loading ? 'Analyzing Image...' : '📂 Choose Image File'}
            </button>
            {selectedFile && <span className="file-name">{selectedFile.name}</span>}
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
                📷 Turn On Camera
              </button>
            ) : (
              <>
                <button
                  className="btn btn-primary"
                  onClick={() => handlePredictWebcam(false)}
                  disabled={loading}
                >
                  {loading ? 'Analyzing...' : '📸 Capture & Predict'}
                </button>
                <button
                  className={`btn ${autoPredict ? 'btn-warning' : 'btn-outline'}`}
                  onClick={() => setAutoPredict(!autoPredict)}
                >
                  {autoPredict ? '⏹ Stop Auto Stream' : '▶ Auto Stream (1.2s)'}
                </button>
                <button className="btn btn-danger" onClick={stopWebcam}>
                  🛑 Turn Off Camera
                </button>
              </>
            )}
          </div>
        )}
      </div>

      {error && <div className="alert-error">{error}</div>}

      {/* Main Split Grid */}
      <div className="grid-2col">
        {/* Left Column: Visual Display / Dropzone Card */}
        <div className="card display-card">
          <div className="card-header-flex">
            <h3 className="card-heading">Visual Feed & Detection Preview</h3>
            {mode === 'upload' && (
              <span className="hint-pill">Drag & drop or click box to upload</span>
            )}
            {mode === 'webcam' && isWebcamActive && (
              <span
                className="hint-pill"
                style={{ borderColor: 'rgba(16, 185, 129, 0.4)', color: '#34D399' }}
              >
                Camera Active
              </span>
            )}
          </div>

          <div
            className={`media-viewer ${isDragOver ? 'drag-over' : ''} ${
              mode === 'upload' && !previewUrl ? 'clickable-zone' : ''
            }`}
            onDragOver={mode === 'upload' ? handleDragOver : undefined}
            onDragLeave={mode === 'upload' ? handleDragLeave : undefined}
            onDrop={mode === 'upload' ? handleDrop : undefined}
            onClick={() => {
              if (mode === 'upload' && !loading && !previewUrl) {
                fileInputRef.current?.click();
              }
            }}
          >
            {/* WEBCAM MODE: ACTIVE */}
            {mode === 'webcam' && isWebcamActive && (
              <div className="webcam-wrapper">
                {webcamView === 'live' || !previewUrl ? (
                  <video
                    ref={(el) => {
                      videoRef.current = el;
                      if (el && streamRef.current && el.srcObject !== streamRef.current) {
                        el.srcObject = streamRef.current;
                        el.play().catch(() => {});
                      }
                    }}
                    autoPlay
                    playsInline
                    muted
                    className="video-element"
                  />
                ) : (
                  <img src={previewUrl} alt="Annotated Snapshot" className="image-preview" />
                )}

                {/* Status Badges */}
                <div className="badge-live-camera">
                  <span>●</span> LIVE CAMERA
                </div>
                {autoPredict && (
                  <div
                    className="badge-live-camera badge-stream-active"
                    style={{ left: '125px' }}
                  >
                    ⚡ AUTO-PREDICT (1.2s)
                  </div>
                )}

                {/* Toggle Live Feed vs Snapshot if snapshot exists */}
                {previewUrl && (
                  <div className="webcam-view-toggle">
                    <button
                      className={`btn-toggle-view ${webcamView === 'live' ? 'active' : ''}`}
                      onClick={() => setWebcamView('live')}
                    >
                      📹 Live Feed
                    </button>
                    <button
                      className={`btn-toggle-view ${webcamView === 'snapshot' ? 'active' : ''}`}
                      onClick={() => setWebcamView('snapshot')}
                    >
                      🖼️ Detection Box
                    </button>
                  </div>
                )}

                {/* Overlay with current emotion */}
                {result && (
                  <div className="webcam-emotion-overlay">
                    <span style={{ fontSize: '18px' }}>
                      {result.primary_emotion === 'Happy'
                        ? '😊'
                        : result.primary_emotion === 'Angry'
                        ? '😠'
                        : result.primary_emotion === 'Sad'
                        ? '😢'
                        : result.primary_emotion === 'Surprise'
                        ? '😲'
                        : result.primary_emotion === 'Fear'
                        ? '😨'
                        : result.primary_emotion === 'Disgust'
                        ? '🤢'
                        : '😐'}
                    </span>
                    <span>
                      {result.primary_emotion} ({(result.primary_confidence * 100).toFixed(0)}%)
                    </span>
                  </div>
                )}

                {loading && (
                  <div className="loading-overlay">
                    <span className="spinner">⏳</span>
                    <p>Detecting Faces & Analyzing Emotion...</p>
                  </div>
                )}
              </div>
            )}

            {/* WEBCAM MODE: INACTIVE */}
            {mode === 'webcam' && !isWebcamActive && (
              <div className="placeholder-box">
                <span className="placeholder-icon">📷</span>
                <strong style={{ fontSize: '15px', color: '#E2E8F0' }}>
                  Live Webcam is Currently Off
                </strong>
                <p style={{ marginTop: '8px', maxWidth: '340px' }}>
                  Click below to activate your camera for real-time facial expression and emotion recognition.
                </p>
                <button
                  className="btn btn-success"
                  style={{ marginTop: '16px' }}
                  onClick={startWebcam}
                >
                  📷 Turn On Camera
                </button>
              </div>
            )}

            {/* UPLOAD OR SAMPLE MODE: WITH PREVIEW */}
            {mode !== 'webcam' && previewUrl && (
              <div className="preview-container">
                <img src={previewUrl} alt="Face Preview" className="image-preview" />
                {mode === 'upload' && (
                  <button
                    className="btn-change-image"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    🔄 Change Image
                  </button>
                )}
                {loading && (
                  <div className="loading-overlay">
                    <span className="spinner">⏳</span>
                    <p>Detecting Faces & Classifying Emotions...</p>
                  </div>
                )}
              </div>
            )}

            {/* UPLOAD OR SAMPLE MODE: NO PREVIEW */}
            {mode !== 'webcam' && !previewUrl && (
              <div className="placeholder-box">
                <span className="placeholder-icon">📁</span>
                <strong style={{ fontSize: '15px', color: '#E2E8F0' }}>
                  Click to Browse or Drag & Drop Image Here
                </strong>
                <p style={{ marginTop: '6px' }}>Supports JPG, PNG, WEBP, and BMP</p>
                {mode === 'upload' && (
                  <button
                    className="btn btn-primary"
                    style={{ marginTop: '14px' }}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    📂 Browse Files
                  </button>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Emotion Metrics Card */}
        <div className="card metrics-card">
          <h3 className="card-heading">Facial Emotion Analysis</h3>

          {loading && !result ? (
            <div className="placeholder-box">
              <span className="placeholder-icon">⏳</span>
              <p>Analyzing facial expressions...</p>
            </div>
          ) : result ? (
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
              <p>Select or upload an image to view emotion predictions and probability scores.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
