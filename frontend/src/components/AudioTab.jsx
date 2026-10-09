import React, { useState, useRef, useEffect } from 'react';
import { predictAudio } from '../services/api';
import { WavRecorder } from '../utils/wavRecorder';
import EmotionBarChart from './EmotionBarChart';

export default function AudioTab({ samples, onAddHistory, setMultimodalAudio }) {
  const [mode, setMode] = useState('upload'); // 'upload' | 'record' | 'sample'
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedSample, setSelectedSample] = useState('');
  const [audioUrl, setAudioUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const fileInputRef = useRef(null);

  // Microphone recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const recorderRef = useRef(null);
  const timerRef = useRef(null);

  // Clean up recording on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (recorderRef.current) {
        try {
          recorderRef.current.stop();
        } catch (e) {}
      }
    };
  }, []);

  const startRecording = async () => {
    setError(null);
    try {
      const recorder = new WavRecorder();
      await recorder.start();
      recorderRef.current = recorder;
      setIsRecording(true);
      setRecordDuration(0);

      timerRef.current = setInterval(() => {
        setRecordDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setError(`Microphone access error: ${err.message}. Please check browser permissions.`);
    }
  };

  const stopRecording = async () => {
    if (!recorderRef.current) return;
    clearInterval(timerRef.current);
    setIsRecording(false);

    try {
      setLoading(true);
      const wavBlob = await recorderRef.current.stop();
      const wavFile = new File([wavBlob], 'mic_recording.wav', { type: 'audio/wav' });

      setSelectedFile(wavFile);
      setAudioUrl(URL.createObjectURL(wavBlob));

      const res = await predictAudio(wavFile);
      setResult(res);

      if (setMultimodalAudio) {
        setMultimodalAudio({ file: wavFile, name: 'Microphone Recording' });
      }
      if (onAddHistory) {
        onAddHistory({
          modality: 'Audio (Mic)',
          emotion: res.emotion,
          confidence: res.confidence,
          source: 'Microphone Recording',
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      recorderRef.current = null;
    }
  };

  // Automatic analysis upon selecting/dropping audio file
  const processAudioFile = async (file) => {
    if (!file) return;
    setSelectedFile(file);
    setAudioUrl(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    try {
      const res = await predictAudio(file);
      setResult(res);
      if (setMultimodalAudio) {
        setMultimodalAudio({ file, name: file.name });
      }
      if (onAddHistory) {
        onAddHistory({
          modality: 'Audio (File)',
          emotion: res.emotion,
          confidence: res.confidence,
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
      processAudioFile(file);
    }
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
    if (file) {
      processAudioFile(file);
    }
  };

  // Sample selection
  const handleSelectSample = async (sampleName) => {
    setSelectedSample(sampleName);
    const sampleObj = samples?.audios?.find((a) => a.name === sampleName);
    if (sampleObj) {
      setAudioUrl(sampleObj.url);
    }
    setLoading(true);
    setError(null);
    try {
      const res = await predictAudio(sampleName);
      setResult(res);
      if (setMultimodalAudio) {
        setMultimodalAudio({ sample: sampleName, name: sampleName });
      }
      if (onAddHistory) {
        onAddHistory({
          modality: 'Audio (Sample)',
          emotion: res.emotion,
          confidence: res.confidence,
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
            onClick={() => setMode('upload')}
          >
            📁 Upload Audio (.wav)
          </button>
          <button
            className={`btn-mode ${mode === 'record' ? 'btn-active' : ''}`}
            onClick={() => setMode('record')}
          >
            🎙️ Record Microphone
          </button>
          <button
            className={`btn-mode ${mode === 'sample' ? 'btn-active' : ''}`}
            onClick={() => setMode('sample')}
          >
            🎵 Preloaded Samples
          </button>
        </div>

        {mode === 'upload' && (
          <div className="input-group">
            <input
              type="file"
              ref={fileInputRef}
              accept=".wav,.mp3,.ogg,.flac,audio/*"
              onChange={handleFileInputChange}
              style={{ display: 'none' }}
            />
            <button
              className="btn btn-primary"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
            >
              {loading ? 'Analyzing Acoustics...' : '📂 Choose Audio File'}
            </button>
            {selectedFile && <span className="file-name">{selectedFile.name}</span>}
          </div>
        )}

        {mode === 'sample' && (
          <div className="input-group">
            <label className="input-label">Select Sample Audio:</label>
            <select
              className="select-dropdown"
              value={selectedSample}
              onChange={(e) => handleSelectSample(e.target.value)}
            >
              <option value="">-- Choose Sample Audio --</option>
              {samples?.audios?.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
        )}

        {mode === 'record' && (
          <div className="input-group">
            {!isRecording ? (
              <button className="btn btn-success" onClick={startRecording} disabled={loading}>
                🔴 Start Recording
              </button>
            ) : (
              <button className="btn btn-danger" onClick={stopRecording}>
                ⏹ Stop & Analyze ({recordDuration}s)
              </button>
            )}
            {isRecording && <span className="badge-pulse">Recording Speech Signal...</span>}
          </div>
        )}
      </div>

      {error && <div className="alert-error">{error}</div>}

      {/* Main Split Grid */}
      <div className="grid-2col">
        {/* Left Column: Player & Signal Card */}
        <div className="card display-card">
          <div className="card-header-flex">
            <h3 className="card-heading">Speech Signal & Audio Playback</h3>
            {mode === 'upload' && (
              <span className="hint-pill">Drag & drop or click box to upload</span>
            )}
          </div>

          <div
            className={`audio-display-box ${isDragOver ? 'drag-over' : ''} ${
              mode === 'upload' && !audioUrl ? 'clickable-zone' : ''
            }`}
            onDragOver={mode === 'upload' ? handleDragOver : undefined}
            onDragLeave={mode === 'upload' ? handleDragLeave : undefined}
            onDrop={mode === 'upload' ? handleDrop : undefined}
            onClick={() => {
              if (mode === 'upload' && !loading && !audioUrl) {
                fileInputRef.current?.click();
              }
            }}
          >
            {/* High-Tech HUD Reticles */}
            <span className="hud-corner hud-tl" />
            <span className="hud-corner hud-tr" />
            <span className="hud-corner hud-bl" />
            <span className="hud-corner hud-br" />

            {mode === 'record' && !isRecording && !audioUrl && (
              <div className="placeholder-box">
                <span className="placeholder-icon">🎙️</span>
                <strong style={{ fontSize: '15px' }}>
                  Microphone is Currently Idle
                </strong>
                <p style={{ marginTop: '8px' }}>
                  Click below to begin speaking and analyze your voice emotion in real time.
                </p>
                <button
                  className="btn btn-success"
                  style={{ marginTop: '16px' }}
                  onClick={startRecording}
                >
                  🔴 Start Recording
                </button>
              </div>
            )}

            {mode === 'record' && isRecording && (
              <div className="recording-live-zone">
                <div className="badge-live-camera">
                  <span className="live-dot" /> RECORDING SPEECH
                </div>
                <div className="recording-timer-hero">{recordDuration}s</div>
                <div className="soundwave-container soundwave-recording">
                  {[...Array(28)].map((_, i) => (
                    <span
                      key={i}
                      className="soundwave-bar soundwave-active"
                      style={{
                        animationDelay: `${(i * 0.05).toFixed(2)}s`,
                      }}
                    />
                  ))}
                </div>
                <p className="recording-prompt">Capturing acoustic speech frequencies in real time...</p>
                <button className="btn btn-danger" onClick={stopRecording} style={{ marginTop: '16px' }}>
                  ⏹ Stop & Analyze Audio
                </button>
              </div>
            )}

            {audioUrl && (
              <div
                className="audio-player-wrapper"
                onClick={(e) => e.stopPropagation()}
              >
                {mode === 'upload' && (
                  <button
                    className="btn-change-image"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    🔄 Change Audio
                  </button>
                )}
                <div className="audio-icon-banner">🎙️</div>
                <h4 className="audio-title">
                  {selectedFile?.name || selectedSample || 'Microphone Recording'}
                </h4>

                {/* Animated Audio Equalizer Visualizer */}
                <div className="soundwave-container">
                  {[...Array(32)].map((_, i) => (
                    <span
                      key={i}
                      className="soundwave-bar"
                      style={{
                        animationDelay: `${(i * 0.04).toFixed(2)}s`,
                      }}
                    />
                  ))}
                </div>

                <audio controls src={audioUrl} className="audio-element" />
                {loading && (
                  <div className="loading-overlay" style={{ marginTop: '16px' }}>
                    <span className="spinner">⏳</span>
                    <p>Extracting 40 MFCCs & Classifying Acoustic Emotion...</p>
                  </div>
                )}
              </div>
            )}

            {!audioUrl && mode !== 'record' && (
              <div className="placeholder-box">
                <span className="placeholder-icon">🔊</span>
                <strong style={{ fontSize: '15px' }}>
                  Click to Browse or Drag & Drop Audio File (.wav)
                </strong>
                <p style={{ marginTop: '6px' }}>Supports WAV, MP3, FLAC, and OGG</p>
                {mode === 'upload' && (
                  <button
                    className="btn btn-primary"
                    style={{ marginTop: '14px' }}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    📂 Browse Audio Files
                  </button>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Emotion Metrics Card */}
        <div className="card metrics-card">
          <h3 className="card-heading">Acoustic Emotion Analysis</h3>

          {loading && !result ? (
            <div className="placeholder-box">
              <span className="placeholder-icon">⏳</span>
              <p>Extracting speech acoustic features...</p>
            </div>
          ) : result ? (
            <div className="metrics-content">
              <div className={`top-result-badge emotion-glow-${result.emotion.toLowerCase()}`}>
                <div className="result-headline">
                  <span className="emotion-title">{result.emotion}</span>
                  <span className="emotion-emoji-hero">
                    {result.emotion === 'Happy' ? '😄' :
                     result.emotion === 'Angry' ? '😠' :
                     result.emotion === 'Sad' ? '😢' :
                     result.emotion === 'Surprise' ? '😲' :
                     result.emotion === 'Fear' ? '😨' :
                     result.emotion === 'Disgust' ? '🤢' : '😐'}
                  </span>
                </div>
                <span className="confidence-pill">
                  {(result.confidence * 100).toFixed(1)}% Confidence
                </span>
              </div>

              <div className="telemetry-bar">
                <span className="telemetry-item"><strong>DURATION:</strong> {result.duration_seconds}s</span>
                <span className="telemetry-item"><strong>SAMPLING:</strong> 22,050 Hz</span>
                <span className="telemetry-item"><strong>ACOUSTIC TENSOR:</strong> 40 MFCCs</span>
                <span className="telemetry-item"><strong>MODEL:</strong> MLP Neural Net</span>
              </div>

              <EmotionBarChart
                probabilities={result.probabilities}
                topEmotion={result.emotion}
              />
            </div>
          ) : (
            <div className="placeholder-box">
              <span className="placeholder-icon">📊</span>
              <p>Select or record audio to view speech emotion metrics and probability distribution.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
