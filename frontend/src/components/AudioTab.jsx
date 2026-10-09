import React, { useState, useRef } from 'react';
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

  // Microphone recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const recorderRef = useRef(null);
  const timerRef = useRef(null);

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

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setAudioUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePredictFile = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictAudio(selectedFile);
      setResult(res);
      if (setMultimodalAudio) {
        setMultimodalAudio({ file: selectedFile, name: selectedFile.name });
      }
      if (onAddHistory) {
        onAddHistory({
          modality: 'Audio (File)',
          emotion: res.emotion,
          confidence: res.confidence,
          source: selectedFile.name,
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

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
              id="audio-file-input"
              accept=".wav,.mp3,.ogg,.flac"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <label htmlFor="audio-file-input" className="btn btn-secondary">
              Choose Audio File
            </label>
            {selectedFile && <span className="file-name">{selectedFile.name}</span>}
            <button
              className="btn btn-primary"
              onClick={handlePredictFile}
              disabled={!selectedFile || loading}
            >
              {loading ? 'Analyzing Acoustics...' : 'Predict Speech Emotion'}
            </button>
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
          <h3 className="card-heading">Speech Signal & Audio Playback</h3>
          <div className="audio-display-box">
            {audioUrl ? (
              <div className="audio-player-wrapper">
                <div className="audio-icon-banner">🎙️</div>
                <h4 className="audio-title">
                  {selectedFile?.name || selectedSample || 'Microphone Recording'}
                </h4>
                <audio controls src={audioUrl} className="audio-element" />
              </div>
            ) : (
              <div className="placeholder-box">
                <span className="placeholder-icon">🔊</span>
                <p>Upload a .wav file, choose a sample, or record your voice to inspect acoustic emotion.</p>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Emotion Metrics Card */}
        <div className="card metrics-card">
          <h3 className="card-heading">Acoustic Emotion Analysis</h3>

          {result ? (
            <div className="metrics-content">
              <div className="top-result-badge">
                <span className="emotion-title">{result.emotion}</span>
                <span className="confidence-pill">
                  {(result.confidence * 100).toFixed(1)}% Confidence
                </span>
              </div>

              <div className="info-chip">
                <span>Duration: {result.duration_seconds} seconds</span>
                <span style={{ marginLeft: '12px' }}>Features: 40 MFCCs</span>
              </div>

              <EmotionBarChart
                probabilities={result.probabilities}
                topEmotion={result.emotion}
              />
            </div>
          ) : (
            <div className="placeholder-box">
              <span className="placeholder-icon">📊</span>
              <p>Awaiting speech audio input to generate acoustic emotion metrics.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
