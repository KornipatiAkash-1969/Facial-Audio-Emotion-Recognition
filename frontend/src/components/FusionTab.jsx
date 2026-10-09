import React, { useState, useRef } from 'react';
import { predictMultimodal } from '../services/api';
import EmotionBarChart from './EmotionBarChart';

export default function FusionTab({
  samples,
  multimodalFace,
  multimodalAudio,
  onAddHistory,
}) {
  const [faceWeight, setFaceWeight] = useState(0.5);
  const [selectedFaceSample, setSelectedFaceSample] = useState('');
  const [selectedAudioSample, setSelectedAudioSample] = useState('');
  const [uploadedFaceFile, setUploadedFaceFile] = useState(null);
  const [uploadedAudioFile, setUploadedAudioFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const faceFileInputRef = useRef(null);
  const audioFileInputRef = useRef(null);

  const activeFace =
    uploadedFaceFile?.name ||
    multimodalFace?.name ||
    (selectedFaceSample ? selectedFaceSample : null);
  const activeAudio =
    uploadedAudioFile?.name ||
    multimodalAudio?.name ||
    (selectedAudioSample ? selectedAudioSample : null);

  const handleRunFusion = async () => {
    if (!activeFace) {
      setError('Please select or upload a face input first (via Face Tab, upload button, or sample dropdown).');
      return;
    }
    if (!activeAudio) {
      setError('Please select or upload a speech audio input first (via Audio Tab, upload button, or sample dropdown).');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const payload = {
        faceWeight,
        audioWeight: 1.0 - faceWeight,
      };

      if (uploadedFaceFile) payload.faceFile = uploadedFaceFile;
      else if (multimodalFace?.base64) payload.faceBase64 = multimodalFace.base64;
      else if (multimodalFace?.file) payload.faceFile = multimodalFace.file;
      else if (selectedFaceSample) payload.faceSample = selectedFaceSample;
      else if (multimodalFace?.sample) payload.faceSample = multimodalFace.sample;

      if (uploadedAudioFile) payload.audioFile = uploadedAudioFile;
      else if (multimodalAudio?.file) payload.audioFile = multimodalAudio.file;
      else if (selectedAudioSample) payload.audioSample = selectedAudioSample;
      else if (multimodalAudio?.sample) payload.audioSample = multimodalAudio.sample;

      const res = await predictMultimodal(payload);
      setResult(res);

      if (onAddHistory) {
        onAddHistory({
          modality: 'Multimodal',
          emotion: res.fused_emotion,
          confidence: res.fused_confidence,
          source: `Face: ${res.face.emotion} | Audio: ${res.audio.emotion}`,
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
      {/* Configuration Header Card */}
      <div className="card fusion-cfg-card">
        <h3 className="card-heading">Multimodal Input Feeds & Modality Weighting</h3>

        <div className="fusion-inputs-grid">
          {/* Face Input Status Card */}
          <div className="input-source-card">
            <span className="source-icon">👤</span>
            <div className="source-details">
              <strong>Visual Facial Input:</strong>
              <p className="source-name">
                {activeFace ? `Active: ${activeFace}` : 'No face selected'}
              </p>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '6px' }}>
                <input
                  type="file"
                  ref={faceFileInputRef}
                  accept="image/*,.jpg,.jpeg,.png,.webp,.bmp"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) {
                      setUploadedFaceFile(f);
                      setSelectedFaceSample('');
                    }
                    e.target.value = '';
                  }}
                />
                <button
                  className="btn btn-secondary"
                  style={{ fontSize: '11px', padding: '5px 10px' }}
                  onClick={() => faceFileInputRef.current?.click()}
                >
                  📁 Upload Photo
                </button>
                <select
                  className="select-dropdown-small"
                  style={{ marginTop: 0 }}
                  value={selectedFaceSample}
                  onChange={(e) => {
                    setSelectedFaceSample(e.target.value);
                    setUploadedFaceFile(null);
                  }}
                >
                  <option value="">-- Or Sample Face --</option>
                  {samples?.faces?.map((f) => (
                    <option key={f.name} value={f.name}>
                      {f.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Audio Input Status Card */}
          <div className="input-source-card">
            <span className="source-icon">🎙️</span>
            <div className="source-details">
              <strong>Acoustic Speech Input:</strong>
              <p className="source-name">
                {activeAudio ? `Active: ${activeAudio}` : 'No audio selected'}
              </p>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '6px' }}>
                <input
                  type="file"
                  ref={audioFileInputRef}
                  accept=".wav,.mp3,.ogg,.flac,audio/*"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) {
                      setUploadedAudioFile(f);
                      setSelectedAudioSample('');
                    }
                    e.target.value = '';
                  }}
                />
                <button
                  className="btn btn-secondary"
                  style={{ fontSize: '11px', padding: '5px 10px' }}
                  onClick={() => audioFileInputRef.current?.click()}
                >
                  📁 Upload Audio
                </button>
                <select
                  className="select-dropdown-small"
                  style={{ marginTop: 0 }}
                  value={selectedAudioSample}
                  onChange={(e) => {
                    setSelectedAudioSample(e.target.value);
                    setUploadedAudioFile(null);
                  }}
                >
                  <option value="">-- Or Sample Audio --</option>
                  {samples?.audios?.map((a) => (
                    <option key={a.name} value={a.name}>
                      {a.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Weighting Slider & Trigger */}
        <div className="slider-action-row">
          <div className="slider-container">
            <div className="slider-label-row">
              <span>Face Weight: {(faceWeight * 100).toFixed(0)}%</span>
              <span>Voice Weight: {((1.0 - faceWeight) * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={faceWeight}
              onChange={(e) => setFaceWeight(parseFloat(e.target.value))}
              className="range-slider"
            />
          </div>

          <button
            className="btn btn-primary btn-large"
            onClick={handleRunFusion}
            disabled={loading}
          >
            {loading ? 'Executing Fusion...' : '⚡ Analyze Multimodal Fusion'}
          </button>
        </div>
      </div>

      {error && <div className="alert-error">{error}</div>}

      {/* Main Results Dashboard */}
      {result && (
        <div className="card fusion-results-card">
          <div className="fusion-header-banner">
            <div>
              <span className="subtitle">Integrated Emotion Decision</span>
              <h2 className="fused-emotion-title">{result.fused_emotion}</h2>
              <span className="fused-confidence-tag">
                {(result.fused_confidence * 100).toFixed(1)}% Fused Confidence
              </span>
            </div>

            <div className="congruency-indicator">
              {result.is_congruent ? (
                <div className="badge-congruent">
                  <span className="badge-icon">✓</span>
                  <div>
                    <strong>CONGRUENT STATE</strong>
                    <p>Synergistic cross-modal emotional alignment</p>
                  </div>
                </div>
              ) : (
                <div className="badge-incongruent">
                  <span className="badge-icon">⚠</span>
                  <div>
                    <strong>AFFECTIVE DISSONANCE</strong>
                    <p>Cross-modal emotional conflict detected</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Modality Breakdown Cards */}
          <div className="modality-breakdown-grid">
            <div className="breakdown-card visual-card">
              <span className="breakdown-title">Visual Facial Cue</span>
              <h3>{result.face.emotion}</h3>
              <p>Confidence: {(result.face.confidence * 100).toFixed(1)}%</p>
              <span className="weight-tag">Weight Applied: {(result.face.weight * 100).toFixed(0)}%</span>
            </div>

            <div className="breakdown-card acoustic-card">
              <span className="breakdown-title">Acoustic Speech Cue</span>
              <h3>{result.audio.emotion}</h3>
              <p>Confidence: {(result.audio.confidence * 100).toFixed(1)}%</p>
              <span className="weight-tag">Weight Applied: {(result.audio.weight * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Combined Distribution & Psychological Insight */}
          <div className="grid-2col" style={{ marginTop: '20px' }}>
            <div className="card-sub">
              <EmotionBarChart
                probabilities={result.combined_probabilities}
                topEmotion={result.fused_emotion}
              />
            </div>

            <div className="card-sub insight-box">
              <h4 className="chart-title">Affective Computing & Psychological Interpretation</h4>
              <p className="insight-text">{result.insight}</p>
              <div className="similarity-meta">
                <span>Cross-Modal Cosine Alignment: <strong>{result.congruency_score}</strong></span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
