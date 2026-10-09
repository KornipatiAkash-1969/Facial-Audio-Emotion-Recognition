import React, { useState, useRef, useMemo } from 'react';
import { predictMultimodal, getApiBase } from '../services/api';
import EmotionBarChart from './EmotionBarChart';
import Loader from './Loader';

const EMOTION_EMOJIS = {
  Angry: '😠',
  Disgust: '🤢',
  Fear: '😨',
  Happy: '😄',
  Neutral: '😐',
  Sad: '😢',
  Surprise: '😲',
};

const PRESET_SCENARIOS = [
  {
    name: '😄 Joy Alignment',
    desc: 'Congruent: Happy Face + Happy Voice',
    face: 'happy_test.jpg',
    audio: 'happy_test.wav',
  },
  {
    name: '🎭 Sarcasm / Dissonance',
    desc: 'Conflict: Smiling Face + Furious Voice',
    face: 'happy_test.jpg',
    audio: 'angry_test.wav',
  },
  {
    name: '😠 Anger Alignment',
    desc: 'Congruent: Furrowed Brow + Harsh Voice',
    face: 'angry_test.jpg',
    audio: 'angry_test.wav',
  },
  {
    name: '😢 Sorrow Alignment',
    desc: 'Congruent: Downturned Lips + Somber Voice',
    face: 'sad_test.jpg',
    audio: 'sad_test.wav',
  },
  {
    name: '😲 Shock Alignment',
    desc: 'Congruent: Wide Eyes + High Pitch Shock',
    face: 'surprise_test.jpg',
    audio: 'surprise_test.wav',
  },
  {
    name: '🤢 Revulsion / Disgust',
    desc: 'Congruent: Wrinkled Nose + Disgusted Tone',
    face: 'disgust_test.jpg',
    audio: 'disgust_test.wav',
  },
];

export default function FusionTab({
  samples,
  multimodalFace,
  multimodalAudio,
  onAddHistory,
}) {
  const [faceWeight, setFaceWeight] = useState(0.5);
  // Default to benchmark test samples so fusion is immediately runnable
  const [selectedFaceSample, setSelectedFaceSample] = useState(() => multimodalFace ? '' : 'happy_test.jpg');
  const [selectedAudioSample, setSelectedAudioSample] = useState(() => multimodalAudio ? '' : 'happy_test.wav');
  const [uploadedFaceFile, setUploadedFaceFile] = useState(null);
  const [uploadedAudioFile, setUploadedAudioFile] = useState(null);
  const [sharedFaceOverride, setSharedFaceOverride] = useState(null);
  const [sharedAudioOverride, setSharedAudioOverride] = useState(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const faceFileInputRef = useRef(null);
  const audioFileInputRef = useRef(null);

  // Active inputs resolution with robust priority
  const activeFace =
    uploadedFaceFile?.name ||
    selectedFaceSample ||
    (sharedFaceOverride?.name ?? multimodalFace?.name) ||
    null;

  const activeAudio =
    uploadedAudioFile?.name ||
    selectedAudioSample ||
    (sharedAudioOverride?.name ?? multimodalAudio?.name) ||
    null;

  // Image preview source
  const facePreviewUrl = useMemo(() => {
    if (uploadedFaceFile) return URL.createObjectURL(uploadedFaceFile);
    const effShared = sharedFaceOverride ?? multimodalFace;
    if (effShared?.base64) return effShared.base64;
    if (effShared?.file) return URL.createObjectURL(effShared.file);
    const sampleName = selectedFaceSample || effShared?.sample;
    if (sampleName) return `${getApiBase()}/media/samples/faces/${sampleName}`;
    return null;
  }, [uploadedFaceFile, selectedFaceSample, sharedFaceOverride, multimodalFace]);

  // Audio preview source
  const audioPreviewUrl = useMemo(() => {
    if (uploadedAudioFile) return URL.createObjectURL(uploadedAudioFile);
    const effShared = sharedAudioOverride ?? multimodalAudio;
    if (effShared?.file) return URL.createObjectURL(effShared.file);
    const sampleName = selectedAudioSample || effShared?.sample;
    if (sampleName) return `${getApiBase()}/media/samples/audio/${sampleName}`;
    return null;
  }, [uploadedAudioFile, selectedAudioSample, sharedAudioOverride, multimodalAudio]);

  const handleApplyPreset = (preset) => {
    setUploadedFaceFile(null);
    setUploadedAudioFile(null);
    setSharedFaceOverride({ sample: preset.face, name: preset.face });
    setSharedAudioOverride({ sample: preset.audio, name: preset.audio });
    setSelectedFaceSample(preset.face);
    setSelectedAudioSample(preset.audio);
    setError(null);
  };

  const handleRunFusion = async () => {
    if (!activeFace) {
      setError('Please select or upload a face input first (via upload, sample dropdown, or quick scenario).');
      return;
    }
    if (!activeAudio) {
      setError('Please select or upload an audio input first (via upload, sample dropdown, or quick scenario).');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const payload = {
        faceWeight,
        audioWeight: 1.0 - faceWeight,
      };

      // 1. Resolve Face Payload
      if (uploadedFaceFile) {
        payload.faceFile = uploadedFaceFile;
      } else if (selectedFaceSample) {
        payload.faceSample = selectedFaceSample;
      } else {
        const effFace = sharedFaceOverride ?? multimodalFace;
        if (effFace?.base64) payload.faceBase64 = effFace.base64;
        else if (effFace?.file) payload.faceFile = effFace.file;
        else if (effFace?.sample) payload.faceSample = effFace.sample;
      }

      // 2. Resolve Audio Payload
      if (uploadedAudioFile) {
        payload.audioFile = uploadedAudioFile;
      } else if (selectedAudioSample) {
        payload.audioSample = selectedAudioSample;
      } else {
        const effAudio = sharedAudioOverride ?? multimodalAudio;
        if (effAudio?.file) payload.audioFile = effAudio.file;
        else if (effAudio?.sample) payload.audioSample = effAudio.sample;
      }

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
      setError(err.message || 'Multimodal fusion calculation failed. Ensure local server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tab-container">
      {/* Configuration Header Card */}
      <div className="card fusion-cfg-card">
        <h3 className="card-heading">Multimodal Input Feeds & Modality Weighting</h3>
        <p className="section-desc" style={{ marginBottom: '14px' }}>
          Simultaneously evaluate visual facial dynamics and acoustic speech prosody using late decision-level Bayesian fusion.
        </p>

        {/* 1-Click Preset Benchmark Scenarios */}
        <div style={{ marginBottom: '18px' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            ⚡ 1-Click Test Scenarios:
          </span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '8px' }}>
            {PRESET_SCENARIOS.map((p) => (
              <button
                key={p.name}
                type="button"
                className="btn btn-secondary"
                style={{ fontSize: '12px', padding: '6px 12px' }}
                title={p.desc}
                onClick={() => handleApplyPreset(p)}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        <div className="fusion-inputs-grid">
          {/* Face Input Status Card */}
          <div className="input-source-card">
            <span className="source-icon">👤</span>
            <div className="source-details" style={{ width: '100%' }}>
              <strong>Visual Facial Input:</strong>
              <p className="source-name" style={{ wordBreak: 'break-all' }}>
                {activeFace ? `Active: ${activeFace}` : 'No face selected'}
              </p>

              {facePreviewUrl && (
                <div style={{ margin: '8px 0', borderRadius: '6px', overflow: 'hidden', maxHeight: '110px', display: 'flex', alignItems: 'center' }}>
                  <img
                    src={facePreviewUrl}
                    alt="Selected Face Preview"
                    style={{ maxHeight: '110px', maxWidth: '100%', objectFit: 'contain', borderRadius: '6px', border: '1px solid var(--border-color)' }}
                  />
                </div>
              )}

              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '6px', flexWrap: 'wrap' }}>
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
                      setSharedFaceOverride(null);
                    }
                    e.target.value = '';
                  }}
                />
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ fontSize: '11px', padding: '5px 10px' }}
                  onClick={() => faceFileInputRef.current?.click()}
                >
                  📁 Upload Photo
                </button>
                <select
                  className="select-dropdown-small"
                  style={{ marginTop: 0, flex: 1, minWidth: '130px' }}
                  value={selectedFaceSample}
                  onChange={(e) => {
                    setSelectedFaceSample(e.target.value);
                    setUploadedFaceFile(null);
                    setSharedFaceOverride(null);
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
            <div className="source-details" style={{ width: '100%' }}>
              <strong>Acoustic Speech Input:</strong>
              <p className="source-name" style={{ wordBreak: 'break-all' }}>
                {activeAudio ? `Active: ${activeAudio}` : 'No audio selected'}
              </p>

              {audioPreviewUrl && (
                <div style={{ margin: '8px 0' }}>
                  <audio controls src={audioPreviewUrl} style={{ width: '100%', height: '36px' }} />
                </div>
              )}

              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '6px', flexWrap: 'wrap' }}>
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
                      setSharedAudioOverride(null);
                    }
                    e.target.value = '';
                  }}
                />
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ fontSize: '11px', padding: '5px 10px' }}
                  onClick={() => audioFileInputRef.current?.click()}
                >
                  📁 Upload Audio
                </button>
                <select
                  className="select-dropdown-small"
                  style={{ marginTop: 0, flex: 1, minWidth: '130px' }}
                  value={selectedAudioSample}
                  onChange={(e) => {
                    setSelectedAudioSample(e.target.value);
                    setUploadedAudioFile(null);
                    setSharedAudioOverride(null);
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
        <div className="slider-action-row" style={{ marginTop: '20px' }}>
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
            type="button"
            className="btn btn-primary btn-large"
            onClick={handleRunFusion}
            disabled={loading}
          >
            {loading ? 'Executing Fusion...' : '⚡ Analyze Multimodal Fusion'}
          </button>
        </div>
      </div>

      {error && <div className="alert-error">{error}</div>}

      {/* Loading Progress State */}
      {loading && (
        <div className="card fusion-loading-banner" style={{ textAlign: 'center', padding: '36px 20px' }}>
          <Loader size="large" text="Executing Decision-Level Late Fusion: Synchronizing Facial Tensors with Acoustic MFCCs..." />
        </div>
      )}

      {/* Main Results Dashboard */}
      {result && (
        <div className={`card fusion-results-card emotion-glow-${result.fused_emotion.toLowerCase()}`}>
          <div className="fusion-header-banner">
            <div>
              <span className="subtitle">Integrated Emotion Decision</span>
              <div className="result-headline" style={{ marginTop: '4px' }}>
                <h2 className="fused-emotion-title">{result.fused_emotion}</h2>
                <span className="emotion-emoji-hero">
                  {EMOTION_EMOJIS[result.fused_emotion] || '✨'}
                </span>
              </div>
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

          {/* Modality Breakdown Cards with Visual Face Box and Audio Player */}
          <div className="modality-breakdown-grid">
            <div className="breakdown-card visual-card">
              <span className="hud-corner hud-tl" />
              <span className="hud-corner hud-tr" />
              <div className="breakdown-header-flex">
                <span className="breakdown-title">Visual Facial Cue</span>
                <span className="breakdown-emoji">{EMOTION_EMOJIS[result.face.emotion] || '👤'}</span>
              </div>
              <h3>{result.face.emotion}</h3>

              {result.annotated_face_image && (
                <div style={{ margin: '10px 0', borderRadius: '8px', overflow: 'hidden', textAlign: 'center' }}>
                  <img
                    src={result.annotated_face_image}
                    alt="Annotated Detected Face"
                    style={{ maxWidth: '100%', maxHeight: '180px', objectFit: 'contain', borderRadius: '6px' }}
                  />
                </div>
              )}

              <div className="breakdown-metric-row">
                <span>Confidence</span>
                <strong>{(result.face.confidence * 100).toFixed(1)}%</strong>
              </div>
              <div className="breakdown-progress-track">
                <div
                  className="breakdown-progress-fill visual-fill"
                  style={{ width: `${Math.max(result.face.confidence * 100, 4)}%` }}
                />
              </div>
              <span className="weight-tag">Decision Weight: {(result.face.weight * 100).toFixed(0)}%</span>
            </div>

            <div className="breakdown-card acoustic-card">
              <span className="hud-corner hud-tl" />
              <span className="hud-corner hud-tr" />
              <div className="breakdown-header-flex">
                <span className="breakdown-title">Acoustic Speech Cue</span>
                <span className="breakdown-emoji">{EMOTION_EMOJIS[result.audio.emotion] || '🎙️'}</span>
              </div>
              <h3>{result.audio.emotion}</h3>

              {audioPreviewUrl && (
                <div style={{ margin: '10px 0' }}>
                  <audio controls src={audioPreviewUrl} style={{ width: '100%', height: '36px' }} />
                </div>
              )}

              <div className="breakdown-metric-row">
                <span>Confidence</span>
                <strong>{(result.audio.confidence * 100).toFixed(1)}%</strong>
              </div>
              <div className="breakdown-progress-track">
                <div
                  className="breakdown-progress-fill acoustic-fill"
                  style={{ width: `${Math.max(result.audio.confidence * 100, 4)}%` }}
                />
              </div>
              <span className="weight-tag">Decision Weight: {(result.audio.weight * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Combined Distribution & Psychological Insight */}
          <div className="grid-2col" style={{ marginTop: '24px' }}>
            <div className="card-sub">
              <EmotionBarChart
                probabilities={result.combined_probabilities}
                topEmotion={result.fused_emotion}
              />
            </div>

            <div className="card-sub insight-box">
              <h4 className="chart-title">Cross-Modal Affective Interpretation</h4>
              <p className="insight-text">{result.insight}</p>

              {/* High-Tech Circular Congruency Gauge */}
              <div className="alignment-gauge-container">
                <div className="gauge-circle-wrap">
                  <svg className="gauge-svg" viewBox="0 0 100 100">
                    <circle className="gauge-bg" cx="50" cy="50" r="40" />
                    <circle
                      className="gauge-progress"
                      cx="50"
                      cy="50"
                      r="40"
                      strokeDasharray="251.2"
                      strokeDashoffset={
                        251.2 * (1 - Math.min(Math.max(result.congruency_score, 0), 1))
                      }
                    />
                  </svg>
                  <div className="gauge-center-text">
                    <span className="gauge-value">
                      {(result.congruency_score * 100).toFixed(0)}%
                    </span>
                    <span className="gauge-unit">ALIGN</span>
                  </div>
                </div>

                <div className="gauge-details">
                  <span className="gauge-title">Cosine Modality Alignment</span>
                  <p className="gauge-desc">
                    {result.is_congruent
                      ? 'High cosine vector similarity between facial expression tensor and acoustic spectral features.'
                      : 'Affective dissonance detected: visual and acoustic channels signal opposing emotional states.'}
                  </p>
                  <div className="telemetry-bar" style={{ marginTop: '10px' }}>
                    <span className="telemetry-item"><strong>SCORE:</strong> {result.congruency_score}</span>
                    <span className="telemetry-item"><strong>STATE:</strong> {result.is_congruent ? 'SYNERGISTIC' : 'DISSONANT'}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
