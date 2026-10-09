import React, { useState, useEffect, useCallback } from 'react';
import {
  fetchSystemStatus,
  fetchSampleFiles,
  getApiBase,
  setCustomApiUrl,
  getCustomApiUrl,
} from './services/api';
import FaceTab from './components/FaceTab';
import AudioTab from './components/AudioTab';
import FusionTab from './components/FusionTab';
import HistoryTab from './components/HistoryTab';
import AboutTab from './components/AboutTab';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('face'); // 'face' | 'audio' | 'fusion' | 'history' | 'about'
  const [status, setStatus] = useState(null);
  const [samples, setSamples] = useState({ faces: [], audios: [] });
  const [backendOnline, setBackendOnline] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('app-theme') || 'dark');
  const [initialLoading, setInitialLoading] = useState(true);

  // Backend connection state
  const [customApiUrl, setCustomApiUrlState] = useState(() => getCustomApiUrl() || getApiBase());
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectMsg, setConnectMsg] = useState(null);

  // Shared inputs for Multimodal tab
  const [multimodalFace, setMultimodalFace] = useState(null);
  const [multimodalAudio, setMultimodalAudio] = useState(null);

  // Session history
  const [history, setHistory] = useState([]);

  // Sync theme attribute on document root
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('app-theme', theme);
  }, [theme]);

  const checkConnection = useCallback(async () => {
    try {
      const s = await fetchSystemStatus();
      setStatus(s);
      setBackendOnline(true);
      const smp = await fetchSampleFiles();
      setSamples(smp);
      return true;
    } catch (err) {
      setBackendOnline(false);
      return false;
    }
  }, []);

  useEffect(() => {
    async function init() {
      await checkConnection();
      setInitialLoading(false);
    }
    init();
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, [checkConnection]);

  const handleConnectUrl = async (urlToTest) => {
    setIsConnecting(true);
    setConnectMsg(null);
    const targetUrl = (urlToTest || '').trim();
    setCustomApiUrl(targetUrl);
    setCustomApiUrlState(targetUrl);

    try {
      const s = await fetchSystemStatus();
      setStatus(s);
      setBackendOnline(true);
      const smp = await fetchSampleFiles();
      setSamples(smp);
      setConnectMsg({
        type: 'success',
        text: `✅ Connected to backend at ${getApiBase()}!`,
      });
      setTimeout(() => {
        setShowConfigModal(false);
      }, 1500);
    } catch (err) {
      setBackendOnline(false);
      setConnectMsg({
        type: 'error',
        text: `❌ Could not connect to ${targetUrl || 'backend'}. Please verify your Render URL.`,
      });
    } finally {
      setIsConnecting(false);
    }
  };

  const addHistoryItem = (item) => {
    const timestamp = new Date().toLocaleTimeString();
    setHistory((prev) => [{ ...item, timestamp }, ...prev]);
  };

  const clearHistory = () => {
    setHistory([]);
  };

  if (initialLoading) {
    return (
      <div className="app-splash-loader">
        <div className="splash-content">
          <div className="splash-logo-wrap">
            <span className="splash-logo">🎭</span>
            <div className="splash-spinner-ring" />
          </div>
          <h2 className="splash-title">AffectSense AI</h2>
          <p className="splash-subtitle">Initializing Neural Emotion Recognition Architecture...</p>
          <div className="splash-progress-track">
            <div className="splash-progress-bar" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-layout">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="nav-brand">
          <span className="brand-logo">🎭</span>
          <div>
            <h1 className="brand-title">AffectSense AI</h1>
            <span className="brand-subtitle">
              Facial & Audio Emotion Recognition • Multimodal Fusion
            </span>
          </div>
        </div>

        {/* Tab Navigation Navigation Pills */}
        <nav className="nav-tabs">
          <button
            className={`tab-btn ${activeTab === 'face' ? 'tab-btn-active' : ''}`}
            onClick={() => setActiveTab('face')}
          >
            👤 Facial Emotion
          </button>
          <button
            className={`tab-btn ${activeTab === 'audio' ? 'tab-btn-active' : ''}`}
            onClick={() => setActiveTab('audio')}
          >
            🎙️ Audio Emotion
          </button>
          <button
            className={`tab-btn ${activeTab === 'fusion' ? 'tab-btn-active' : ''}`}
            onClick={() => setActiveTab('fusion')}
          >
            🧬 Multimodal Fusion
          </button>
          <button
            className={`tab-btn ${activeTab === 'history' ? 'tab-btn-active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            📜 History {history.length > 0 && <span className="badge-count">{history.length}</span>}
          </button>
          <button
            className={`tab-btn ${activeTab === 'about' ? 'tab-btn-active' : ''}`}
            onClick={() => setActiveTab('about')}
          >
            ⚙️ About
          </button>
        </nav>

        {/* Navigation Controls: Theme Switcher & Status */}
        <div className="nav-controls">
          <div className="theme-toggle-group">
            <button
              className={`btn-theme ${theme === 'dark' ? 'theme-active' : ''}`}
              onClick={() => setTheme('dark')}
              title="Obsidian Dark Theme"
            >
              🌙 Dark
            </button>
            <button
              className={`btn-theme ${theme === 'light' ? 'theme-active' : ''}`}
              onClick={() => setTheme('light')}
              title="Modern Clean Light Theme"
            >
              ☀️ Light
            </button>
            <button
              className={`btn-theme ${theme === 'cyber' ? 'theme-active' : ''}`}
              onClick={() => setTheme('cyber')}
              title="Cyberpunk Neon Theme"
            >
              ⚡ Cyber
            </button>
          </div>

          <button
            className={`nav-status-btn ${backendOnline ? 'status-online' : 'status-offline'}`}
            onClick={() => setShowConfigModal(true)}
            title="Click to configure Render or Localhost Backend URL"
          >
            <span className={`status-dot ${backendOnline ? 'dot-online' : 'dot-offline'}`} />
            <span className="status-text">
              {backendOnline ? 'Backend Online' : 'Connect Backend'}
            </span>
            <span className="gear-icon">⚙️</span>
          </button>
        </div>
      </header>

      {/* Main Content View */}
      <main className="main-content">
        {!backendOnline && (
          <div className="backend-connect-card">
            <div className="backend-connect-header">
              <span className="backend-connect-icon">☁️</span>
              <div>
                <h3 className="backend-connect-title">Connect Render Backend Service</h3>
                <p className="backend-connect-desc">
                  This frontend is running live on <strong>Vercel</strong>. To connect your cloud or local ML engine, enter your <strong>Render backend URL</strong> below (or connect to localhost).
                </p>
              </div>
            </div>

            <div className="backend-connect-form">
              <div className="input-group-flex">
                <input
                  type="url"
                  className="backend-url-input"
                  placeholder="https://your-service.onrender.com"
                  value={customApiUrl}
                  onChange={(e) => setCustomApiUrlState(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleConnectUrl(customApiUrl)}
                />
                <button
                  className="btn btn-primary"
                  onClick={() => handleConnectUrl(customApiUrl)}
                  disabled={isConnecting}
                >
                  {isConnecting ? 'Connecting...' : '🔗 Connect Render'}
                </button>
                <button
                  className="btn btn-outline"
                  onClick={() => handleConnectUrl('http://localhost:8000')}
                  disabled={isConnecting}
                >
                  💻 Localhost:8000
                </button>
              </div>

              {connectMsg && (
                <div className={`connect-status-msg ${connectMsg.type}`}>
                  {connectMsg.text}
                </div>
              )}

              <div className="backend-connect-tips">
                <span>
                  💡 <strong>Render Setup:</strong> Create a Web Service on <a href="https://render.com" target="_blank" rel="noreferrer">render.com</a> from your repo with start command <code>uvicorn server:app --host 0.0.0.0 --port $PORT</code>. Once active, paste your <code>https://...onrender.com</code> URL above!
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Backend Configuration Modal */}
        {showConfigModal && (
          <div className="modal-backdrop" onClick={() => setShowConfigModal(false)}>
            <div className="modal-card" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>⚙️ Backend API Configuration</h3>
                <button className="btn-close-modal" onClick={() => setShowConfigModal(false)}>✕</button>
              </div>
              <div className="modal-body">
                <p className="modal-desc">
                  Configure the FastAPI endpoint that powers facial and audio emotion recognition:
                </p>
                <div className="input-group-vertical">
                  <label className="input-label">Backend Service URL:</label>
                  <input
                    type="url"
                    className="backend-url-input"
                    placeholder="https://your-service.onrender.com or http://localhost:8000"
                    value={customApiUrl}
                    onChange={(e) => setCustomApiUrlState(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleConnectUrl(customApiUrl)}
                  />
                </div>
                {connectMsg && (
                  <div className={`connect-status-msg ${connectMsg.type}`}>
                    {connectMsg.text}
                  </div>
                )}
                <div className="modal-actions">
                  <button
                    className="btn btn-primary"
                    onClick={() => handleConnectUrl(customApiUrl)}
                    disabled={isConnecting}
                  >
                    {isConnecting ? 'Testing...' : '🔗 Save & Connect'}
                  </button>
                  <button
                    className="btn btn-outline"
                    onClick={() => handleConnectUrl('http://localhost:8000')}
                    disabled={isConnecting}
                  >
                    💻 Use Localhost:8000
                  </button>
                  <button
                    className="btn btn-outline"
                    onClick={() => {
                      setCustomApiUrl('');
                      setCustomApiUrlState('');
                      handleConnectUrl('');
                    }}
                  >
                    🔄 Reset Default
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'face' && (
          <FaceTab
            samples={samples}
            onAddHistory={addHistoryItem}
            setMultimodalFace={setMultimodalFace}
          />
        )}

        {activeTab === 'audio' && (
          <AudioTab
            samples={samples}
            onAddHistory={addHistoryItem}
            setMultimodalAudio={setMultimodalAudio}
          />
        )}

        {activeTab === 'fusion' && (
          <FusionTab
            samples={samples}
            multimodalFace={multimodalFace}
            multimodalAudio={multimodalAudio}
            onAddHistory={addHistoryItem}
          />
        )}

        {activeTab === 'history' && (
          <HistoryTab history={history} onClearHistory={clearHistory} />
        )}

        {activeTab === 'about' && <AboutTab status={status} />}
      </main>
    </div>
  );
}
