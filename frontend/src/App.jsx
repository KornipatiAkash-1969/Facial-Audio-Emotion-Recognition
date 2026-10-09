import React, { useState, useEffect } from 'react';
import { fetchSystemStatus, fetchSampleFiles } from './services/api';
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

  useEffect(() => {
    async function init() {
      try {
        const s = await fetchSystemStatus();
        setStatus(s);
        setBackendOnline(true);
      } catch (err) {
        setBackendOnline(false);
      }

      try {
        const smp = await fetchSampleFiles();
        setSamples(smp);
      } catch (err) {
        console.error('Failed to load samples:', err);
      } finally {
        setInitialLoading(false);
      }
    }

    init();
    const interval = setInterval(init, 8000);
    return () => clearInterval(interval);
  }, []);

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

          <div className="nav-status">
            <span className={`status-dot ${backendOnline ? 'dot-online' : 'dot-offline'}`} />
            <span className="status-text">
              {backendOnline ? 'REST API Online' : 'Connecting...'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content View */}
      <main className="main-content">
        {!backendOnline && (
          <div className="alert-warning">
            ⚠️ <strong>Backend API server not detected on port 8000.</strong> Run{' '}
            <code>python server.py</code> in the project directory to connect the AI models.
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
