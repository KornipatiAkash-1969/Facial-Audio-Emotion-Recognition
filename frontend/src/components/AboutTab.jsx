import React from 'react';

export default function AboutTab({ status }) {
  return (
    <div className="tab-container">
      <div className="card">
        <h3 className="card-heading">System Architecture & Machine Learning Specifications</h3>

        <div className="about-grid">
          <div className="about-section">
            <h4>📸 Visual Modality (Face)</h4>
            <ul>
              <li><strong>Face Detector:</strong> OpenCV Haar Cascade (Frontal Face Default)</li>
              <li><strong>Classifier:</strong> 4-Block Deep Convolutional Neural Network (CNN)</li>
              <li><strong>Input Tensor:</strong> 48×48 normalized single-channel grayscale</li>
              <li><strong>Parameter Count:</strong> ~2.49 Million parameters</li>
              <li><strong>Classes:</strong> 7 Universal Facial Emotions (FER-2013)</li>
            </ul>
          </div>

          <div className="about-section">
            <h4>🎙️ Acoustic Modality (Speech)</h4>
            <ul>
              <li><strong>Feature Extractor:</strong> 40 Mel-Frequency Cepstral Coefficients (MFCCs)</li>
              <li><strong>Classifier:</strong> Multi-Layer Perceptron (MLP) with Standard Scaler</li>
              <li><strong>Test Accuracy:</strong> 99.82% evaluated on TESS dataset</li>
              <li><strong>Sample Rate:</strong> 22,050 Hz mono signal</li>
              <li><strong>Classes:</strong> 7 Universal Speech Emotions (TESS)</li>
            </ul>
          </div>

          <div className="about-section">
            <h4>🧬 Multimodal Fusion Engine</h4>
            <ul>
              <li><strong>Integration Method:</strong> Decision-Level Weighted Probability Late Fusion</li>
              <li><strong>Formula:</strong> <code>P_fused = w_f * P_face + w_a * P_audio</code></li>
              <li><strong>Congruency Metric:</strong> High-dimensional Cosine Similarity between visual and acoustic distributions</li>
              <li><strong>Psychological Diagnostics:</strong> Detection of affective conflict, social masking, and sarcasm</li>
            </ul>
          </div>

          <div className="about-section">
            <h4>⚡ Technology Stack</h4>
            <ul>
              <li><strong>Frontend:</strong> React.js, Vite, HTML5 Canvas, Web Audio API</li>
              <li><strong>Backend API:</strong> Python FastAPI, Uvicorn (ASGI)</li>
              <li><strong>Machine Learning:</strong> TensorFlow, Keras, Librosa, Scikit-Learn, OpenCV</li>
              <li><strong>Audio Processing:</strong> SoundFile, NumPy, SciPy</li>
            </ul>
          </div>
          <div className="about-section">
            <h4>☁️ Cloud Deployment Architecture</h4>
            <ul>
              <li><strong>Frontend (Vercel):</strong> React SPA hosted globally at edge with automatic caching and zero-latency static media delivery.</li>
              <li><strong>Backend (Render):</strong> FastAPI REST engine deployed as a Python Web Service or Docker container on Render.</li>
              <li><strong>Cloud Linking:</strong> Set <code>VITE_API_URL</code> on Vercel or click <strong>Connect Backend</strong> in the top navbar to paste your Render URL.</li>
              <li><strong>Cross-Origin:</strong> Pre-configured FastAPI CORS middleware permits secure requests from any Vercel domain.</li>
            </ul>
          </div>
        </div>

        <div className="card" style={{ marginTop: '20px', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
          <h4 style={{ color: 'var(--accent-cyan)', marginBottom: '10px' }}>🚀 Deploying Backend on Render (Step-by-Step)</h4>
          <ol style={{ paddingLeft: '20px', fontSize: '13px', lineHeight: '1.7', color: 'var(--text-secondary)' }}>
            <li>Push this repository to your GitHub account (<code>git push origin master</code>).</li>
            <li>Go to <a href="https://render.com" target="_blank" rel="noreferrer" style={{ color: 'var(--accent-blue)' }}>render.com</a> and click <strong>New → Web Service</strong>.</li>
            <li>Connect your GitHub repository.</li>
            <li>Set <strong>Runtime:</strong> <code>Python</code> (or <code>Docker</code>).</li>
            <li>Set <strong>Build Command:</strong> <code>pip install -r requirements.txt</code></li>
            <li>Set <strong>Start Command:</strong> <code>uvicorn server:app --host 0.0.0.0 --port $PORT</code></li>
            <li>Click <strong>Deploy Web Service</strong>.</li>
            <li>Once active, copy your Render URL (e.g., <code>https://your-service.onrender.com</code>) and paste it into the <strong>Connect Backend</strong> modal on Vercel!</li>
          </ol>
        </div>

        <div className="credits-card">
          <h4>Dataset Attribution & Research Credits</h4>
          <p>
            • <strong>Toronto Emotional Speech Set (TESS):</strong> Created by Kate Dupuis and M. Kathleen Pichora-Fuller at the University of Toronto Psychology Department.<br />
            • <strong>Facial Expression Recognition (FER-2013):</strong> Curated by Pierre-Luc Carrier and Aaron Courville for the ICML 2013 representation learning challenge.
          </p>
        </div>
      </div>
    </div>
  );
}
