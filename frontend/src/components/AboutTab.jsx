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
