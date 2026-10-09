import React from 'react';

const EMOTION_GRADIENTS = {
  Happy: 'linear-gradient(90deg, #F59E0B 0%, #FBBF24 50%, #FDE68A 100%)',
  Angry: 'linear-gradient(90deg, #DC2626 0%, #EF4444 50%, #F87171 100%)',
  Sad: 'linear-gradient(90deg, #2563EB 0%, #3B82F6 50%, #60A5FA 100%)',
  Surprise: 'linear-gradient(90deg, #D97706 0%, #F59E0B 50%, #FCD34D 100%)',
  Fear: 'linear-gradient(90deg, #7C3AED 0%, #8B5CF6 50%, #A78BFA 100%)',
  Disgust: 'linear-gradient(90deg, #059669 0%, #10B981 50%, #34D399 100%)',
  Neutral: 'linear-gradient(90deg, #475569 0%, #64748B 50%, #94A3B8 100%)',
};

const EMOTION_EMOJIS = {
  Angry: '😠',
  Disgust: '🤢',
  Fear: '😨',
  Happy: '😄',
  Neutral: '😐',
  Sad: '😢',
  Surprise: '😲',
};

export default function EmotionBarChart({ probabilities, topEmotion }) {
  if (!probabilities) return null;

  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="emotion-chart">
      <div className="chart-header-flex">
        <h4 className="chart-title">Probability Spectrum</h4>
        <span className="spectrum-badge">7-Class Softmax</span>
      </div>

      <div className="chart-bars">
        {entries.map(([emotion, prob], idx) => {
          const pct = (prob * 100).toFixed(1);
          const isTop = emotion === topEmotion;
          const gradient = EMOTION_GRADIENTS[emotion] || 'linear-gradient(90deg, #6366F1, #06B6D4)';
          const emoji = EMOTION_EMOJIS[emotion] || '✨';

          return (
            <div
              key={emotion}
              className={`bar-row ${isTop ? 'bar-top' : ''}`}
              title={`${emotion}: ${pct}% probability`}
            >
              <div className="bar-label">
                <span className="bar-emoji">{emoji}</span>
                <span className="bar-name">{emotion}</span>
                {isTop && <span className="top-pill">TOP</span>}
              </div>

              <div className="bar-track">
                <div
                  className={`bar-fill ${isTop ? 'bar-fill-top' : ''}`}
                  style={{
                    width: `${Math.max(prob * 100, 2)}%`,
                    background: gradient,
                  }}
                >
                  <span className="shimmer-glint"></span>
                </div>
              </div>

              <div className="bar-value-wrapper">
                <span className="bar-value">{pct}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
