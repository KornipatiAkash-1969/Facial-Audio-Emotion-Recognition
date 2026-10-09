import React from 'react';

const EMOTION_COLORS = {
  Angry: '#E74C3C',
  Disgust: '#27AE60',
  Fear: '#8E44AD',
  Happy: '#F39C12',
  Neutral: '#7F8C8D',
  Sad: '#2980B9',
  Surprise: '#E67E22',
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
      <h4 className="chart-title">Emotion Probability Distribution</h4>
      <div className="chart-bars">
        {entries.map(([emotion, prob]) => {
          const pct = (prob * 100).toFixed(1);
          const isTop = emotion === topEmotion;
          const color = EMOTION_COLORS[emotion] || '#3B82F6';
          const emoji = EMOTION_EMOJIS[emotion] || '✨';

          return (
            <div key={emotion} className={`bar-row ${isTop ? 'bar-top' : ''}`}>
              <div className="bar-label">
                <span className="bar-emoji">{emoji}</span>
                <span className="bar-name">{emotion}</span>
              </div>
              <div className="bar-track">
                <div
                  className="bar-fill"
                  style={{
                    width: `${Math.max(prob * 100, 1)}%`,
                    backgroundColor: color,
                  }}
                />
              </div>
              <span className="bar-value">{pct}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
