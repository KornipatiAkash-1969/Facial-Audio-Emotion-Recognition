import React, { useState } from 'react';

export default function HistoryTab({ history, onClearHistory }) {
  const [filter, setFilter] = useState('ALL');

  const filtered = history.filter((item) => {
    if (filter === 'ALL') return true;
    return item.modality.toUpperCase().includes(filter);
  });

  const exportCsv = () => {
    if (!history.length) return;
    const headers = ['Index,Timestamp,Modality,Predicted_Emotion,Confidence,Source'];
    const rows = history.map(
      (h, idx) =>
        `${idx + 1},"${h.timestamp}","${h.modality}","${h.emotion}","${(h.confidence * 100).toFixed(1)}%","${h.source || ''}"`
    );
    const csvContent = [headers, ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `emotion_prediction_history_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="tab-container">
      <div className="card">
        <div className="history-header">
          <div>
            <h3 className="card-heading">Prediction History & Analytics</h3>
            <p className="history-sub">
              Logged predictions across Facial, Acoustic, and Multimodal modalities.
            </p>
          </div>

          <div className="history-actions">
            <div className="filter-buttons">
              {['ALL', 'FACE', 'AUDIO', 'MULTIMODAL'].map((f) => (
                <button
                  key={f}
                  className={`btn-filter ${filter === f ? 'filter-active' : ''}`}
                  onClick={() => setFilter(f)}
                >
                  {f}
                </button>
              ))}
            </div>

            <button className="btn btn-secondary" onClick={exportCsv} disabled={!history.length}>
              📥 Export CSV
            </button>
            <button className="btn btn-danger" onClick={onClearHistory} disabled={!history.length}>
              🗑️ Clear
            </button>
          </div>
        </div>

        {filtered.length > 0 ? (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Timestamp</th>
                  <th>Modality</th>
                  <th>Predicted Emotion</th>
                  <th>Confidence</th>
                  <th>Source / Details</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((item, index) => (
                  <tr key={index}>
                    <td>{index + 1}</td>
                    <td>{item.timestamp}</td>
                    <td>
                      <span className="badge-modality">{item.modality}</span>
                    </td>
                    <td>
                      <strong className="emotion-name">{item.emotion}</strong>
                    </td>
                    <td>{(item.confidence * 100).toFixed(1)}%</td>
                    <td className="source-cell">{item.source || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="placeholder-box">
            <span className="placeholder-icon">📜</span>
            <p>No predictions recorded yet. Run predictions from Face, Audio, or Fusion tabs.</p>
          </div>
        )}
      </div>
    </div>
  );
}
