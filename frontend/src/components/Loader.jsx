import React from 'react';

export default function Loader({ size = 'medium', text = '', className = '' }) {
  return (
    <div className={`modern-loader-container ${size} ${className}`}>
      <div className="modern-spinner">
        <div className="spinner-ring" />
        <div className="spinner-core" />
      </div>
      {text && <p className="loader-text">{text}</p>}
    </div>
  );
}
