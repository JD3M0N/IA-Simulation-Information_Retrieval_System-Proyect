import React from 'react';
import RouteStats from './RouteStats';

const StatsModal = ({ showDetailedStats, setShowDetailedStats, optimizedRoutes }) => {
  if (!showDetailedStats) return null;
  
  return (
    <div style={{
      position: "absolute",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      zIndex: 150,
      maxWidth: "600px",
      maxHeight: "80vh",
      overflowY: "auto",
      padding: "20px",
      backgroundColor: "white",
      borderRadius: "8px",
      boxShadow: "0 4px 12px rgba(0,0,0,0.2)"
    }}>
      <button 
        onClick={() => setShowDetailedStats(false)}
        style={{
          position: "absolute",
          top: "10px",
          right: "10px",
          background: "none",
          border: "none",
          fontSize: "18px",
          cursor: "pointer"
        }}
      >
        ✕
      </button>
      <RouteStats optimizedRoutes={optimizedRoutes} />
    </div>
  );
};

export default StatsModal;