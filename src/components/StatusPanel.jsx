import React from 'react';

const StatusPanel = ({ 
  connectionStatus, 
  errorMessage, 
  vehicleCount,
  showOptimizationForm, 
  setShowOptimizationForm 
}) => {
  return (
    <div
      style={{
        position: "absolute",
        top: "10px",
        left: "10px",
        zIndex: 100,
        padding: "10px",
        backgroundColor: "rgba(255,255,255,0.7)",
        borderRadius: "4px",
        maxWidth: "300px",
      }}
    >
      <div>
        <strong>Estado:</strong> {connectionStatus}
      </div>
      {errorMessage && (
        <div style={{ color: "red" }}>{errorMessage}</div>
      )}
      <div>
        <strong>Vehículos:</strong> {vehicleCount}
      </div>
      <button 
        onClick={() => setShowOptimizationForm(!showOptimizationForm)}
        style={{
          padding: "6px 10px",
          marginTop: "8px",
          backgroundColor: "#4CAF50",
          color: "white",
          border: "none",
          borderRadius: "4px",
          cursor: "pointer"
        }}
      >
        {showOptimizationForm ? "Ocultar Optimización" : "Mostrar Optimización"}
      </button>
    </div>
  );
};

export default StatusPanel;