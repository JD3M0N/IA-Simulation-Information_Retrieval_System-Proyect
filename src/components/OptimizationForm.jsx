import React, { useState, useEffect, memo } from "react";

const OptimizationForm = memo(({ 
  optimizationParams, 
  setOptimizationParams, 
  requestOptimization, 
  setOptimizedRoutes 
}) => {
  // Estado local para los valores de entrada
  const [localStartPoint, setLocalStartPoint] = useState(optimizationParams.start_point);
  const [localTargetPoints, setLocalTargetPoints] = useState(
    optimizationParams.target_points.join(',')
  );
  const [localNumTrucks, setLocalNumTrucks] = useState(optimizationParams.num_trucks);
  
  // Manejar la sincronización del estado local cuando cambian los props
  useEffect(() => {
    setLocalStartPoint(optimizationParams.start_point);
    setLocalTargetPoints(optimizationParams.target_points.join(','));
    setLocalNumTrucks(optimizationParams.num_trucks);
  }, [optimizationParams]);
  
  // Función para aplicar los cambios al estado principal
  const applyChanges = () => {
    setOptimizationParams({
      ...optimizationParams,
      start_point: localStartPoint,
      target_points: localTargetPoints.split(',').map(p => p.trim()).filter(p => p),
      num_trucks: parseInt(localNumTrucks)
    });
  };
  
  return (
    <div style={{
      position: "absolute",
      top: "10px",
      right: "10px",
      zIndex: 100,
      padding: "15px",
      backgroundColor: "rgba(255,255,255,0.8)",
      borderRadius: "4px",
      maxWidth: "100%"
    }}>
      <h3>Configuración de Optimización</h3>
      
      <div style={{ marginBottom: "10px" }}>
        <label>
          Punto de inicio:
          <input 
            type="text" 
            value={localStartPoint} 
            onChange={e => setLocalStartPoint(e.target.value)}
            style={{ width: "100%", padding: "5px", marginTop: "5px" }}
          />
        </label>
      </div>
      
      <div style={{ marginBottom: "10px" }}>
        <label>
          Puntos objetivo (separados por coma):
          <input 
            type="text" 
            value={localTargetPoints} 
            onChange={e => setLocalTargetPoints(e.target.value)}
            style={{ width: "100%", padding: "5px", marginTop: "5px" }}
          />
        </label>
      </div>
      
      <div style={{ marginBottom: "15px" }}>
        <label>
          Número de camiones:
          <input 
            type="number" 
            value={localNumTrucks} 
            onChange={e => setLocalNumTrucks(e.target.value)}
            min="1"
            style={{ width: "100%", padding: "5px", marginTop: "5px" }}
          />
        </label>
      </div>
      
      <div style={{ display: "flex", gap: "10px" }}>
        <button 
          onClick={() => {
            applyChanges();
            requestOptimization();
          }}
          style={{
            flex: 1,
            padding: "8px",
            backgroundColor: "#2196F3",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer"
          }}
        >
          Optimizar Rutas
        </button>
        
        <button 
          onClick={() => setOptimizedRoutes([])}
          style={{
            flex: 1,
            padding: "8px",
            backgroundColor: "#f44336",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer"
          }}
        >
          Limpiar Rutas
        </button>
      </div>
    </div>
  );
});

export default OptimizationForm;