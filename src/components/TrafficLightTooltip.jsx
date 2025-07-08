/**
 * Tooltip para semáforos con información detallada
 * Muestra datos del sistema modular de semáforos
 */

import React from 'react';
import './TrafficLightTooltip.css';

const TrafficLightTooltip = ({ 
  trafficLight, 
  position, 
  visible = false,
  showAdvancedInfo = false 
}) => {
  if (!visible || !trafficLight || !position) {
    return null;
  }

  // Formatear tiempo restante en la fase actual
  const formatTimeRemaining = (seconds) => {
    if (!seconds || seconds < 0) return 'N/A';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  // Formatear progreso del ciclo
  const formatCycleProgress = (progress) => {
    if (progress === undefined || progress === null) return 'N/A';
    return `${Math.round(progress * 100)}%`;
  };

  // Formatear eficiencia
  const formatEfficiency = (efficiency) => {
    if (!efficiency && efficiency !== 0) return 'N/A';
    return `${Math.round(efficiency * 100)}%`;
  };

  // Determinar el estado visual del semáforo
  const getStateIcon = (state) => {
    const icons = {
      green: '🟢',
      yellow: '🟡', 
      red: '🔴',
      off: '⚫'
    };
    return icons[state] || '❓';
  };

  // Determinar color de fondo basado en el estado
  const getStateColor = (state) => {
    const colors = {
      green: '#e8f5e8',
      yellow: '#fff9e6',
      red: '#ffe6e6',
      off: '#f0f0f0'
    };
    return colors[state] || '#f8f9fa';
  };

  return (
    <div 
      className="traffic-light-tooltip"
      style={{
        left: position.x + 15,
        top: position.y - 10,
        backgroundColor: getStateColor(trafficLight.state)
      }}
    >
      {/* Encabezado principal */}
      <div className="tooltip-header">
        <span className="state-icon">{getStateIcon(trafficLight.state)}</span>
        <div className="header-info">
          <div className="light-id">Semáforo #{trafficLight.node_id}</div>
          <div className="state-text">Estado: {trafficLight.state?.toUpperCase()}</div>
        </div>
      </div>

      {/* Información básica */}
      <div className="tooltip-section">
        <div className="info-row">
          <span className="label">📍 Ubicación:</span>
          <span className="value">
            {trafficLight.lat?.toFixed(6)}, {trafficLight.lon?.toFixed(6)}
          </span>
        </div>
        
        <div className="info-row">
          <span className="label">🏁 Zona:</span>
          <span className="value">Zona {trafficLight.zone || 0}</span>
        </div>

        <div className="info-row">
          <span className="label">🧭 Dirección:</span>
          <span className="value">{trafficLight.direction || 'N/A'}</span>
        </div>
      </div>

      {/* Información de temporización */}
      <div className="tooltip-section">
        <div className="section-title">⏱️ Temporización</div>
        
        <div className="info-row">
          <span className="label">Tiempo restante:</span>
          <span className="value">{formatTimeRemaining(trafficLight.phase_remaining)}</span>
        </div>
        
        <div className="info-row">
          <span className="label">Progreso del ciclo:</span>
          <span className="value">{formatCycleProgress(trafficLight.cycle_progress)}</span>
        </div>
      </div>

      {/* Características especiales */}
      {(trafficLight.adaptive || trafficLight.emergency_override) && (
        <div className="tooltip-section">
          <div className="section-title">⚡ Características</div>
          
          {trafficLight.adaptive && (
            <div className="info-row special">
              <span className="label">🧠 Adaptativo:</span>
              <span className="value active">Activo</span>
            </div>
          )}
          
          {trafficLight.emergency_override && (
            <div className="info-row special emergency">
              <span className="label">🚨 Emergencia:</span>
              <span className="value emergency">Override Activo</span>
            </div>
          )}
        </div>
      )}

      {/* Información avanzada (solo si está habilitada) */}
      {showAdvancedInfo && (
        <>
          <div className="tooltip-section">
            <div className="section-title">📊 Rendimiento</div>
            
            <div className="info-row">
              <span className="label">Eficiencia:</span>
              <span className={`value ${trafficLight.efficiency < 0.7 ? 'warning' : 'good'}`}>
                {formatEfficiency(trafficLight.efficiency)}
              </span>
            </div>
            
            {trafficLight.light_id && (
              <div className="info-row">
                <span className="label">ID Sistema:</span>
                <span className="value technical">{trafficLight.light_id}</span>
              </div>
            )}
          </div>

          {/* Barra de progreso del ciclo */}
          {trafficLight.cycle_progress !== undefined && (
            <div className="tooltip-section">
              <div className="section-title">📈 Progreso del Ciclo</div>
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ 
                    width: `${Math.round(trafficLight.cycle_progress * 100)}%`,
                    backgroundColor: trafficLight.state === 'green' ? '#4CAF50' : 
                                   trafficLight.state === 'yellow' ? '#FF9800' : '#F44336'
                  }}
                ></div>
              </div>
              <div className="progress-text">
                {formatCycleProgress(trafficLight.cycle_progress)}
              </div>
            </div>
          )}
        </>
      )}

      {/* Pie del tooltip */}
      <div className="tooltip-footer">
        <small>💡 Click para más opciones</small>
      </div>
    </div>
  );
};

export default TrafficLightTooltip;
