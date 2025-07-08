/**
 * Panel de control para semáforos
 * Permite modificar estados y ver métricas del sistema modular
 */

import React, { useState, useEffect } from 'react';
import './TrafficLightControlPanel.css';

const TrafficLightControlPanel = ({ 
  selectedTrafficLight,
  trafficLights = [],
  onStateChange,
  onRequestMetrics,
  systemMetrics = null,
  websocket = null,
  visible = false,
  onClose
}) => {
  const [selectedState, setSelectedState] = useState('green');
  const [duration, setDuration] = useState(30);
  const [isLoading, setIsLoading] = useState(false);
  const [lastAction, setLastAction] = useState(null);
  const [showAdvancedControls, setShowAdvancedControls] = useState(false);

  // Estados disponibles para los semáforos
  const availableStates = [
    { value: 'green', label: 'Verde', icon: '🟢', color: '#4CAF50' },
    { value: 'yellow', label: 'Amarillo', icon: '🟡', color: '#FF9800' },
    { value: 'red', label: 'Rojo', icon: '🔴', color: '#F44336' }
  ];

  // Actualizar estado seleccionado cuando cambia el semáforo
  useEffect(() => {
    if (selectedTrafficLight) {
      setSelectedState(selectedTrafficLight.state || 'green');
    }
  }, [selectedTrafficLight]);

  // Función para cambiar el estado del semáforo
  const handleStateChange = async () => {
    if (!selectedTrafficLight || !websocket) return;

    setIsLoading(true);
    try {
      const message = {
        type: 'modify_traffic_light',
        node_id: selectedTrafficLight.node_id,
        state: selectedState,
        duration: duration
      };

      websocket.send(JSON.stringify(message));
      
      setLastAction({
        type: 'state_change',
        state: selectedState,
        duration: duration,
        timestamp: new Date()
      });

      if (onStateChange) {
        onStateChange(selectedTrafficLight.node_id, selectedState, duration);
      }

    } catch (error) {
      console.error('Error cambiando estado del semáforo:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Función para solicitar métricas del sistema
  const handleRequestMetrics = async () => {
    if (!websocket) return;

    setIsLoading(true);
    try {
      const message = {
        type: 'get_traffic_metrics'
      };

      websocket.send(JSON.stringify(message));
      
      setLastAction({
        type: 'metrics_request',
        timestamp: new Date()
      });

      if (onRequestMetrics) {
        onRequestMetrics();
      }

    } catch (error) {
      console.error('Error solicitando métricas:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Función para activar modo de emergencia
  const handleEmergencyMode = async () => {
    if (!selectedTrafficLight || !websocket) return;

    setIsLoading(true);
    try {
      const message = {
        type: 'emergency_event',
        location: {
          lat: selectedTrafficLight.lat,
          lon: selectedTrafficLight.lon
        },
        radius: 0.005
      };

      websocket.send(JSON.stringify(message));
      
      setLastAction({
        type: 'emergency_mode',
        timestamp: new Date()
      });

    } catch (error) {
      console.error('Error activando modo de emergencia:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Estadísticas rápidas del sistema
  const getSystemStats = () => {
    if (!trafficLights || trafficLights.length === 0) {
      return { total: 0, green: 0, yellow: 0, red: 0, off: 0 };
    }

    return trafficLights.reduce((stats, light) => {
      stats.total++;
      stats[light.state] = (stats[light.state] || 0) + 1;
      return stats;
    }, { total: 0, green: 0, yellow: 0, red: 0, off: 0 });
  };

  const stats = getSystemStats();

  if (!visible) return null;

  return (
    <div className="traffic-light-control-panel">
      {/* Encabezado del panel */}
      <div className="panel-header">
        <h3>🚦 Control de Semáforos</h3>
        <button className="close-btn" onClick={onClose}>×</button>
      </div>

      {/* Información del semáforo seleccionado */}
      {selectedTrafficLight ? (
        <div className="selected-light-info">
          <div className="light-header">
            <span className="light-icon">
              {selectedTrafficLight.state === 'green' ? '🟢' : 
               selectedTrafficLight.state === 'yellow' ? '🟡' : 
               selectedTrafficLight.state === 'red' ? '🔴' : '⚫'}
            </span>
            <div>
              <h4>Semáforo #{selectedTrafficLight.node_id}</h4>
              <p>Estado actual: {selectedTrafficLight.state?.toUpperCase()}</p>
            </div>
          </div>

          {/* Controles de cambio de estado */}
          <div className="state-controls">
            <label>Nuevo Estado:</label>
            <div className="state-selector">
              {availableStates.map(state => (
                <button
                  key={state.value}
                  className={`state-btn ${selectedState === state.value ? 'active' : ''}`}
                  style={{ 
                    '--state-color': state.color,
                    backgroundColor: selectedState === state.value ? state.color : 'transparent'
                  }}
                  onClick={() => setSelectedState(state.value)}
                >
                  <span className="state-icon">{state.icon}</span>
                  <span className="state-label">{state.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Control de duración */}
          <div className="duration-control">
            <label htmlFor="duration">Duración (segundos):</label>
            <div className="duration-input-group">
              <input
                id="duration"
                type="number"
                min="5"
                max="300"
                value={duration}
                onChange={(e) => setDuration(parseInt(e.target.value))}
                className="duration-input"
              />
              <div className="duration-presets">
                {[15, 30, 60, 120].map(preset => (
                  <button
                    key={preset}
                    className={`preset-btn ${duration === preset ? 'active' : ''}`}
                    onClick={() => setDuration(preset)}
                  >
                    {preset}s
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Botón de aplicar cambios */}
          <button
            className="apply-btn"
            onClick={handleStateChange}
            disabled={isLoading}
          >
            {isLoading ? '⏳ Aplicando...' : '✅ Aplicar Cambio'}
          </button>

          {/* Controles avanzados */}
          <div className="advanced-controls">
            <button
              className="toggle-advanced"
              onClick={() => setShowAdvancedControls(!showAdvancedControls)}
            >
              {showAdvancedControls ? '🔽' : '▶️'} Controles Avanzados
            </button>
            
            {showAdvancedControls && (
              <div className="advanced-actions">
                <button
                  className="emergency-btn"
                  onClick={handleEmergencyMode}
                  disabled={isLoading}
                >
                  🚨 Modo Emergencia
                </button>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="no-selection">
          <p>👆 Selecciona un semáforo en el mapa para controlarlo</p>
        </div>
      )}

      {/* Estadísticas rápidas del sistema */}
      <div className="system-stats">
        <h4>📊 Estado del Sistema</h4>
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Total:</span>
            <span className="stat-value">{stats.total}</span>
          </div>
          <div className="stat-item green">
            <span className="stat-label">🟢 Verde:</span>
            <span className="stat-value">{stats.green || 0}</span>
          </div>
          <div className="stat-item yellow">
            <span className="stat-label">🟡 Amarillo:</span>
            <span className="stat-value">{stats.yellow || 0}</span>
          </div>
          <div className="stat-item red">
            <span className="stat-label">🔴 Rojo:</span>
            <span className="stat-value">{stats.red || 0}</span>
          </div>
        </div>

        <button
          className="metrics-btn"
          onClick={handleRequestMetrics}
          disabled={isLoading}
        >
          {isLoading ? '⏳ Cargando...' : '📈 Obtener Métricas Detalladas'}
        </button>
      </div>

      {/* Métricas del sistema (si están disponibles) */}
      {systemMetrics && (
        <div className="detailed-metrics">
          <h4>📈 Métricas Detalladas</h4>
          
          {systemMetrics.performance && (
            <div className="metrics-section">
              <div className="metric-row">
                <span>Eficiencia promedio:</span>
                <span className="metric-value">
                  {Math.round(systemMetrics.performance.average_efficiency * 100)}%
                </span>
              </div>
              <div className="metric-row">
                <span>Eficiencia de red:</span>
                <span className="metric-value">
                  {Math.round(systemMetrics.performance.network_efficiency * 100)}%
                </span>
              </div>
            </div>
          )}

          {systemMetrics.controller_status && (
            <div className="metrics-section">
              <div className="metric-row">
                <span>Semáforos operativos:</span>
                <span className="metric-value">
                  {systemMetrics.controller_status.operational_lights}
                </span>
              </div>
              <div className="metric-row">
                <span>Optimización:</span>
                <span className={`metric-value ${systemMetrics.controller_status.optimization_enabled ? 'active' : 'inactive'}`}>
                  {systemMetrics.controller_status.optimization_enabled ? 'Activa' : 'Inactiva'}
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Última acción */}
      {lastAction && (
        <div className="last-action">
          <small>
            Última acción: {lastAction.type === 'state_change' ? 
              `Cambio a ${lastAction.state} por ${lastAction.duration}s` :
              lastAction.type === 'metrics_request' ? 'Solicitud de métricas' :
              'Modo emergencia activado'
            } - {lastAction.timestamp.toLocaleTimeString()}
          </small>
        </div>
      )}
    </div>
  );
};

export default TrafficLightControlPanel;
