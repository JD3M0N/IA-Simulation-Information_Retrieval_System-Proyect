/**
 * Utilidades para demostración y funciones de semáforos en el frontend
 * Funciones helper para interactuar con el sistema modular de semáforos
 */

// Configuración de demo
export const DEMO_CONFIG = {
  autoChangeInterval: 5000, // 5 segundos
  emergencyDuration: 10000, // 10 segundos
  states: ['green', 'yellow', 'red'],
  durations: {
    green: 30,
    yellow: 5,
    red: 25
  }
};

/**
 * Simula cambios automáticos de semáforos para demostración
 */
export class TrafficLightDemo {
  constructor(websocket, trafficLights) {
    this.websocket = websocket;
    this.trafficLights = trafficLights;
    this.isRunning = false;
    this.interval = null;
    this.listeners = [];
  }

  // Iniciar demostración automática
  start() {
    if (this.isRunning) return;
    
    this.isRunning = true;
    console.log('🚦 Iniciando demostración automática de semáforos');
    
    this.interval = setInterval(() => {
      this.performRandomChange();
    }, DEMO_CONFIG.autoChangeInterval);

    this.notifyListeners('demo_started');
  }

  // Detener demostración
  stop() {
    if (!this.isRunning) return;
    
    this.isRunning = false;
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    
    console.log('🚦 Demostración automática detenida');
    this.notifyListeners('demo_stopped');
  }

  // Realizar cambio aleatorio
  performRandomChange() {
    if (!this.trafficLights || this.trafficLights.length === 0) return;

    // Seleccionar semáforo aleatorio
    const randomLight = this.trafficLights[Math.floor(Math.random() * this.trafficLights.length)];
    
    // Seleccionar nuevo estado (diferente al actual)
    const availableStates = DEMO_CONFIG.states.filter(state => state !== randomLight.state);
    const newState = availableStates[Math.floor(Math.random() * availableStates.length)];
    
    // Obtener duración apropiada
    const duration = DEMO_CONFIG.durations[newState] || 30;

    // Enviar cambio
    this.changeTrafficLightState(randomLight.node_id, newState, duration);
    
    console.log(`🚦 Demo: Semáforo ${randomLight.node_id} cambiado a ${newState} por ${duration}s`);
    
    this.notifyListeners('light_changed', {
      lightId: randomLight.node_id,
      oldState: randomLight.state,
      newState,
      duration
    });
  }

  // Simular emergencia
  simulateEmergency(lightId = null) {
    const targetLight = lightId ? 
      this.trafficLights.find(light => light.node_id === lightId) :
      this.trafficLights[Math.floor(Math.random() * this.trafficLights.length)];

    if (!targetLight) return false;

    // Activar protocolo de emergencia
    const message = {
      type: 'emergency_event',
      location: {
        lat: targetLight.lat,
        lon: targetLight.lon
      },
      radius: 0.005
    };

    this.websocket.send(JSON.stringify(message));
    
    console.log(`🚨 Emergencia simulada en semáforo ${targetLight.node_id}`);
    
    this.notifyListeners('emergency_simulated', {
      lightId: targetLight.node_id,
      location: { lat: targetLight.lat, lon: targetLight.lon }
    });

    return true;
  }

  // Cambiar estado de semáforo específico
  changeTrafficLightState(nodeId, state, duration = 30) {
    if (!this.websocket) return false;

    const message = {
      type: 'modify_traffic_light',
      node_id: nodeId,
      state: state,
      duration: duration
    };

    this.websocket.send(JSON.stringify(message));
    return true;
  }

  // Optimizar red de semáforos
  optimizeNetwork() {
    if (!this.websocket) return false;

    // En el futuro, cuando esté implementado en el backend
    const message = {
      type: 'optimize_traffic_network'
    };

    console.log('🎯 Solicitando optimización de red de semáforos');
    this.websocket.send(JSON.stringify(message));
    return true;
  }

  // Añadir listener para eventos
  addListener(callback) {
    this.listeners.push(callback);
  }

  // Remover listener
  removeListener(callback) {
    this.listeners = this.listeners.filter(listener => listener !== callback);
  }

  // Notificar a listeners
  notifyListeners(event, data = null) {
    this.listeners.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error('Error en listener de demostración:', error);
      }
    });
  }

  // Actualizar lista de semáforos
  updateTrafficLights(newTrafficLights) {
    this.trafficLights = newTrafficLights;
  }
}

/**
 * Funciones helper para el manejo de semáforos
 */

// Obtener color basado en el estado del semáforo
export const getTrafficLightStateColor = (state) => {
  const colors = {
    green: '#4CAF50',
    yellow: '#FF9800',
    red: '#F44336',
    off: '#9E9E9E'
  };
  return colors[state] || colors.off;
};

// Formatear tiempo restante
export const formatTimeRemaining = (seconds) => {
  if (!seconds || seconds < 0) return 'N/A';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
};

// Calcular estadísticas de estado de semáforos
export const calculateTrafficLightStats = (trafficLights) => {
  if (!trafficLights || trafficLights.length === 0) {
    return {
      total: 0,
      states: { green: 0, yellow: 0, red: 0, off: 0 },
      adaptive: 0,
      emergency: 0,
      efficiency: 0
    };
  }

  const stats = trafficLights.reduce((acc, light) => {
    acc.total++;
    acc.states[light.state] = (acc.states[light.state] || 0) + 1;
    
    if (light.adaptive) acc.adaptive++;
    if (light.emergency_override) acc.emergency++;
    if (light.efficiency) acc.efficiency += light.efficiency;
    
    return acc;
  }, {
    total: 0,
    states: { green: 0, yellow: 0, red: 0, off: 0 },
    adaptive: 0,
    emergency: 0,
    efficiency: 0
  });

  // Calcular eficiencia promedio
  stats.efficiency = stats.total > 0 ? stats.efficiency / stats.total : 0;

  return stats;
};

// Detectar problemas en semáforos
export const detectTrafficLightIssues = (trafficLights) => {
  const issues = [];

  trafficLights.forEach(light => {
    // Eficiencia baja
    if (light.efficiency && light.efficiency < 0.6) {
      issues.push({
        type: 'low_efficiency',
        lightId: light.node_id,
        message: `Semáforo ${light.node_id} tiene baja eficiencia (${Math.round(light.efficiency * 100)}%)`,
        severity: 'warning'
      });
    }

    // Semáforo apagado
    if (light.state === 'off') {
      issues.push({
        type: 'light_off',
        lightId: light.node_id,
        message: `Semáforo ${light.node_id} está apagado`,
        severity: 'error'
      });
    }

    // Override de emergencia activo por mucho tiempo
    if (light.emergency_override) {
      issues.push({
        type: 'emergency_active',
        lightId: light.node_id,
        message: `Semáforo ${light.node_id} en modo emergencia`,
        severity: 'info'
      });
    }
  });

  return issues;
};

// Generar recomendaciones basadas en datos de semáforos
export const generateTrafficLightRecommendations = (trafficLights, systemMetrics) => {
  const recommendations = [];
  const stats = calculateTrafficLightStats(trafficLights);

  // Recomendación de eficiencia general
  if (stats.efficiency < 0.7) {
    recommendations.push({
      type: 'efficiency',
      priority: 'high',
      message: `La eficiencia promedio del sistema es baja (${Math.round(stats.efficiency * 100)}%). Considera optimizar los tiempos de fase.`,
      action: 'optimize_network'
    });
  }

  // Recomendación sobre semáforos adaptativos
  const adaptivePercentage = (stats.adaptive / stats.total) * 100;
  if (adaptivePercentage < 50) {
    recommendations.push({
      type: 'adaptive',
      priority: 'medium',
      message: `Solo ${Math.round(adaptivePercentage)}% de los semáforos son adaptativos. Habilitar más semáforos adaptativos puede mejorar el flujo.`,
      action: 'enable_adaptive'
    });
  }

  // Recomendación sobre distribución de estados
  const redPercentage = (stats.states.red / stats.total) * 100;
  if (redPercentage > 40) {
    recommendations.push({
      type: 'red_lights',
      priority: 'medium',
      message: `${Math.round(redPercentage)}% de los semáforos están en rojo. Esto puede indicar congestión.`,
      action: 'review_timing'
    });
  }

  return recommendations;
};

// Hook personalizado para manejar eventos de semáforos
export const useTrafficLightEvents = (websocket, onEvent) => {
  const handleMessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      
      // Eventos relacionados con semáforos
      if (data.type === 'traffic_light_modified') {
        onEvent('light_modified', data);
      } else if (data.type === 'traffic_metrics') {
        onEvent('metrics_received', data);
      } else if (data.type === 'traffic_light_error') {
        onEvent('error', data);
      }
    } catch (error) {
      console.error('Error procesando mensaje de semáforo:', error);
    }
  };

  if (websocket) {
    websocket.addEventListener('message', handleMessage);
    
    return () => {
      websocket.removeEventListener('message', handleMessage);
    };
  }

  return () => {};
};

export default {
  TrafficLightDemo,
  getTrafficLightStateColor,
  formatTimeRemaining,
  calculateTrafficLightStats,
  detectTrafficLightIssues,
  generateTrafficLightRecommendations,
  useTrafficLightEvents,
  DEMO_CONFIG
};
