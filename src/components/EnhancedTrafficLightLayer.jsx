/**
 * Capa mejorada para visualización de semáforos
 * Componente modular y escalable para el sistema de semáforos inteligentes
 */

import React, { useMemo, useState, useCallback } from 'react';
import { IconLayer, ScatterplotLayer } from '@deck.gl/layers';

// Configuración de iconos para semáforos
const TRAFFIC_LIGHT_MAPPING = {
  green: {
    url: '/icons/traffic_light_green.png',
    width: 64,
    height: 64,
    color: [0, 255, 0, 200],
    fallbackColor: [0, 255, 0, 200]
  },
  yellow: {
    url: '/icons/traffic_light_yellow.png', 
    width: 64,
    height: 64,
    color: [255, 255, 0, 200],
    fallbackColor: [255, 255, 0, 200]
  },
  red: {
    url: '/icons/traffic_light_red.png',
    width: 64,
    height: 64,
    color: [255, 0, 0, 200],
    fallbackColor: [255, 0, 0, 200]
  },
  off: {
    url: '/icons/traffic_light_off.png',
    width: 64,
    height: 64,
    color: [128, 128, 128, 150],
    fallbackColor: [128, 128, 128, 150]
  }
};

// Configuración de zona de influencia
const ZONE_COLORS = {
  0: [100, 100, 255, 100],    // Azul claro
  1: [100, 255, 100, 100],    // Verde claro  
  2: [255, 255, 100, 100],    // Amarillo claro
  3: [255, 100, 100, 100],    // Rojo claro
  4: [255, 100, 255, 100]     // Magenta claro
};

/**
 * Componente principal para la capa de semáforos mejorada
 */
const EnhancedTrafficLightLayer = ({
  trafficLights = [],
  onTrafficLightClick = null,
  onTrafficLightHover = null,
  showInfluenceZones = false,
  showPerformanceIndicators = false,
  showAdaptiveInfo = false,
  selectedTrafficLight = null,
  highlightEmergencyOverrides = true,
  layerOpacity = 1.0,
  iconSize = 1.0
}) => {
  
  const [hoveredLight, setHoveredLight] = useState(null);

  // Procesar datos de semáforos para optimizar el renderizado
  const processedTrafficLights = useMemo(() => {
    return trafficLights.map(light => ({
      ...light,
      // Asegurar que tenemos todos los campos necesarios
      state: light.state || 'off',
      zone: light.zone !== undefined ? light.zone : 0,
      adaptive: light.adaptive || false,
      emergency_override: light.emergency_override || false,
      phase_remaining: light.phase_remaining || 0,
      cycle_progress: light.cycle_progress || 0,
      // Calcular posición
      position: [light.lon, light.lat],
      // ID único para tracking
      uniqueId: light.light_id || `traffic_light_${light.node_id}`,
      // Información de rendimiento
      efficiency: light.efficiency || 0.75
    }));
  }, [trafficLights]);

  // Datos para zonas de influencia
  const influenceZones = useMemo(() => {
    if (!showInfluenceZones) return [];
    
    return processedTrafficLights.map(light => ({
      position: light.position,
      zone: light.zone,
      radius: 50 + (light.zone * 20), // Radio variable por zona
      color: ZONE_COLORS[light.zone] || ZONE_COLORS[0]
    }));
  }, [processedTrafficLights, showInfluenceZones]);

  // Manejar eventos de hover
  const handleHover = useCallback((info) => {
    if (info.object) {
      setHoveredLight(info.object);
      if (onTrafficLightHover) {
        onTrafficLightHover(info.object, info);
      }
    } else {
      setHoveredLight(null);
      if (onTrafficLightHover) {
        onTrafficLightHover(null, info);
      }
    }
  }, [onTrafficLightHover]);

  // Manejar eventos de click
  const handleClick = useCallback((info) => {
    if (info.object && onTrafficLightClick) {
      onTrafficLightClick(info.object, info);
    }
  }, [onTrafficLightClick]);

  // Función para obtener el tamaño dinámico del icono
  const getIconSize = useCallback((light) => {
    let baseSize = 32 * iconSize;
    
    // Aumentar tamaño si está seleccionado
    if (selectedTrafficLight && light.uniqueId === selectedTrafficLight.uniqueId) {
      baseSize *= 1.5;
    }
    
    // Aumentar tamaño si hay override de emergencia
    if (highlightEmergencyOverrides && light.emergency_override) {
      baseSize *= 1.3;
    }
    
    // Aumentar tamaño si está en hover
    if (hoveredLight && light.uniqueId === hoveredLight.uniqueId) {
      baseSize *= 1.2;
    }
    
    return baseSize;
  }, [selectedTrafficLight, hoveredLight, highlightEmergencyOverrides, iconSize]);

  // Función para obtener el color del semáforo
  const getTrafficLightColor = useCallback((light) => {
    const mapping = TRAFFIC_LIGHT_MAPPING[light.state] || TRAFFIC_LIGHT_MAPPING.off;
    let color = [...mapping.fallbackColor];
    
    // Modificar opacidad basada en la configuración de la capa
    color[3] = Math.round(color[3] * layerOpacity);
    
    // Resaltar si hay override de emergencia
    if (highlightEmergencyOverrides && light.emergency_override) {
      // Añadir efecto de parpadeo o borde rojo
      color = [255, 100, 100, color[3]];
    }
    
    // Resaltar si está seleccionado
    if (selectedTrafficLight && light.uniqueId === selectedTrafficLight.uniqueId) {
      // Añadir brillo
      color = color.map((c, i) => i < 3 ? Math.min(255, c + 50) : c);
    }
    
    return color;
  }, [selectedTrafficLight, highlightEmergencyOverrides, layerOpacity]);

  // Capas de visualización
  const layers = useMemo(() => {
    const result = [];

    // 1. Capa de zonas de influencia (si está habilitada)
    if (showInfluenceZones && influenceZones.length > 0) {
      result.push(
        new ScatterplotLayer({
          id: 'traffic-light-influence-zones',
          data: influenceZones,
          pickable: false,
          stroked: true,
          filled: true,
          opacity: 0.3,
          getPosition: d => d.position,
          getRadius: d => d.radius,
          getFillColor: d => d.color,
          getLineColor: d => [...d.color.slice(0, 3), 255],
          getLineWidth: 2,
          radiusUnits: 'meters'
        })
      );
    }

    // 2. Capa principal de semáforos con iconos
    result.push(
      new IconLayer({
        id: 'traffic-lights-icons',
        data: processedTrafficLights,
        pickable: true,
        sizeUnits: 'pixels',
        sizeScale: 1,
        alphaCutoff: 0.1,
        getPosition: d => d.position,
        getIcon: d => ({
          url: TRAFFIC_LIGHT_MAPPING[d.state]?.url || TRAFFIC_LIGHT_MAPPING.off.url,
          width: TRAFFIC_LIGHT_MAPPING[d.state]?.width || 64,
          height: TRAFFIC_LIGHT_MAPPING[d.state]?.height || 64,
          anchorY: 32,
          anchorX: 32
        }),
        getSize: getIconSize,
        getColor: getTrafficLightColor,
        onHover: handleHover,
        onClick: handleClick,
        updateTriggers: {
          getSize: [selectedTrafficLight, hoveredLight, iconSize],
          getColor: [selectedTrafficLight, highlightEmergencyOverrides, layerOpacity]
        }
      })
    );

    // 3. Capa de fallback con círculos (si no hay iconos disponibles)
    result.push(
      new ScatterplotLayer({
        id: 'traffic-lights-fallback',
        data: processedTrafficLights,
        pickable: true,
        stroked: true,
        filled: true,
        opacity: 0.8,
        getPosition: d => d.position,
        getRadius: d => getIconSize(d) / 2,
        getFillColor: getTrafficLightColor,
        getLineColor: [255, 255, 255, 200],
        getLineWidth: 2,
        radiusMinPixels: 8,
        radiusMaxPixels: 50,
        visible: false, // Activar solo si los iconos fallan
        onHover: handleHover,
        onClick: handleClick,
        updateTriggers: {
          getRadius: [selectedTrafficLight, hoveredLight, iconSize],
          getFillColor: [selectedTrafficLight, highlightEmergencyOverrides, layerOpacity]
        }
      })
    );

    // 4. Capa de indicadores de rendimiento (si está habilitada)
    if (showPerformanceIndicators) {
      result.push(
        new ScatterplotLayer({
          id: 'traffic-light-performance',
          data: processedTrafficLights.filter(d => d.efficiency < 0.7), // Solo mostrar los de bajo rendimiento
          pickable: false,
          stroked: true,
          filled: false,
          getPosition: d => d.position,
          getRadius: d => getIconSize(d) + 15,
          getLineColor: [255, 165, 0, 180], // Naranja para alertas
          getLineWidth: 3,
          radiusUnits: 'pixels'
        })
      );
    }

    // 5. Capa de información adaptativa (si está habilitada)
    if (showAdaptiveInfo) {
      result.push(
        new ScatterplotLayer({
          id: 'traffic-light-adaptive-info',
          data: processedTrafficLights.filter(d => d.adaptive),
          pickable: false,
          stroked: true,
          filled: false,
          getPosition: d => d.position,
          getRadius: d => getIconSize(d) + 25,
          getLineColor: [0, 255, 255, 150], // Cyan para modo adaptativo
          getLineWidth: 2,
          radiusUnits: 'pixels'
        })
      );
    }

    return result;
  }, [
    processedTrafficLights,
    influenceZones,
    showInfluenceZones,
    showPerformanceIndicators,
    showAdaptiveInfo,
    getIconSize,
    getTrafficLightColor,
    handleHover,
    handleClick
  ]);

  return layers;
};

export default EnhancedTrafficLightLayer;
