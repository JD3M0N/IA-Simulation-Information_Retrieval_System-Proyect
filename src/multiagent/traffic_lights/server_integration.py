"""
Integración del Sistema de Semáforos para el Servidor WebSocket
Facilita la conexión entre el servidor principal y el sistema modular de semáforos
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Importar el sistema modular de semáforos
from src.multiagent.traffic_lights import (
    TrafficLightController,
    TrafficLightIntegration,
    initialize_traffic_light_system
)


class ServerTrafficLightManager:
    """
    Manager específico para el servidor que facilita el uso del sistema modular
    """
    
    def __init__(self):
        """Inicializa el manager de semáforos para el servidor"""
        self.logger = logging.getLogger("ServerTrafficLightManager")
        
        # Componentes del sistema modular
        self.traffic_integration: Optional[TrafficLightIntegration] = None
        self.traffic_controller: Optional[TrafficLightController] = None
        
        # Estado de integración
        self.is_initialized = False
        self.environment = None
        
        self.logger.info("Manager de semáforos del servidor inicializado")
    
    async def initialize_with_environment(self, environment):
        """
        Inicializa el sistema de semáforos con un environment específico
        
        Args:
            environment: Environment del sistema multiagente
        """
        try:
            self.logger.info("Inicializando sistema modular de semáforos...")
            self.environment = environment
            
            # Inicializar el sistema integrado
            self.traffic_integration = await initialize_traffic_light_system(environment)
            self.traffic_controller = self.traffic_integration.traffic_controller
            
            self.is_initialized = True
            self.logger.info("✅ Sistema modular de semáforos inicializado correctamente")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error inicializando sistema de semáforos: {e}")
            return False
    
    def get_traffic_lights_for_websocket(self) -> List[Dict[str, Any]]:
        """
        Obtiene datos de semáforos formateados para envío por WebSocket
        
        Returns:
            Lista de diccionarios con datos de semáforos
        """
        if not self.is_initialized or not self.traffic_integration:
            return []
        
        try:
            traffic_data = []
            traffic_states = self.traffic_integration.get_traffic_light_status()
            
            for light_id, status in traffic_states.items():
                # Obtener datos adicionales del agente
                light_agent = self.traffic_controller.traffic_lights.get(light_id)
                
                if light_agent and light_agent.data:
                    traffic_data.append({
                        "node_id": light_agent.data.node_id,
                        "lat": light_agent.data.latitude,
                        "lon": light_agent.data.longitude,
                        "state": status["current_phase"].value.lower(),  # green, yellow, red
                        "zone": self._calculate_zone(light_agent.data.latitude, light_agent.data.longitude),
                        "direction": status.get("primary_direction", "north").lower(),
                        "light_id": light_id,
                        "phase_remaining": status.get("phase_remaining_time", 0),
                        "cycle_progress": status.get("cycle_progress", 0.0),
                        "adaptive": status.get("is_adaptive", False),
                        "emergency_override": status.get("emergency_override", False)
                    })
            
            return traffic_data
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de semáforos: {e}")
            return []
    
    def _calculate_zone(self, lat: float, lon: float) -> int:
        """Calcula zona basada en coordenadas (compatibilidad con sistema legacy)"""
        # Simple zona basada en cuadrantes
        if lat > 23.115 and lon > -82.370:
            return 1
        elif lat > 23.115:
            return 2
        elif lon > -82.370:
            return 3
        else:
            return 0
    
    async def modify_traffic_light(self, node_id: int, new_state: str, duration: float = None) -> bool:
        """
        Modifica el estado de un semáforo específico
        
        Args:
            node_id: ID del nodo del semáforo
            new_state: Nuevo estado (green, yellow, red)
            duration: Duración opcional en segundos
            
        Returns:
            bool: True si se modificó correctamente
        """
        if not self.is_initialized or not self.traffic_integration:
            return False
        
        try:
            # Buscar el light_id correspondiente al node_id
            light_id = self.traffic_integration.new_to_legacy_mapping.get(node_id)
            if not light_id:
                # Buscar en el mapeo inverso
                light_id = self.traffic_integration.legacy_to_new_mapping.get(node_id)
            
            if not light_id:
                self.logger.warning(f"No se encontró semáforo para nodo {node_id}")
                return False
            
            # Usar el controlador para modificar el semáforo
            success = await self.traffic_controller.set_traffic_light_state(
                light_id, new_state, duration
            )
            
            if success:
                self.logger.info(f"Semáforo {light_id} (nodo {node_id}) cambiado a {new_state}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error modificando semáforo {node_id}: {e}")
            return False
    
    async def handle_emergency_at_location(self, lat: float, lon: float, radius: float = 0.01) -> bool:
        """
        Maneja emergencia cerca de una ubicación específica
        
        Args:
            lat: Latitud del evento
            lon: Longitud del evento  
            radius: Radio de influencia en grados
            
        Returns:
            bool: True si se activó el protocolo de emergencia
        """
        if not self.is_initialized or not self.traffic_controller:
            return False
        
        try:
            # Activar protocolo de emergencia en el área
            success = await self.traffic_controller.activate_emergency_protocol(
                emergency_location=(lat, lon),
                affected_radius=radius
            )
            
            if success:
                self.logger.info(f"Protocolo de emergencia activado en ({lat:.6f}, {lon:.6f})")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error activando emergencia: {e}")
            return False
    
    def get_traffic_light_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas del sistema de semáforos
        
        Returns:
            Diccionario con métricas del sistema
        """
        if not self.is_initialized or not self.traffic_integration:
            return {}
        
        try:
            return self.traffic_integration.get_system_metrics()
        except Exception as e:
            self.logger.error(f"Error obteniendo métricas: {e}")
            return {}
    
    async def optimize_traffic_network(self) -> bool:
        """
        Ejecuta optimización de la red de semáforos
        
        Returns:
            bool: True si la optimización fue exitosa
        """
        if not self.is_initialized or not self.traffic_controller:
            return False
        
        try:
            success = await self.traffic_controller.optimize_network()
            
            if success:
                self.logger.info("Optimización de red de semáforos completada")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error en optimización: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Verifica si el sistema está listo para usar"""
        return self.is_initialized and self.traffic_integration is not None


# Instancia global del manager para el servidor
server_traffic_manager = ServerTrafficLightManager()


async def initialize_server_traffic_lights(environment) -> bool:
    """
    Función helper para inicializar semáforos en el servidor
    
    Args:
        environment: Environment del sistema multiagente
        
    Returns:
        bool: True si se inicializó correctamente
    """
    return await server_traffic_manager.initialize_with_environment(environment)


def get_server_traffic_lights_data() -> List[Dict[str, Any]]:
    """
    Función helper para obtener datos de semáforos formateados para WebSocket
    
    Returns:
        Lista de datos de semáforos
    """
    return server_traffic_manager.get_traffic_lights_for_websocket()


async def modify_server_traffic_light(node_id: int, new_state: str, duration: float = None) -> bool:
    """
    Función helper para modificar un semáforo desde el servidor
    
    Args:
        node_id: ID del nodo
        new_state: Nuevo estado
        duration: Duración opcional
        
    Returns:
        bool: True si se modificó correctamente
    """
    return await server_traffic_manager.modify_traffic_light(node_id, new_state, duration)


def get_server_traffic_metrics() -> Dict[str, Any]:
    """
    Función helper para obtener métricas del sistema
    
    Returns:
        Diccionario con métricas
    """
    return server_traffic_manager.get_traffic_light_metrics()
