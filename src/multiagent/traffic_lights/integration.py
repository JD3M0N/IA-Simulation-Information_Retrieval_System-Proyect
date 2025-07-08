"""
Integración del Sistema de Semáforos con el Entorno Multi-Agente
Conecta los semáforos modulares con el sistema existente
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import networkx as nx

from .traffic_light_controller import TrafficLightController
from .traffic_light_agent import TrafficLightAgent
from .traffic_light_models import TrafficLightData, TrafficLightPhase, TrafficDirection
from .traffic_light_optimization import TrafficLightOptimizer

# Importar del sistema existente
from ..environment import Environment
from ..civilian_traffic import CivilianTrafficAgent


class TrafficLightIntegration:
    """
    Clase de integración que conecta el sistema de semáforos modular
    con el entorno multi-agente existente
    """
    
    def __init__(self, environment: Environment):
        """
        Inicializa la integración
        
        Args:
            environment: Entorno multi-agente existente
        """
        self.logger = logging.getLogger("TrafficLightIntegration")
        self.environment = environment
        
        # Inicializar controlador de semáforos
        self.traffic_controller = TrafficLightController(environment.street_graph)
        
        # Inicializar optimizador
        self.optimizer = TrafficLightOptimizer()
        
        # Mapeo entre sistemas
        self.legacy_to_new_mapping: Dict[int, str] = {}  # node_id -> light_id
        self.new_to_legacy_mapping: Dict[str, int] = {}  # light_id -> node_id
        
        # Estado de integración
        self.is_integrated = False
        self.sync_interval = 5.0  # segundos
        
        self.logger.info("Integración de semáforos inicializada")
    
    async def integrate_existing_traffic_lights(self):
        """Integra semáforos existentes del environment con el nuevo sistema"""
        try:
            self.logger.info("Iniciando integración de semáforos existentes")
            
            # Migrar semáforos del sistema legacy
            migrated_count = 0
            
            for node_id, legacy_light in self.environment.traffic_lights.items():
                success = await self._migrate_legacy_traffic_light(node_id, legacy_light)
                if success:
                    migrated_count += 1
            
            self.logger.info(f"Migrados {migrated_count} semáforos al nuevo sistema")
            
            # Iniciar el controlador
            await self.traffic_controller.start_system()
            
            # Iniciar sincronización
            asyncio.create_task(self._sync_systems())
            
            self.is_integrated = True
            
        except Exception as e:
            self.logger.error(f"Error en integración: {e}")
            raise
    
    async def _migrate_legacy_traffic_light(self, node_id: int, legacy_light) -> bool:
        """Migra un semáforo del sistema legacy al nuevo"""
        try:
            # Obtener coordenadas del nodo
            node_data = self.environment.street_graph.nodes.get(node_id, {})
            latitude = node_data.get('lat', 0.0)
            longitude = node_data.get('lon', 0.0)
            
            # Crear ID único para el nuevo semáforo
            light_id = f"traffic_light_{node_id}"
            
            # Añadir al controlador
            success = await self.traffic_controller.add_traffic_light(
                light_id=light_id,
                node_id=node_id,
                latitude=latitude,
                longitude=longitude
            )
            
            if success:
                # Configurar estado inicial desde legacy
                new_light = self.traffic_controller.traffic_lights[light_id]
                
                # Mapear estado del sistema legacy
                legacy_state = getattr(legacy_light, 'state', 'red')
                if legacy_state == 'green':
                    new_phase = TrafficLightPhase.GREEN
                elif legacy_state == 'yellow':
                    new_phase = TrafficLightPhase.YELLOW
                else:
                    new_phase = TrafficLightPhase.RED
                
                new_light.data.current_phase = new_phase
                new_light.data.phase_start_time = datetime.now()
                
                # Configurar tiempos desde legacy
                if hasattr(legacy_light, 'green_duration'):
                    new_light.data.phase_config[TrafficLightPhase.GREEN].duration = legacy_light.green_duration
                if hasattr(legacy_light, 'red_duration'):
                    new_light.data.phase_config[TrafficLightPhase.RED].duration = legacy_light.red_duration
                if hasattr(legacy_light, 'yellow_duration'):
                    new_light.data.phase_config[TrafficLightPhase.YELLOW].duration = getattr(legacy_light, 'yellow_duration', 5.0)
                
                # Actualizar mapeos
                self.legacy_to_new_mapping[node_id] = light_id
                self.new_to_legacy_mapping[light_id] = node_id
                
                # Iniciar operación del semáforo
                await new_light.start_operation()
                
                self.logger.debug(f"Semáforo migrado: nodo {node_id} -> {light_id}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error migrando semáforo en nodo {node_id}: {e}")
        
        return False
    
    async def _sync_systems(self):
        """Sincroniza continuamente entre el sistema legacy y el nuevo"""
        while self.is_integrated:
            try:
                # Actualizar datos de vehículos en semáforos
                await self._update_vehicle_data()
                
                # Sincronizar estados de semáforos
                await self._sync_traffic_light_states()
                
                # Aplicar optimizaciones si es necesario
                await self._apply_optimizations()
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                self.logger.error(f"Error en sincronización: {e}")
                await asyncio.sleep(self.sync_interval * 2)
    
    async def _update_vehicle_data(self):
        """Actualiza datos de vehículos en los semáforos"""
        try:
            # Recopilar datos de vehículos del environment
            vehicles_data = {}
            
            # Obtener vehículos civiles
            for agent in self.environment.civilian_vehicles:
                if hasattr(agent, 'agent_id') and hasattr(agent, 'lat') and hasattr(agent, 'lon'):
                    vehicles_data[agent.agent_id] = {
                        "lat": agent.lat,
                        "lon": agent.lon,
                        "speed": getattr(agent, 'current_speed', 0),
                        "emergency": False  # Los vehículos civiles no son de emergencia
                    }
            
            # Actualizar controlador con datos de vehículos
            await self.traffic_controller.update_vehicle_data(vehicles_data)
            
        except Exception as e:
            self.logger.error(f"Error actualizando datos de vehículos: {e}")
    
    async def _sync_traffic_light_states(self):
        """Sincroniza estados entre sistemas legacy y nuevo"""
        try:
            for light_id, traffic_light in self.traffic_controller.traffic_lights.items():
                node_id = self.new_to_legacy_mapping.get(light_id)
                
                if node_id and node_id in self.environment.traffic_lights:
                    legacy_light = self.environment.traffic_lights[node_id]
                    
                    # Sincronizar estado del nuevo al legacy
                    current_phase = traffic_light.data.current_phase
                    
                    if current_phase == TrafficLightPhase.GREEN:
                        legacy_light.state = 'green'
                    elif current_phase == TrafficLightPhase.YELLOW:
                        legacy_light.state = 'yellow'
                    else:
                        legacy_light.state = 'red'
                    
                    # Actualizar último cambio
                    legacy_light.last_change = traffic_light.data.phase_start_time
            
        except Exception as e:
            self.logger.error(f"Error sincronizando estados: {e}")
    
    async def _apply_optimizations(self):
        """Aplica optimizaciones periódicas"""
        try:
            current_time = datetime.now()
            
            # Optimizar cada 10 minutos
            if not hasattr(self, '_last_optimization_time'):
                self._last_optimization_time = current_time
            
            if current_time - self._last_optimization_time >= timedelta(minutes=10):
                await self._run_network_optimization()
                self._last_optimization_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error aplicando optimizaciones: {e}")
    
    async def _run_network_optimization(self):
        """Ejecuta optimización de la red de semáforos"""
        try:
            self.logger.info("Iniciando optimización de red de semáforos")
            
            # Obtener intersecciones para optimizar
            intersections = self.traffic_controller.intersections
            
            if len(intersections) > 0:
                # Ejecutar optimización adaptativa
                results = await self.optimizer.optimize_network(intersections, "adaptive")
                
                # Log resultados
                total_improvement = sum(r.improvement_percent for r in results) / len(results)
                self.logger.info(f"Optimización completada: {total_improvement:.1f}% mejora promedio")
            
        except Exception as e:
            self.logger.error(f"Error en optimización de red: {e}")
    
    # Métodos públicos para interfaz externa
    
    async def handle_emergency_vehicle(self, vehicle_data: Dict[str, Any]) -> bool:
        """
        Maneja vehículo de emergencia en el sistema
        
        Args:
            vehicle_data: Datos del vehículo de emergencia
            
        Returns:
            bool: True si se manejó correctamente
        """
        try:
            vehicle_lat = vehicle_data.get("lat", 0)
            vehicle_lon = vehicle_data.get("lon", 0)
            
            # Encontrar intersección más cercana
            closest_intersection = None
            min_distance = float('inf')
            
            for intersection_id, intersection in self.traffic_controller.intersections.items():
                distance = ((intersection.latitude - vehicle_lat) ** 2 + 
                           (intersection.longitude - vehicle_lon) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_intersection = intersection_id
            
            if closest_intersection and min_distance < 0.01:  # Aproximadamente 1 km
                direction = self._calculate_approach_direction(vehicle_data, closest_intersection)
                success = await self.traffic_controller.handle_emergency_vehicle(
                    closest_intersection, direction
                )
                
                if success:
                    self.logger.warning(f"Vehículo de emergencia procesado en {closest_intersection}")
                    return True
            
        except Exception as e:
            self.logger.error(f"Error manejando vehículo de emergencia: {e}")
        
        return False
    
    def _calculate_approach_direction(self, vehicle_data: Dict[str, Any], 
                                    intersection_id: str) -> str:
        """Calcula dirección de aproximación del vehículo"""
        # Implementación simplificada
        # En la práctica, usaríamos datos de velocidad y ubicaciones anteriores
        
        return "north"  # Por defecto
    
    async def modify_traffic_light_external(self, node_id: int, new_state: str = None, 
                                          duration: int = None) -> bool:
        """
        Modifica un semáforo desde interfaz externa (compatible con sistema legacy)
        
        Args:
            node_id: ID del nodo (sistema legacy)
            new_state: Nuevo estado del semáforo
            duration: Duración en segundos
            
        Returns:
            bool: True si se modificó correctamente
        """
        try:
            light_id = self.legacy_to_new_mapping.get(node_id)
            
            if light_id:
                success = await self.traffic_controller.modify_traffic_light(
                    light_id, new_state, duration
                )
                
                if success:
                    self.logger.info(f"Semáforo modificado externamente: nodo {node_id}")
                    return True
            
        except Exception as e:
            self.logger.error(f"Error modificando semáforo externo: {e}")
        
        return False
    
    def get_traffic_light_status(self, node_id: int = None) -> Dict[str, Any]:
        """
        Obtiene estado de semáforos (compatible con sistema legacy)
        
        Args:
            node_id: ID específico del nodo, None para todos
            
        Returns:
            Dict con estados de semáforos
        """
        try:
            if node_id:
                # Estado específico
                light_id = self.legacy_to_new_mapping.get(node_id)
                if light_id and light_id in self.traffic_controller.traffic_lights:
                    traffic_light = self.traffic_controller.traffic_lights[light_id]
                    return {
                        node_id: {
                            "state": traffic_light.data.current_phase.value,
                            "time_in_phase": traffic_light.data.time_in_phase,
                            "efficiency": traffic_light.data.efficiency_score,
                            "is_operational": traffic_light.data.is_operational
                        }
                    }
                return {}
            else:
                # Todos los estados
                states = {}
                for light_id, traffic_light in self.traffic_controller.traffic_lights.items():
                    node_id_mapped = self.new_to_legacy_mapping.get(light_id)
                    if node_id_mapped:
                        states[node_id_mapped] = {
                            "state": traffic_light.data.current_phase.value,
                            "time_in_phase": traffic_light.data.time_in_phase,
                            "efficiency": traffic_light.data.efficiency_score,
                            "is_operational": traffic_light.data.is_operational
                        }
                return states
            
        except Exception as e:
            self.logger.error(f"Error obteniendo estado de semáforos: {e}")
            return {}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema de semáforos"""
        try:
            controller_status = self.traffic_controller.get_system_status()
            optimization_summary = self.optimizer.get_optimization_summary()
            
            return {
                "integration_status": {
                    "is_integrated": self.is_integrated,
                    "legacy_lights_migrated": len(self.legacy_to_new_mapping),
                    "sync_interval": self.sync_interval
                },
                "controller_status": controller_status,
                "optimization_summary": optimization_summary,
                "performance": {
                    "average_efficiency": self._calculate_average_efficiency(),
                    "total_optimizations": optimization_summary.get("total_optimizations", 0),
                    "system_uptime": self._calculate_uptime()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo métricas: {e}")
            return {"error": str(e)}
    
    def _calculate_average_efficiency(self) -> float:
        """Calcula eficiencia promedio del sistema"""
        try:
            if not self.traffic_controller.traffic_lights:
                return 0.0
            
            total_efficiency = 0.0
            operational_count = 0
            
            for traffic_light in self.traffic_controller.traffic_lights.values():
                if traffic_light.data.is_operational:
                    total_efficiency += traffic_light.data.efficiency_score
                    operational_count += 1
            
            return total_efficiency / max(operational_count, 1)
            
        except Exception:
            return 0.0
    
    def _calculate_uptime(self) -> str:
        """Calcula tiempo de funcionamiento del sistema"""
        if hasattr(self, '_integration_start_time'):
            uptime = datetime.now() - self._integration_start_time
            return str(uptime)
        return "N/A"
    
    async def shutdown(self):
        """Cierra la integración y limpia recursos"""
        try:
            self.is_integrated = False
            
            # Detener controlador
            await self.traffic_controller.stop_system()
            
            # Limpiar mapeos
            self.legacy_to_new_mapping.clear()
            self.new_to_legacy_mapping.clear()
            
            self.logger.info("Integración de semáforos cerrada")
            
        except Exception as e:
            self.logger.error(f"Error cerrando integración: {e}")


# Funciones auxiliares para facilitar la integración

async def initialize_traffic_light_system(environment: Environment) -> TrafficLightIntegration:
    """
    Inicializa y configura el sistema de semáforos integrado
    
    Args:
        environment: Entorno multi-agente existente
        
    Returns:
        TrafficLightIntegration: Sistema integrado listo para usar
    """
    integration = TrafficLightIntegration(environment)
    await integration.integrate_existing_traffic_lights()
    integration._integration_start_time = datetime.now()
    
    return integration


def get_traffic_light_data_for_vehicle(integration: TrafficLightIntegration,
                                     vehicle: CivilianTrafficAgent,
                                     max_distance: float = 0.002) -> Dict[str, Any]:
    """
    Obtiene datos de semáforos visibles para un vehículo específico
    
    Args:
        integration: Sistema de integración
        vehicle: Agente de vehículo
        max_distance: Distancia máxima de visibilidad
        
    Returns:
        Dict con semáforos visibles y sus estados
    """
    try:
        visible_lights = {}
        
        vehicle_lat = getattr(vehicle, 'lat', 0)
        vehicle_lon = getattr(vehicle, 'lon', 0)
        
        for light_id, traffic_light in integration.traffic_controller.traffic_lights.items():
            # Calcular distancia
            light_lat = traffic_light.data.latitude
            light_lon = traffic_light.data.longitude
            
            distance = ((vehicle_lat - light_lat) ** 2 + (vehicle_lon - light_lon) ** 2) ** 0.5
            
            if distance <= max_distance:
                visible_lights[light_id] = {
                    "state": traffic_light.data.current_phase.value,
                    "time_in_phase": traffic_light.data.time_in_phase,
                    "distance": distance * 111000,  # Convertir a metros aproximadamente
                    "node_id": traffic_light.data.node_id
                }
        
        return visible_lights
        
    except Exception as e:
        logging.error(f"Error obteniendo semáforos visibles: {e}")
        return {}
