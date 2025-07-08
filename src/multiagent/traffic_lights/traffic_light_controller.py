"""
Controlador Centralizado de Semáforos
Coordina múltiples semáforos para optimización de tráfico a nivel de red
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import networkx as nx
from collections import defaultdict
import json

from .traffic_light_agent import TrafficLightAgent
from .traffic_light_models import (
    TrafficLightData, TrafficLightPhase, IntersectionData,
    TrafficDirection, PriorityLevel, TrafficLightEvent
)


class TrafficLightController:
    """
    Controlador maestro que coordina múltiples semáforos
    Implementa coordinación de corredores verdes y optimización de red
    """
    
    def __init__(self, street_graph: nx.Graph):
        """
        Inicializa el controlador de semáforos
        
        Args:
            street_graph: Grafo de calles donde están ubicados los semáforos
        """
        self.logger = logging.getLogger("TrafficLightController")
        self.street_graph = street_graph
        
        # Registro de semáforos
        self.traffic_lights: Dict[str, TrafficLightAgent] = {}
        self.intersections: Dict[str, IntersectionData] = {}
        
        # Coordinación
        self.green_wave_corridors: List[List[str]] = []  # Secuencias de semáforos coordinados
        self.coordination_enabled = True
        self.coordination_interval = 30.0  # segundos
        
        # Optimización
        self.optimization_enabled = True
        self.optimization_frequency = timedelta(minutes=10)
        self.last_optimization = datetime.now()
        
        # Eventos y emergencias
        self.active_events: List[TrafficLightEvent] = []
        self.emergency_zones: Set[str] = set()
        self.system_wide_override = False
        
        # Métricas globales
        self.network_metrics = {
            "total_vehicles_processed": 0,
            "average_travel_time": 0.0,
            "network_efficiency": 0.0,
            "total_stops": 0,
            "fuel_consumption": 0.0
        }
        
        # Estado operacional
        self.is_running = False
        self.update_interval = 2.0  # segundos
        
        self.logger.info("Controlador de semáforos inicializado")
    
    async def add_traffic_light(self, light_id: str, node_id: int, 
                              latitude: float = 0.0, longitude: float = 0.0) -> bool:
        """
        Añade un nuevo semáforo al sistema
        
        Args:
            light_id: Identificador único del semáforo
            node_id: ID del nodo en el grafo de calles
            latitude: Latitud de ubicación
            longitude: Longitud de ubicación
            
        Returns:
            bool: True si se añadió correctamente
        """
        try:
            if light_id in self.traffic_lights:
                self.logger.warning(f"Semáforo {light_id} ya existe")
                return False
            
            # Obtener coordenadas del grafo si no se proporcionan
            if latitude == 0.0 and longitude == 0.0 and node_id in self.street_graph.nodes:
                node_data = self.street_graph.nodes[node_id]
                latitude = node_data.get('lat', 0.0)
                longitude = node_data.get('lon', 0.0)
            
            # Crear agente de semáforo
            traffic_light = TrafficLightAgent(
                light_id=light_id,
                node_id=node_id,
                latitude=latitude,
                longitude=longitude
            )
            
            self.traffic_lights[light_id] = traffic_light
            
            # Crear datos de intersección si no existe
            intersection_id = f"intersection_{node_id}"
            if intersection_id not in self.intersections:
                self.intersections[intersection_id] = IntersectionData(
                    intersection_id=intersection_id,
                    node_id=node_id,
                    latitude=latitude,
                    longitude=longitude
                )
            
            # Añadir semáforo a la intersección
            self.intersections[intersection_id].traffic_lights[light_id] = traffic_light.data
            
            self.logger.info(f"Semáforo {light_id} añadido en nodo {node_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error añadiendo semáforo {light_id}: {e}")
            return False
    
    async def remove_traffic_light(self, light_id: str) -> bool:
        """Remueve un semáforo del sistema"""
        try:
            if light_id not in self.traffic_lights:
                return False
            
            # Detener operación del semáforo
            await self.traffic_lights[light_id].stop_operation()
            
            # Remover de registros
            del self.traffic_lights[light_id]
            
            # Remover de intersecciones
            for intersection in self.intersections.values():
                if light_id in intersection.traffic_lights:
                    del intersection.traffic_lights[light_id]
                    break
            
            self.logger.info(f"Semáforo {light_id} removido")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removiendo semáforo {light_id}: {e}")
            return False
    
    async def start_system(self):
        """Inicia el sistema de control de semáforos"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Iniciar todos los semáforos
        for traffic_light in self.traffic_lights.values():
            await traffic_light.start_operation()
        
        # Detectar corredores verdes automáticamente
        await self._detect_green_corridors()
        
        # Iniciar ciclo de coordinación
        asyncio.create_task(self._coordination_cycle())
        
        # Iniciar ciclo de optimización
        asyncio.create_task(self._optimization_cycle())
        
        self.logger.info(f"Sistema de control iniciado con {len(self.traffic_lights)} semáforos")
    
    async def stop_system(self):
        """Detiene el sistema de control"""
        self.is_running = False
        
        # Detener todos los semáforos
        for traffic_light in self.traffic_lights.values():
            await traffic_light.stop_operation()
        
        self.logger.info("Sistema de control detenido")
    
    async def _coordination_cycle(self):
        """Ciclo de coordinación entre semáforos"""
        while self.is_running:
            try:
                if self.coordination_enabled and not self.system_wide_override:
                    await self._coordinate_green_waves()
                    await self._balance_intersection_loads()
                    await self._synchronize_adjacent_lights()
                
                await asyncio.sleep(self.coordination_interval)
                
            except Exception as e:
                self.logger.error(f"Error en ciclo de coordinación: {e}")
                await asyncio.sleep(10)
    
    async def _optimization_cycle(self):
        """Ciclo de optimización de la red"""
        while self.is_running:
            try:
                current_time = datetime.now()
                if (self.optimization_enabled and 
                    current_time - self.last_optimization >= self.optimization_frequency):
                    
                    await self._optimize_network_performance()
                    self.last_optimization = current_time
                
                await asyncio.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                self.logger.error(f"Error en ciclo de optimización: {e}")
                await asyncio.sleep(30)
    
    async def _detect_green_corridors(self):
        """Detecta automáticamente corredores para ondas verdes"""
        try:
            self.green_wave_corridors = []
            
            # Buscar secuencias lineales de semáforos
            processed_lights = set()
            
            for light_id, traffic_light in self.traffic_lights.items():
                if light_id in processed_lights:
                    continue
                
                corridor = await self._find_corridor_from_light(light_id, processed_lights)
                if len(corridor) >= 3:  # Mínimo 3 semáforos para corridor
                    self.green_wave_corridors.append(corridor)
                    processed_lights.update(corridor)
            
            self.logger.info(f"Detectados {len(self.green_wave_corridors)} corredores verdes")
            
        except Exception as e:
            self.logger.error(f"Error detectando corredores: {e}")
    
    async def _find_corridor_from_light(self, start_light_id: str, 
                                      processed_lights: Set[str]) -> List[str]:
        """Encuentra un corridor comenzando desde un semáforo específico"""
        corridor = [start_light_id]
        current_node = self.traffic_lights[start_light_id].data.node_id
        
        # Buscar semáforos conectados en línea recta
        visited_nodes = {current_node}
        
        while True:
            next_light = None
            next_node = None
            
            # Buscar siguiente semáforo en la secuencia
            for neighbor in self.street_graph.neighbors(current_node):
                if neighbor in visited_nodes:
                    continue
                
                # Buscar semáforo en este nodo
                candidate_light = None
                for light_id, traffic_light in self.traffic_lights.items():
                    if (traffic_light.data.node_id == neighbor and 
                        light_id not in processed_lights and 
                        light_id not in corridor):
                        candidate_light = light_id
                        break
                
                if candidate_light:
                    # Verificar si está en línea recta (simplificado)
                    if len(corridor) == 1 or await self._is_in_line(corridor[-2:] + [candidate_light]):
                        next_light = candidate_light
                        next_node = neighbor
                        break
            
            if next_light:
                corridor.append(next_light)
                visited_nodes.add(next_node)
                current_node = next_node
            else:
                break
        
        return corridor
    
    async def _is_in_line(self, light_sequence: List[str]) -> bool:
        """Verifica si una secuencia de semáforos está en línea recta"""
        if len(light_sequence) < 3:
            return True
        
        # Obtener coordenadas
        coords = []
        for light_id in light_sequence:
            traffic_light = self.traffic_lights[light_id]
            coords.append((traffic_light.data.latitude, traffic_light.data.longitude))
        
        # Verificar colinealidad (simplificado)
        # En una implementación real, usaríamos cálculos geométricos más precisos
        return True  # Simplificado por ahora
    
    async def _coordinate_green_waves(self):
        """Coordina ondas verdes en los corredores detectados"""
        for corridor in self.green_wave_corridors:
            try:
                await self._implement_green_wave(corridor)
            except Exception as e:
                self.logger.error(f"Error coordinando corridor {corridor}: {e}")
    
    async def _implement_green_wave(self, corridor: List[str]):
        """Implementa onda verde en un corridor específico"""
        if len(corridor) < 2:
            return
        
        # Calcular velocidad promedio del corridor (simplificado)
        average_speed = 40.0  # km/h por defecto
        
        # Calcular desfases óptimos
        reference_light = corridor[0]
        reference_agent = self.traffic_lights[reference_light]
        
        for i, light_id in enumerate(corridor[1:], 1):
            try:
                # Calcular distancia al semáforo de referencia
                distance = await self._calculate_distance(reference_light, light_id)
                
                # Calcular desfase temporal
                travel_time = (distance / 1000.0) / (average_speed / 3.6)  # segundos
                optimal_offset = travel_time % 60.0  # Ciclo de 60 segundos
                
                # Aplicar coordinación (simplificado)
                await self._apply_coordination_offset(light_id, optimal_offset)
                
            except Exception as e:
                self.logger.error(f"Error coordinando semáforo {light_id}: {e}")
    
    async def _calculate_distance(self, light1_id: str, light2_id: str) -> float:
        """Calcula distancia entre dos semáforos"""
        light1 = self.traffic_lights[light1_id]
        light2 = self.traffic_lights[light2_id]
        
        # Cálculo simplificado de distancia euclidiana
        lat_diff = light1.data.latitude - light2.data.latitude
        lon_diff = light1.data.longitude - light2.data.longitude
        
        # Convertir a metros (aproximado)
        distance = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111000  # 111 km por grado
        return distance
    
    async def _apply_coordination_offset(self, light_id: str, offset: float):
        """Aplica desfase de coordinación a un semáforo"""
        # En una implementación completa, esto ajustaría los tiempos del semáforo
        # Por ahora solo registramos la intención
        self.logger.debug(f"Aplicando offset de {offset:.1f}s a semáforo {light_id}")
    
    async def _balance_intersection_loads(self):
        """Balancea cargas entre intersecciones adyacentes"""
        for intersection_id, intersection in self.intersections.items():
            try:
                # Analizar carga de tráfico
                total_vehicles = intersection.get_total_vehicle_count()
                max_queue = intersection.get_max_queue_length()
                
                if max_queue > 15:  # Umbral de congestión
                    await self._redistribute_traffic_load(intersection_id)
                    
            except Exception as e:
                self.logger.error(f"Error balanceando intersección {intersection_id}: {e}")
    
    async def _redistribute_traffic_load(self, intersection_id: str):
        """Redistribuye carga de tráfico en intersección congestionada"""
        intersection = self.intersections[intersection_id]
        
        # Encontrar intersecciones adyacentes menos congestionadas
        adjacent_intersections = await self._find_adjacent_intersections(intersection_id)
        
        # Implementar redistribución (simplificado)
        for light_id in intersection.traffic_lights:
            if light_id in self.traffic_lights:
                # Extender ciclos verdes para facilitar flujo
                traffic_light = self.traffic_lights[light_id]
                # TODO: Implementar lógica de extensión de verde
                pass
    
    async def _find_adjacent_intersections(self, intersection_id: str) -> List[str]:
        """Encuentra intersecciones adyacentes"""
        adjacent = []
        
        try:
            intersection = self.intersections[intersection_id]
            center_node = intersection.node_id
            
            # Buscar intersecciones en nodos vecinos
            for neighbor in self.street_graph.neighbors(center_node):
                neighbor_intersection_id = f"intersection_{neighbor}"
                if neighbor_intersection_id in self.intersections:
                    adjacent.append(neighbor_intersection_id)
        
        except Exception as e:
            self.logger.error(f"Error buscando intersecciones adyacentes: {e}")
        
        return adjacent
    
    async def _synchronize_adjacent_lights(self):
        """Sincroniza semáforos adyacentes para mejor flujo"""
        for light_id, traffic_light in self.traffic_lights.items():
            try:
                node_id = traffic_light.data.node_id
                
                # Buscar semáforos en nodos adyacentes
                for neighbor_node in self.street_graph.neighbors(node_id):
                    neighbor_light = None
                    
                    for other_id, other_light in self.traffic_lights.items():
                        if other_light.data.node_id == neighbor_node:
                            neighbor_light = other_light
                            break
                    
                    if neighbor_light:
                        await self._synchronize_light_pair(traffic_light, neighbor_light)
                        
            except Exception as e:
                self.logger.error(f"Error sincronizando semáforo {light_id}: {e}")
    
    async def _synchronize_light_pair(self, light1: TrafficLightAgent, 
                                    light2: TrafficLightAgent):
        """Sincroniza un par de semáforos adyacentes"""
        # Implementar lógica de sincronización
        # Por ahora solo registramos la intención
        self.logger.debug(f"Sincronizando semáforos {light1.data.light_id} y {light2.data.light_id}")
    
    async def _optimize_network_performance(self):
        """Optimiza el rendimiento de toda la red de semáforos"""
        try:
            self.logger.info("Iniciando optimización de red de semáforos")
            
            # Recopilar métricas de todos los semáforos
            network_data = await self._collect_network_metrics()
            
            # Identificar puntos problemáticos
            problem_areas = await self._identify_problem_areas(network_data)
            
            # Aplicar optimizaciones
            for area in problem_areas:
                await self._apply_optimization(area)
            
            # Actualizar métricas globales
            await self._update_network_metrics(network_data)
            
            self.logger.info("Optimización de red completada")
            
        except Exception as e:
            self.logger.error(f"Error en optimización de red: {e}")
    
    async def _collect_network_metrics(self) -> Dict[str, Any]:
        """Recopila métricas de toda la red"""
        metrics = {
            "lights": {},
            "intersections": {},
            "global": {
                "total_lights": len(self.traffic_lights),
                "operational_lights": 0,
                "total_vehicles": 0,
                "average_efficiency": 0.0
            }
        }
        
        total_efficiency = 0.0
        operational_count = 0
        
        for light_id, traffic_light in self.traffic_lights.items():
            state = traffic_light.get_current_state()
            traffic_data = traffic_light.get_traffic_data()
            
            metrics["lights"][light_id] = {
                "state": state,
                "traffic": traffic_data
            }
            
            if state["is_operational"]:
                operational_count += 1
                total_efficiency += state["metrics"]["efficiency_score"]
            
            metrics["global"]["total_vehicles"] += traffic_data["nearby_vehicles"]
        
        metrics["global"]["operational_lights"] = operational_count
        if operational_count > 0:
            metrics["global"]["average_efficiency"] = total_efficiency / operational_count
        
        return metrics
    
    async def _identify_problem_areas(self, network_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica áreas problemáticas en la red"""
        problems = []
        
        # Identificar semáforos con baja eficiencia
        for light_id, data in network_data["lights"].items():
            efficiency = data["state"]["metrics"]["efficiency_score"]
            wait_time = data["state"]["metrics"]["average_wait_time"]
            
            if efficiency < 0.5 or wait_time > 90:
                problems.append({
                    "type": "low_efficiency",
                    "light_id": light_id,
                    "efficiency": efficiency,
                    "wait_time": wait_time
                })
        
        # Identificar intersecciones congestionadas
        for intersection_id, intersection in self.intersections.items():
            total_vehicles = intersection.get_total_vehicle_count()
            max_queue = intersection.get_max_queue_length()
            
            if total_vehicles > 20 or max_queue > 10:
                problems.append({
                    "type": "congestion",
                    "intersection_id": intersection_id,
                    "vehicle_count": total_vehicles,
                    "max_queue": max_queue
                })
        
        return problems
    
    async def _apply_optimization(self, problem: Dict[str, Any]):
        """Aplica optimización específica para un problema identificado"""
        problem_type = problem.get("type")
        
        if problem_type == "low_efficiency":
            await self._optimize_low_efficiency_light(problem)
        elif problem_type == "congestion":
            await self._optimize_congested_intersection(problem)
    
    async def _optimize_low_efficiency_light(self, problem: Dict[str, Any]):
        """Optimiza un semáforo con baja eficiencia"""
        light_id = problem["light_id"]
        
        if light_id in self.traffic_lights:
            traffic_light = self.traffic_lights[light_id]
            
            # Ajustar configuración adaptativa
            traffic_light.adaptation_enabled = True
            traffic_light.learning_rate = 0.2  # Aumentar velocidad de aprendizaje
            
            self.logger.info(f"Optimización aplicada al semáforo {light_id}")
    
    async def _optimize_congested_intersection(self, problem: Dict[str, Any]):
        """Optimiza una intersección congestionada"""
        intersection_id = problem["intersection_id"]
        
        if intersection_id in self.intersections:
            intersection = self.intersections[intersection_id]
            
            # Extender tiempos verdes para reducir colas
            for light_id in intersection.traffic_lights:
                if light_id in self.traffic_lights:
                    traffic_light = self.traffic_lights[light_id]
                    # TODO: Implementar extensión de verde inteligente
                    pass
            
            self.logger.info(f"Optimización aplicada a intersección {intersection_id}")
    
    async def _update_network_metrics(self, network_data: Dict[str, Any]):
        """Actualiza métricas globales de la red"""
        global_metrics = network_data["global"]
        
        self.network_metrics.update({
            "total_vehicles_processed": global_metrics["total_vehicles"],
            "network_efficiency": global_metrics["average_efficiency"],
            "active_lights": global_metrics["operational_lights"],
            "last_update": datetime.now().isoformat()
        })
    
    # Métodos públicos para control externo
    
    async def handle_emergency_vehicle(self, intersection_id: str, direction: str):
        """Maneja la presencia de un vehículo de emergencia"""
        try:
            if intersection_id in self.intersections:
                intersection = self.intersections[intersection_id]
                
                # Activar preempción en todos los semáforos de la intersección
                for light_id in intersection.traffic_lights:
                    if light_id in self.traffic_lights:
                        event = {
                            "type": "emergency_vehicle",
                            "direction": direction
                        }
                        self.traffic_lights[light_id].event_queue.append(event)
                
                # Registrar evento
                event = TrafficLightEvent(
                    event_id=f"emergency_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    event_type="emergency",
                    intersection_id=intersection_id,
                    details={"direction": direction},
                    severity=PriorityLevel.EMERGENCY
                )
                self.active_events.append(event)
                
                self.logger.warning(f"Vehículo de emergencia procesado en {intersection_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error manejando vehículo de emergencia: {e}")
        
        return False
    
    async def modify_traffic_light(self, light_id: str, new_phase: str = None, 
                                 duration: int = None) -> bool:
        """Modifica manualmente un semáforo"""
        try:
            if light_id not in self.traffic_lights:
                return False
            
            traffic_light = self.traffic_lights[light_id]
            
            if new_phase:
                await traffic_light.force_phase_change(new_phase, duration or 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error modificando semáforo {light_id}: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado general del sistema"""
        return {
            "is_running": self.is_running,
            "total_lights": len(self.traffic_lights),
            "operational_lights": sum(1 for tl in self.traffic_lights.values() 
                                    if tl.data.is_operational),
            "coordination_enabled": self.coordination_enabled,
            "optimization_enabled": self.optimization_enabled,
            "active_events": len(self.active_events),
            "green_corridors": len(self.green_wave_corridors),
            "network_metrics": self.network_metrics,
            "last_optimization": self.last_optimization.isoformat()
        }
    
    def get_light_states(self) -> Dict[str, Any]:
        """Obtiene estados de todos los semáforos"""
        states = {}
        
        for light_id, traffic_light in self.traffic_lights.items():
            states[light_id] = traffic_light.get_current_state()
        
        return states
    
    async def update_vehicle_data(self, vehicles_data: Dict[str, Any]):
        """Actualiza datos de vehículos para todos los semáforos"""
        for light_id, traffic_light in self.traffic_lights.items():
            # Filtrar vehículos cercanos a este semáforo
            nearby_vehicles = self._filter_nearby_vehicles(
                traffic_light, vehicles_data
            )
            traffic_light.update_vehicle_data(nearby_vehicles)
    
    def _filter_nearby_vehicles(self, traffic_light: TrafficLightAgent, 
                               vehicles_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filtra vehículos cercanos a un semáforo específico"""
        nearby = []
        max_distance = 0.002  # Aproximadamente 200 metros
        
        light_lat = traffic_light.data.latitude
        light_lon = traffic_light.data.longitude
        
        for vehicle_id, vehicle_data in vehicles_data.items():
            vehicle_lat = vehicle_data.get("lat", 0)
            vehicle_lon = vehicle_data.get("lon", 0)
            
            # Calcular distancia simple
            distance = ((light_lat - vehicle_lat) ** 2 + 
                       (light_lon - vehicle_lon) ** 2) ** 0.5
            
            if distance <= max_distance:
                nearby.append({
                    "id": vehicle_id,
                    "distance": distance * 111000,  # Convertir a metros aproximadamente
                    "speed": vehicle_data.get("speed", 0),
                    "direction": "north",  # Simplificado
                    "emergency": vehicle_data.get("emergency", False),
                    "waiting_time": 0  # Simplificado
                })
        
        return nearby
