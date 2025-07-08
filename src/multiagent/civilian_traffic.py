"""
Agente de Tráfico Civil para Simulación de Transporte
Simula el comportamiento de vehículos civiles en el entorno urbano
"""

import sys
import os
import random
import asyncio
import numpy as np
import math
import logging
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from .Civilian_enums import *

# Importar clases base del sistema multi-agente - SIMPLIFIED
# Communication manager disabled for standalone operation
from enum import Enum

# Importar environment para interactuar con el estado del entorno
sys.path.append("src/multiagent")
# from environment import WeatherCondition, RoadCondition
from Environment_enums import *

class CivilianTrafficAgent:
    """
    Agente que simula el comportamiento de vehículos civiles
    Versión standalone sin dependencias de comunicación
    """
    
    def __init__(self, vehicle_id: str, initial_position: Tuple[float, float],
                 initial_node: int, behavior: CivilianBehavior = CivilianBehavior.NORMAL):
        # Initialize agent properties
        self.agent_id = vehicle_id
        self.agent_type = "civilian_traffic"
        self.position = initial_position
        self.state = "active"  # Simple state instead of AgentState enum
        
        # Add logger for compatibility
        import logging
        self.logger = logging.getLogger(f"CivilianAgent_{vehicle_id}")
        
        # Add missing attributes for compatibility
        self.metrics = {
            "distance_traveled": 0.0,
            "total_travel_time": 0.0,
            "stops_count": 0,
            "decisions_made": 0
        }
        
        self.lat = initial_position[0]
        self.lon = initial_position[1]
        
        # Inicialización para seguimiento de movimiento
        self._last_position = initial_position
        self._movement_direction = random.uniform(0, 2 * 3.14159)  # Dirección inicial aleatoria
        
        # Características del vehículo civil
        self.behavior = behavior
        self.movement_state = MovementState.IDLE
        self.vehicle_type = random.choice(["car", "motorcycle", "van", "bus"])
        
        # Estado de navegación
        self.current_node = initial_node
        self.next_node = None
        self.target_node = None
        self.route = []
        self.progress = 0.0  # Progreso en la arista actual (0-1)
        
        # Parámetros de movimiento según comportamiento
        self.base_speed = self._calculate_base_speed()
        self.current_speed = 0.0
        self.max_speed = self._calculate_max_speed()
        self.acceleration = self._calculate_acceleration()
        self.deceleration = self._calculate_deceleration()
        
        # Parámetros físicos del vehículo
        self.fuel_level = random.uniform(20.0, 100.0)
        self.fuel_consumption_rate = self._calculate_fuel_consumption()
        self.capacity = self._calculate_capacity()
        self.current_load = random.randint(0, int(self.capacity * 0.3))
        
        # Percepción del entorno
        self.perceived_weather = WeatherCondition.CLEAR
        self.perceived_traffic_density = 0.0
        self.perceived_road_condition = RoadCondition.GOOD
        self.nearby_vehicles = []
        self.visible_traffic_lights = {}
        
        # Memoria y aprendizaje
        self.route_memory = {}  # Memoria de rutas anteriores
        self.traffic_memory = {}  # Memoria de congestión por arista
        self.weather_adaptations = {}  # Adaptaciones aprendidas por clima
        
        # Comportamiento social
        self.cooperation_level = self._calculate_cooperation_level()
        self.risk_tolerance = self._calculate_risk_tolerance()
        self.patience_level = self._calculate_patience_level()
        
        # Sistema de rastros para visualización
        self.trail = []
        self.max_trail_length = 20
        
        # Métricas específicas
        self.distance_traveled = 0.0
        self.total_travel_time = 0.0
        self.stops_count = 0
        self.route_changes = 0
        self.emergency_responses = 0
        self.traffic_violations = 0
        
        # Objetivo y destino
        self.has_destination = False
        self.destination_type = None  # work, home, shopping, recreation
        self.departure_time = None
        self.arrival_time = None
        self.trip_purpose = None
        
    def get_distance_to(self, position: Tuple[float, float]) -> float:
        """Calcula distancia euclidiana a una posición"""
        import math
        lat1, lon1 = self.lat, self.lon
        lat2, lon2 = position
        
        # Usar fórmula de Haversine para distancia más precisa
        R = 6371.0  # Radio de la Tierra en km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def _calculate_base_speed(self) -> float:
        """Calcula velocidad base según comportamiento"""
        speed_map = {
            CivilianBehavior.CONSERVATIVE: random.uniform(35, 45),
            CivilianBehavior.NORMAL: random.uniform(45, 55),
            CivilianBehavior.AGGRESSIVE: random.uniform(55, 70),
            CivilianBehavior.CAUTIOUS: random.uniform(30, 40),
            CivilianBehavior.RECKLESS: random.uniform(60, 80)
        }
        return speed_map[self.behavior]
    
    def _calculate_max_speed(self) -> float:
        """Calcula velocidad máxima según comportamiento y tipo de vehículo"""
        vehicle_max_speeds = {
            "car": 120,
            "motorcycle": 140,
            "van": 100,
            "bus": 80
        }
        
        base_max = vehicle_max_speeds[self.vehicle_type]
        
        if self.behavior == CivilianBehavior.RECKLESS:
            return base_max * 1.2
        elif self.behavior == CivilianBehavior.AGGRESSIVE:
            return base_max * 1.1
        elif self.behavior == CivilianBehavior.CAUTIOUS:
            return base_max * 0.8
        elif self.behavior == CivilianBehavior.CONSERVATIVE:
            return base_max * 0.9
        else:
            return base_max
    
    def _calculate_acceleration(self) -> float:
        """Calcula aceleración según comportamiento"""
        accel_map = {
            CivilianBehavior.CONSERVATIVE: random.uniform(1.5, 2.5),
            CivilianBehavior.NORMAL: random.uniform(2.0, 3.0),
            CivilianBehavior.AGGRESSIVE: random.uniform(3.0, 4.5),
            CivilianBehavior.CAUTIOUS: random.uniform(1.0, 2.0),
            CivilianBehavior.RECKLESS: random.uniform(4.0, 6.0)
        }
        return accel_map[self.behavior]
    
    def _calculate_deceleration(self) -> float:
        """Calcula desaceleración según comportamiento"""
        decel_map = {
            CivilianBehavior.CONSERVATIVE: random.uniform(3.0, 4.0),
            CivilianBehavior.NORMAL: random.uniform(3.5, 4.5),
            CivilianBehavior.AGGRESSIVE: random.uniform(4.0, 5.0),
            CivilianBehavior.CAUTIOUS: random.uniform(2.5, 3.5),
            CivilianBehavior.RECKLESS: random.uniform(5.0, 7.0)
        }
        return decel_map[self.behavior]
    
    async def _assign_random_destination(self):
        """Asigna un destino aleatorio al vehículo"""
        if hasattr(self, 'street_graph') and self.street_graph and len(list(self.street_graph.nodes)) > 0:
            # Obtener nodos disponibles
            available_nodes = list(self.street_graph.nodes)
            
            # Filtrar nodo actual si es posible
            if self.current_node in available_nodes and len(available_nodes) > 1:
                available_nodes.remove(self.current_node)
            
            # Seleccionar destino aleatorio
            if available_nodes:
                self.target_node = random.choice(available_nodes)
                self.has_destination = True
                
                # Intentar calcular ruta
                try:
                    if self.current_node != self.target_node:
                        path = self._find_path(self.current_node, self.target_node)
                        if path and len(path) > 1:
                            self.route = path[1:]  # Excluir nodo actual
                            self.next_node = self.route[0] if self.route else None
                            self.movement_state = MovementState.MOVING
                        else:
                            # Si no se puede calcular ruta, movimiento aleatorio
                            self._setup_random_movement()
                    else:
                        # Si está en el destino, buscar nuevo destino
                        self._setup_random_movement()
                except Exception as e:
                    # Fallback a movimiento aleatorio
                    self._setup_random_movement()
            else:
                self._setup_random_movement()
        else:
            # Si no hay grafo disponible, movimiento aleatorio
            self._setup_random_movement()
    
    def _setup_random_movement(self):
        """Configura movimiento aleatorio cuando no se puede calcular ruta"""
        self.has_destination = False
        self.target_node = None
        self.route = []
        self.next_node = None
        self.movement_state = MovementState.MOVING
        # El movimiento será manejado en _move_vehicle() con coordenadas directas
    
    def _find_path(self, start_node, target_node):
        """Encuentra la ruta más corta entre dos nodos"""
        try:
            import networkx as nx
            if hasattr(self, 'street_graph') and self.street_graph:
                return nx.shortest_path(self.street_graph, source=start_node, target=target_node, weight='weight')
            elif hasattr(self, '_street_graph') and self._street_graph:
                return nx.shortest_path(self._street_graph, source=start_node, target=target_node, weight='weight')
            else:
                return None
        except Exception as e:
            return None
    
    def _calculate_fuel_consumption(self) -> float:
        """Calcula tasa de consumo de combustible"""
        consumption_map = {
            "car": random.uniform(6, 12),  # L/100km
            "motorcycle": random.uniform(3, 6),
            "van": random.uniform(8, 15),
            "bus": random.uniform(20, 35)
        }
        base_consumption = consumption_map[self.vehicle_type]
        
        # Ajustar por comportamiento
        if self.behavior == CivilianBehavior.AGGRESSIVE:
            return base_consumption * 1.3
        elif self.behavior == CivilianBehavior.RECKLESS:
            return base_consumption * 1.5
        elif self.behavior == CivilianBehavior.CONSERVATIVE:
            return base_consumption * 0.8
        
        return base_consumption
    
    def _calculate_capacity(self) -> int:
        """Calcula capacidad del vehículo en kg"""
        capacity_map = {
            "car": random.randint(400, 600),
            "motorcycle": random.randint(150, 300),
            "van": random.randint(800, 1500),
            "bus": random.randint(2000, 5000)
        }
        return capacity_map[self.vehicle_type]
    
    def _calculate_cooperation_level(self) -> float:
        """Calcula nivel de cooperación (0-1)"""
        cooperation_map = {
            CivilianBehavior.CONSERVATIVE: random.uniform(0.7, 0.9),
            CivilianBehavior.NORMAL: random.uniform(0.5, 0.8),
            CivilianBehavior.AGGRESSIVE: random.uniform(0.2, 0.5),
            CivilianBehavior.CAUTIOUS: random.uniform(0.8, 1.0),
            CivilianBehavior.RECKLESS: random.uniform(0.0, 0.3)
        }
        return cooperation_map[self.behavior]
    
    def _calculate_risk_tolerance(self) -> float:
        """Calcula tolerancia al riesgo (0-1)"""
        risk_map = {
            CivilianBehavior.CONSERVATIVE: random.uniform(0.1, 0.3),
            CivilianBehavior.NORMAL: random.uniform(0.3, 0.6),
            CivilianBehavior.AGGRESSIVE: random.uniform(0.6, 0.8),
            CivilianBehavior.CAUTIOUS: random.uniform(0.0, 0.2),
            CivilianBehavior.RECKLESS: random.uniform(0.8, 1.0)
        }
        return risk_map[self.behavior]
    
    def _calculate_patience_level(self) -> float:
        """Calcula nivel de paciencia (0-1)"""
        patience_map = {
            CivilianBehavior.CONSERVATIVE: random.uniform(0.7, 0.9),
            CivilianBehavior.NORMAL: random.uniform(0.4, 0.7),
            CivilianBehavior.AGGRESSIVE: random.uniform(0.1, 0.4),
            CivilianBehavior.CAUTIOUS: random.uniform(0.8, 1.0),
            CivilianBehavior.RECKLESS: random.uniform(0.0, 0.2)
        }
        return patience_map[self.behavior]
    
    async def perceive(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Percibe el estado del entorno incluyendo tráfico, clima, otros vehículos
        """
        perception = {}
        
        try:
            # Obtener estado del clima
            weather_info = environment_state.get("weather", {})
            if weather_info:
                perception["weather_condition"]    = weather_info.get("condition", "despejado")
                perception["visibility"]           = weather_info.get("visibility", 10.0)
                perception["precipitation"]        = weather_info.get("precipitation", 0.0)
                perception["wind_speed"]           = weather_info.get("wind_speed", 0.0)
                perception["weather_risk_index"] = weather_info.get("risk_index", 0.0)
                print(f"[DEBUG-Agent] Percibido risk_index: {perception['weather_risk_index']:.2f}")
                
                # Actualizar percepción climática interna
                self._update_weather_perception(weather_info)
            
            # Percibir estado de la red vial
            road_network = environment_state.get("road_network", {})
            if road_network:
                congestion = road_network.get("congestion", {})
                traffic_lights = road_network.get("traffic_lights", {})
                road_conditions = road_network.get("road_conditions", {})
                
                perception["congestion_level"] = self._analyze_local_congestion(congestion)
                perception["traffic_lights"] = self._filter_visible_traffic_lights(traffic_lights)
                perception["road_conditions"] = self._analyze_local_road_conditions(road_conditions)
            
            # Percibir otros vehículos
            vehicles = environment_state.get("vehicles", {})
            if vehicles:
                perception["nearby_vehicles"] = self._detect_nearby_vehicles(vehicles)
                perception["local_traffic_density"] = self._calculate_local_traffic_density(vehicles)
            
            # Percibir eventos de tráfico
            traffic_events = environment_state.get("traffic_events", [])
            if traffic_events:
                perception["relevant_events"] = self._filter_relevant_events(traffic_events)
            
            # Percibir zonas especiales
            special_zones = environment_state.get("special_zones", {})
            if special_zones:
                perception["nearby_special_zones"] = self._detect_nearby_special_zones(special_zones)
            
            # Información temporal y contextual
            optimization_context = environment_state.get("optimization_context", {})
            perception["is_rush_hour"] = optimization_context.get("is_rush_hour", False)
            perception["emergency_active"] = optimization_context.get("emergency_active", False)
            
        except Exception as e:
            self.logger.error(f"Error en percepción del vehículo civil {self.agent_id}: {e}")
            perception = self._get_default_perception()
        
        return perception
    
    def _update_weather_perception(self, weather_info: Dict[str, Any]):
        """Actualiza la percepción interna del clima"""
        condition = weather_info.get("condition", "despejado")
        
        weather_mapping = {
            "despejado": WeatherCondition.CLEAR,
            "nublado": WeatherCondition.CLOUDY,
            "lluvia_ligera": WeatherCondition.LIGHT_RAIN,
            "lluvia_fuerte": WeatherCondition.HEAVY_RAIN,
            "tormenta": WeatherCondition.STORM,
            "niebla": WeatherCondition.FOG,
            "calor_extremo": WeatherCondition.EXTREME_HEAT
        }
        
        self.perceived_weather = weather_mapping.get(condition, WeatherCondition.CLEAR)
    
    def _analyze_local_congestion(self, congestion: Dict[str, float]) -> float:
        """Analiza la congestión local alrededor del vehículo"""
        if not congestion or not self.current_node:
            return 0.0
        
        local_congestion = 0.0
        checked_edges = 0
        
        # Buscar aristas que involucren el nodo actual
        for edge_key, congestion_level in congestion.items():
            if str(self.current_node) in str(edge_key):
                local_congestion += congestion_level
                checked_edges += 1
        
        return local_congestion / max(1, checked_edges)
    
    def _filter_visible_traffic_lights(self, traffic_lights: Dict[str, Any]) -> Dict[str, Any]:
        """Filtra semáforos visibles para el vehículo"""
        visible_lights = {}
        
        if not self.current_node:
            return visible_lights
        
        # Buscar semáforos en el nodo actual o nodos adyacentes
        for light_id, light_info in traffic_lights.items():
            try:
                light_node = int(light_id)
                # Si el semáforo está en el nodo actual o próximo
                if light_node == self.current_node or light_node == self.next_node:
                    visible_lights[light_id] = light_info
            except (ValueError, TypeError):
                continue
        
        self.visible_traffic_lights = visible_lights
        return visible_lights
    
    def _analyze_local_road_conditions(self, road_conditions: Dict[str, Any]) -> str:
        """Analiza las condiciones de las vías locales"""
        if not road_conditions or not self.current_node:
            return "good"
        
        # Buscar condiciones de aristas relacionadas con el nodo actual
        for edge_key, condition_info in road_conditions.items():
            if str(self.current_node) in edge_key:
                condition = condition_info.get("condition", "buena")
                return condition
        
        return "good"
    
    def _detect_nearby_vehicles(self, vehicles: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detecta vehículos cercanos"""
        nearby = []
        detection_radius = 0.01  # Radio de detección
        
        for vehicle_id, vehicle_info in vehicles.items():
            if vehicle_id == self.agent_id:
                continue
            
            vehicle_pos = (vehicle_info.get("lat", 0), vehicle_info.get("lon", 0))
            distance = self.get_distance_to(vehicle_pos)
            
            if distance <= detection_radius:
                nearby.append({
                    "id": vehicle_id,
                    "type": vehicle_info.get("type", "unknown"),
                    "position": vehicle_pos,
                    "distance": distance,
                    "speed": vehicle_info.get("speed", 0),
                    "emergency_priority": vehicle_info.get("emergency_priority", False)
                })
        
        self.nearby_vehicles = nearby
        return nearby
    
    def _calculate_local_traffic_density(self, vehicles: Dict[str, Any]) -> float:
        """Calcula la densidad de tráfico local"""
        local_area = 0.05  # Área de análisis local
        vehicles_in_area = 0
        
        for vehicle_id, vehicle_info in vehicles.items():
            if vehicle_id == self.agent_id:
                continue
            
            vehicle_pos = (vehicle_info.get("lat", 0), vehicle_info.get("lon", 0))
            distance = self.get_distance_to(vehicle_pos)
            
            if distance <= local_area:
                vehicles_in_area += 1
        
        # Densidad como vehículos por unidad de área
        area = np.pi * (local_area ** 2)
        return vehicles_in_area / area
    
    def _filter_relevant_events(self, traffic_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filtra eventos de tráfico relevantes para el vehículo"""
        relevant = []
        max_distance = 0.1  # Distancia máxima para considerar relevante
        
        for event in traffic_events:
            # Analizar proximidad y relevancia del evento
            event_severity = event.get("severity", 1)
            event_type = event.get("type", "unknown")
            
            # Eventos de alta severidad son siempre relevantes
            if event_severity >= 4:
                relevant.append(event)
            elif event_type in ["accidente", "cierre_vial", "emergencia"]:
                relevant.append(event)
        
        return relevant
    
    def _detect_nearby_special_zones(self, special_zones: Dict[str, List]) -> Dict[str, List]:
        """Detecta zonas especiales cercanas"""
        nearby_zones = {}
        detection_radius = 0.05
        
        for zone_type, zones in special_zones.items():
            nearby_zones[zone_type] = []
            for zone in zones:
                # Simplificación: asumir que las zonas tienen coordenadas
                if isinstance(zone, dict) and "lat" in zone and "lon" in zone:
                    zone_pos = (zone["lat"], zone["lon"])
                    distance = self.get_distance_to(zone_pos)
                    if distance <= detection_radius:
                        nearby_zones[zone_type].append(zone)
        
        return nearby_zones
    
    def _get_default_perception(self) -> Dict[str, Any]:
        """Retorna percepción por defecto en caso de error"""
        return {
            "weather_condition": "despejado",
            "congestion_level": 0.0,
            "traffic_lights": {},
            "road_conditions": "good",
            "nearby_vehicles": [],
            "local_traffic_density": 0.0,
            "relevant_events": [],
            "is_rush_hour": False,
            "emergency_active": False
        }
    
    async def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toma decisiones basadas en la percepción del entorno y comportamiento del vehículo
        """
        decision = {}
        
        try:
            # Decisión de velocidad objetivo
            target_speed = self._decide_target_speed(perception)
            decision["target_speed"] = target_speed
            
            # Decisión de ruta
            route_decision = self._decide_route_action(perception)
            decision.update(route_decision)
            
            # Decisión de comportamiento ante semáforos
            traffic_light_decision = self._decide_traffic_light_behavior(perception)
            decision.update(traffic_light_decision)
            
            # Decisión de comportamiento ante emergencias
            emergency_decision = self._decide_emergency_response(perception)
            decision.update(emergency_decision)
            
            # Decisión de cooperación con otros vehículos
            cooperation_decision = self._decide_cooperation_behavior(perception)
            decision.update(cooperation_decision)
            
            # Decisión de navegación
            navigation_decision = self._decide_navigation(perception)
            decision.update(navigation_decision)
            
            # Actualizar métricas de decisión
            self.metrics["decisions_made"] += 1
            
        except Exception as e:
            self.logger.error(f"Error en decisión del vehículo civil {self.agent_id}: {e}")
            decision = self._get_default_decision()
        
        return decision
    
    def _decide_target_speed(self, perception: Dict[str, Any]) -> float:
        """Decide la velocidad objetivo basada en las condiciones"""
        target_speed = self.base_speed
        
        # Ajuste por índice de riesgo fuzzy
        risk = perception.get("weather_risk_index", 0.0)
        print(f"[DEBUG-Agent] Riesgo climático: {risk:.2f}/10")

        # Factor lineal: riesgo 0 → factor=1.0 ; riesgo 10 → factor=0.5
        factor = 1.0 - 0.5 * (risk / 10.0)
        print(f"[DEBUG-Agent] Factor de reducción de velocidad: {factor:.2f}")

        target_speed *= factor
        
        # Ajustar por congestión
        congestion_level = perception.get("congestion_level", 0.0)
        if congestion_level > 2.0:
            target_speed *= 0.6
        elif congestion_level > 1.5:
            target_speed *= 0.8
        
        # Ajustar por densidad de tráfico local
        traffic_density = perception.get("local_traffic_density", 0.0)
        if traffic_density > 10:
            target_speed *= 0.7
        elif traffic_density > 5:
            target_speed *= 0.85
        
        # Ajustar por condiciones de la vía
        road_conditions = perception.get("road_conditions", "good")
        if road_conditions in ["mala", "cerrada"]:
            target_speed *= 0.5
        elif road_conditions == "regular":
            target_speed *= 0.8
        
        # Ajustar por comportamiento personal
        if self.behavior == CivilianBehavior.CAUTIOUS:
            target_speed *= 0.9
        elif self.behavior == CivilianBehavior.AGGRESSIVE:
            target_speed *= 1.1
        elif self.behavior == CivilianBehavior.RECKLESS:
            target_speed *= 1.2
        
        return max(10, min(target_speed, self.max_speed))
    
    def _decide_route_action(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Decide acciones relacionadas con la ruta"""
        decision = {"change_route": False, "find_alternative": False}
        
        # Verificar eventos relevantes que requieran cambio de ruta
        relevant_events = perception.get("relevant_events", [])
        for event in relevant_events:
            if event.get("type") in ["cierre_vial", "accidente"]:
                decision["change_route"] = True
                decision["reason"] = f"Evento: {event.get('type')}"
                break
        
        # Verificar congestión alta
        congestion_level = perception.get("congestion_level", 0.0)
        if congestion_level > 3.0 and self.patience_level < 0.5:
            decision["find_alternative"] = True
            decision["reason"] = "Congestión alta"
        
        return decision
    
    def _decide_traffic_light_behavior(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Decide comportamiento ante semáforos"""
        decision = {"respect_traffic_light": True, "stop_at_yellow": True}
        
        traffic_lights = perception.get("traffic_lights", {})
        
        for light_id, light_info in traffic_lights.items():
            light_state = light_info.get("state", "green")
            
            if light_state == "yellow":
                # Decisión de parar o acelerar en amarillo
                if self.behavior == CivilianBehavior.RECKLESS:
                    decision["stop_at_yellow"] = False
                elif self.behavior == CivilianBehavior.AGGRESSIVE and self.risk_tolerance > 0.7:
                    decision["stop_at_yellow"] = False
                else:
                    decision["stop_at_yellow"] = True
            
            elif light_state == "red":
                # Respeto al semáforo en rojo
                if self.behavior == CivilianBehavior.RECKLESS and random.random() < 0.1:
                    decision["respect_traffic_light"] = False
                    self.traffic_violations += 1
                else:
                    decision["respect_traffic_light"] = True
        
        return decision
    
    def _decide_emergency_response(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Decide respuesta ante situaciones de emergencia"""
        decision = {"emergency_stop": False, "yield_way": False, "slow_down": False}
        
        # Verificar vehículos de emergencia cercanos
        nearby_vehicles = perception.get("nearby_vehicles", [])
        for vehicle in nearby_vehicles:
            if vehicle.get("emergency_priority", False):
                decision["yield_way"] = True
                decision["slow_down"] = True
                self.emergency_responses += 1
                break
        
        # Verificar emergencias activas
        if perception.get("emergency_active", False):
            decision["slow_down"] = True
        
        # Verificar eventos críticos
        relevant_events = perception.get("relevant_events", [])
        for event in relevant_events:
            if event.get("type") == "emergencia" and event.get("severity", 1) >= 4:
                decision["emergency_stop"] = True
                break
        
        return decision
    
    def _decide_cooperation_behavior(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Decide comportamiento cooperativo con otros vehículos"""
        decision = {"allow_merge": False, "maintain_distance": True, "signal_intentions": True}
        
        # Decisión basada en nivel de cooperación
        if self.cooperation_level > 0.7:
            decision["allow_merge"] = True
            decision["maintain_distance"] = True
        elif self.cooperation_level < 0.3:
            decision["maintain_distance"] = False
            decision["signal_intentions"] = False
        
        # Ajustar por densidad de tráfico
        traffic_density = perception.get("local_traffic_density", 0.0)
        if traffic_density > 8 and self.cooperation_level > 0.5:
            decision["allow_merge"] = True
        
        return decision
    
    def _decide_navigation(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Decide acciones de navegación"""
        decision = {"continue_route": True, "request_new_destination": False}
        
        # Verificar si necesita nuevo destino
        if not self.has_destination or not self.route or self.next_node is None:
            decision["request_new_destination"] = True
            decision["continue_route"] = False
        
        # Verificar si llegó al destino
        if self.current_node == self.target_node:
            decision["continue_route"] = False
            decision["destination_reached"] = True
            decision["request_new_destination"] = True
        
        # Asegurar velocidad mínima si tiene destino
        if self.has_destination and self.current_speed < 5.0:
            self.current_speed = max(self.base_speed * 0.5, 5.0)
        
        return decision
    
    def _get_default_decision(self) -> Dict[str, Any]:
        """Retorna decisión por defecto en caso de error"""
        return {
            "target_speed": self.base_speed * 0.8,
            "change_route": False,
            "respect_traffic_light": True,
            "slow_down": False,
            "continue_route": True
        }
    
    async def act(self, decision: Dict[str, Any]) -> bool:
        """
        Ejecuta las acciones determinadas por las decisiones
        """
        try:
            old_position = (self.lat, self.lon)
            
            # Actualizar velocidad
            target_speed = decision.get("target_speed", self.current_speed)
            await self._update_speed(target_speed)
            
            # Ejecutar acciones de navegación
            if decision.get("change_route", False):
                await self._request_route_change(decision.get("reason", ""))
            
            if decision.get("request_new_destination", False):
                await self._request_new_destination()
            
            # Ejecutar respuestas de emergencia
            if decision.get("emergency_stop", False):
                await self._execute_emergency_stop()
            elif decision.get("yield_way", False):
                await self._yield_way()
            elif decision.get("slow_down", False):
                await self._slow_down()
            
            # Mover el vehículo si está en movimiento
            if self.movement_state == MovementState.MOVING and self.current_speed > 0:
                await self._move_vehicle()
                
                # Debug específico para el primer vehículo
                if self.agent_id == 'vehicle0':
                    new_position = (self.lat, self.lon)
                    position_changed = abs(old_position[0] - new_position[0]) > 0.000001 or abs(old_position[1] - new_position[1]) > 0.000001
                    if position_changed:
                        # Logs detallados comentados para mejorar rendimiento
                        # print(f"🏃‍♂️ {self.agent_id}: MOVIDO de ({old_position[0]:.6f}, {old_position[1]:.6f}) a ({new_position[0]:.6f}, {new_position[1]:.6f})")
                        # print(f"   Estado: {self.movement_state.value}, velocidad: {self.current_speed:.1f} km/h, progreso: {self.progress:.3f}")
                        # print(f"   Ruta: {self.current_node} -> {self.next_node}, destino: {self.target_node}")
                        pass
            
            # Comunicar estado si hay cambios significativos
            if decision.get("signal_intentions", True):
                await self._communicate_intentions(decision)
            
            # Actualizar métricas
            self._update_movement_metrics()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error ejecutando acción en vehículo civil {self.agent_id}: {e}")
            return False
    
    async def _update_speed(self, target_speed: float):
        """Actualiza la velocidad del vehículo gradualmente"""
        speed_diff = target_speed - self.current_speed
        
        if speed_diff > 0:  # Acelerar
            speed_change = min(self.acceleration, speed_diff)
            self.current_speed += speed_change
        elif speed_diff < 0:  # Desacelerar
            speed_change = min(self.deceleration, abs(speed_diff))
            self.current_speed -= speed_change
        
        # Limitar velocidad máxima
        self.current_speed = max(0, min(self.current_speed, self.max_speed))
        
        # Actualizar estado de movimiento
        if self.current_speed > 0:
            self.movement_state = MovementState.MOVING
        else:
            self.movement_state = MovementState.IDLE
    
    async def _request_route_change(self, reason: str):
        """Solicita cambio de ruta al sistema"""
        # Communication manager disabled - route change request skipped
        # Log comentado para mejorar rendimiento
        # print(f"🔄 Vehicle {self.agent_id} would request route change: {reason}")
        self.route_changes += 1
    
    async def _request_new_destination(self):
        """Solicita un nuevo destino para el vehículo"""
        # Usar asignación directa en lugar de comunicación (más eficiente)
        await self._assign_random_destination()
        
        # Communication manager disabled - destination notification skipped
        # Log comentado para mejorar rendimiento
        # print(f"🎯 Vehicle {self.agent_id} assigned new destination: node {self.target_node}")
    
    async def _execute_emergency_stop(self):
        """Ejecuta parada de emergencia"""
        self.current_speed = 0.0
        self.movement_state = MovementState.EMERGENCY_STOP
        
        # Communication manager disabled - emergency notification skipped
        # Log de emergencia comentado (mantener solo si es crítico)
        # print(f"🚨 Vehicle {self.agent_id} emergency stop at ({self.lat:.6f}, {self.lon:.6f})")
    
    async def _yield_way(self):
        """Cede el paso a vehículos prioritarios"""
        self.current_speed *= 0.5
        self.movement_state = MovementState.WAITING
        
        # Communication manager disabled - yield notification skipped
        # Log comentado para mejorar rendimiento
        # print(f"⚠️ Vehicle {self.agent_id} yielding way at ({self.lat:.6f}, {self.lon:.6f})")
    
    async def _slow_down(self):
        """Reduce la velocidad gradualmente"""
        self.current_speed *= 0.7
        
    async def _move_vehicle(self):
        """Mueve el vehículo a lo largo de su ruta"""
        # Si no tiene ruta, asignar una nueva automáticamente
        if not self.route or self.next_node is None:
            await self._assign_random_destination()
            
            # Si aún no tiene ruta después del intento, hacer movimiento básico
            if not self.route or self.next_node is None:
                if self.current_speed > 0:
                    # Movimiento básico sin ruta específica - movimiento más visible
                    import math
                    direction = random.uniform(0, 2 * math.pi)
                    distance = (self.current_speed / 3600.0) * 0.1  # Conversión km/h a coordenadas por segundo
                    
                    self.lat += distance * math.cos(direction) * 0.01  # Factor de escala para visibilidad
                    self.lon += distance * math.sin(direction) * 0.01
                    self.position = (self.lat, self.lon)
                    self.update_position(self.position)
                return
        
        # Calcular progreso en la arista actual con factor de tiempo real
        time_factor = 0.1  # Simulamos pasos de 100ms
        speed_ms = self.current_speed / 3.6  # Convertir km/h a m/s
        
        # Obtener distancia de la arista actual
        edge_distance = self._get_current_edge_distance()
        if edge_distance > 0:
            # Progreso basado en velocidad real y tiempo
            progress_increment = (speed_ms * time_factor) / (edge_distance * 1000)  # km a m
            self.progress += progress_increment
        else:
            # Fallback si no se puede calcular distancia
            self.progress += self.current_speed * 0.01
        
        # Verificar si llegó al siguiente nodo
        if self.progress >= 1.0:
            await self._advance_to_next_node()
        
        # Actualizar posición interpolada
        self._update_interpolated_position()
        
        # Consumir combustible
        self._consume_fuel()
    
    def _get_current_edge_distance(self) -> float:
        """Calcula la distancia de la arista actual en km"""
        if not self.next_node or not hasattr(self, 'street_graph'):
            return 1.0  # Distancia por defecto
        
        try:
            if self.street_graph and self.street_graph.has_edge(self.current_node, self.next_node):
                edge_data = self.street_graph[self.current_node][self.next_node]
                # Buscar la distancia en los datos de la arista
                if isinstance(edge_data, dict):
                    for key, data in edge_data.items():
                        if 'weight' in data:
                            return data['weight']
                        elif 'length' in data:
                            return data['length']
                elif 'weight' in edge_data:
                    return edge_data['weight']
                elif 'length' in edge_data:
                    return edge_data['length']
            
            # Si no hay datos de distancia, calcular usando coordenadas
            current_data = self.street_graph.nodes.get(self.current_node, {})
            next_data = self.street_graph.nodes.get(self.next_node, {})
            
            if 'lat' in current_data and 'lat' in next_data:
                return self.get_distance_to((next_data['lat'], next_data['lon']))
        except Exception as e:
            pass
        
        return 0.5  # Distancia por defecto en km
    
    async def _advance_to_next_node(self):
        """Avanza al siguiente nodo en la ruta"""
        if self.next_node is None:
            return
        
        # Actualizar nodo actual
        self.current_node = self.next_node
        self.progress = 0.0
        
        # Encontrar siguiente nodo en la ruta
        if self.route:
            try:
                current_index = self.route.index(self.current_node)
                if current_index < len(self.route) - 1:
                    self.next_node = self.route[current_index + 1]
                else:
                    # Llegó al destino
                    self.next_node = None
                    self.target_node = None
                    self.has_destination = False
                    self.movement_state = MovementState.IDLE
                    await self._handle_destination_reached()
                    
                    # NUEVO: Asignar nuevo destino inmediatamente
                    await self._request_new_destination()
            except ValueError:
                # Error en la ruta, solicitar nueva
                await self._request_route_change("Route error")
    
    def _update_interpolated_position(self):
        """Actualiza la posición interpolada entre nodos usando coordenadas reales"""
        if self.next_node is None or self.progress <= 0:
            return
        
        try:
            # Obtener las coordenadas de los nodos desde el grafo del entorno
            if hasattr(self, 'street_graph') and self.street_graph:
                # Obtener coordenadas reales de los nodos
                current_node_data = self.street_graph.nodes[self.current_node]
                next_node_data = self.street_graph.nodes[self.next_node]
                
                current_lat = current_node_data.get('lat', self.lat)
                current_lon = current_node_data.get('lon', self.lon)
                next_lat = next_node_data.get('lat', self.lat)
                next_lon = next_node_data.get('lon', self.lon)
                
                # Interpolación suave con clamp del progreso
                progress_clamped = max(0.0, min(1.0, self.progress))
                
                # Interpolación lineal entre nodos
                interpolated_lat = current_lat + (next_lat - current_lat) * progress_clamped
                interpolated_lon = current_lon + (next_lon - current_lon) * progress_clamped
                
                # Actualizar posición
                old_position = (self.lat, self.lon)
                self.lat = interpolated_lat
                self.lon = interpolated_lon
                self.position = (self.lat, self.lon)
                self.update_position(self.position)
                
                # Debug comentado para mejorar rendimiento
                # if self.agent_id == 'vehicle0' and abs(old_position[0] - self.lat) > 0.000001:
                #     print(f"🚗 {self.agent_id}: Movido de ({old_position[0]:.6f}, {old_position[1]:.6f}) a ({self.lat:.6f}, {self.lon:.6f}), progreso: {progress_clamped:.3f}")
                
            elif hasattr(self, '_street_graph') and self._street_graph:
                # Usar grafo alternativo si está disponible
                current_node_data = self._street_graph.nodes[self.current_node]
                next_node_data = self._street_graph.nodes[self.next_node]
                
                current_lat = current_node_data.get('lat', self.lat)
                current_lon = current_node_data.get('lon', self.lon)
                next_lat = next_node_data.get('lat', self.lat)
                next_lon = next_node_data.get('lon', self.lon)
                
                progress_clamped = max(0.0, min(1.0, self.progress))
                
                interpolated_lat = current_lat + (next_lat - current_lat) * progress_clamped
                interpolated_lon = current_lon + (next_lon - current_lon) * progress_clamped
                
                self.lat = interpolated_lat
                self.lon = interpolated_lon
                self.position = (self.lat, self.lon)
                self.update_position(self.position)
            else:
                # Fallback: movimiento simulado con dirección consistente
                import math
                if not hasattr(self, '_movement_direction'):
                    self._movement_direction = random.uniform(0, 2 * math.pi)
                
                # Cambiar dirección ocasionalmente
                if random.random() < 0.05:  # 5% de probabilidad de cambio de dirección
                    self._movement_direction += random.uniform(-0.5, 0.5)
                
                distance = (self.current_speed / 3600.0) * 0.1 * 0.01  # Movimiento más realista
                
                self.lat += distance * math.cos(self._movement_direction)
                self.lon += distance * math.sin(self._movement_direction)
                self.position = (self.lat, self.lon)
                self.update_position(self.position)
                
        except Exception as e:
            # Si hay error, hacer movimiento simulado básico pero consistente
            import math
            if not hasattr(self, '_movement_direction'):
                self._movement_direction = random.uniform(0, 2 * math.pi)
            
            distance = (self.current_speed / 3600.0) * 0.1 * 0.01
            
            self.lat += distance * math.cos(self._movement_direction)
            self.lon += distance * math.sin(self._movement_direction)
            self.position = (self.lat, self.lon)
            self.update_position(self.position)
    
    def _consume_fuel(self):
        """Consume combustible basado en la velocidad y distancia"""
        if self.current_speed > 0:
            # Consumo simplificado basado en velocidad
            consumption = (self.fuel_consumption_rate / 100) * (self.current_speed / 50) * 0.01
            self.fuel_level = max(0, self.fuel_level - consumption)
    
    def _update_trail(self):
        """Actualiza el rastro del vehículo para visualización"""
        if self.position:
            self.trail.append([self.lon, self.lat])  # deck.gl expects [lon, lat]
            
            # Limitar longitud del rastro
            if len(self.trail) > self.max_trail_length:
                self.trail.pop(0)
    
    def update_position(self, position):
        """Actualiza la posición del agente"""
        self.position = position
        if position and len(position) >= 2:
            self.lat = position[0]
            self.lon = position[1]
            
            # Actualizar rastro para visualización
            self._update_trail()
            
            # Actualizar distancia recorrida
            if hasattr(self, '_last_position') and self._last_position:
                distance = self.get_distance_to(self._last_position)
                self.distance_traveled += distance
            
            self._last_position = position
    
    async def _handle_destination_reached(self):
        """Maneja la llegada al destino"""
        self.arrival_time = datetime.now()
        
        if self.departure_time:
            travel_time = (self.arrival_time - self.departure_time).total_seconds()
            self.total_travel_time += travel_time
        
        # Communication manager disabled - arrival notification skipped
        # Log comentado para mejorar rendimiento
        # print(f"🎯 Vehicle {self.agent_id} reached destination at ({self.lat:.6f}, {self.lon:.6f})")
    
    async def _communicate_intentions(self, decision: Dict[str, Any]):
        """Comunica intenciones a otros agentes"""
        # Communication manager disabled - intention communication skipped
        # print(f"💭 Vehicle {self.agent_id} intentions: speed={self.current_speed:.1f}, state={self.movement_state.value}")
    
    def _update_movement_metrics(self):
        """Actualiza métricas de movimiento"""
        if self.current_speed > 0:
            # Estimar distancia recorrida (simplificado)
            distance_increment = self.current_speed * 0.001  # km
            self.distance_traveled += distance_increment
    
    def assign_route_and_destination(self, route: List[int], target_node: int, 
                                   destination_type: str = "general"):
        """Asigna nueva ruta y destino al vehículo"""
        self.route = route
        self.target_node = target_node
        self.destination_type = destination_type
        self.has_destination = True
        self.departure_time = datetime.now()
        self.progress = 0.0
        
        # Configurar siguiente nodo
        if len(route) > 1:
            self.next_node = route[1]  # El primer nodo debería ser el actual
        else:
            self.next_node = target_node
        
        self.movement_state = MovementState.MOVING
    
    def get_vehicle_status(self) -> Dict[str, Any]:
        """Retorna estado completo del vehículo civil"""
        return {
            "vehicle_id": self.agent_id,
            "vehicle_type": self.vehicle_type,
            "behavior": self.behavior.value,
            "movement_state": self.movement_state.value,
            "position": self.position,
            "current_node": self.current_node,
            "next_node": self.next_node,
            "target_node": self.target_node,
            "current_speed": self.current_speed,
            "progress": self.progress,
            "fuel_level": self.fuel_level,
            "has_destination": self.has_destination,
            "destination_type": self.destination_type,
            "cooperation_level": self.cooperation_level,
            "risk_tolerance": self.risk_tolerance,
            "patience_level": self.patience_level,
            "metrics": {
                "distance_traveled": self.distance_traveled,
                "total_travel_time": self.total_travel_time,
                "stops_count": self.stops_count,
                "route_changes": self.route_changes,
                "emergency_responses": self.emergency_responses,
                "traffic_violations": self.traffic_violations
            }
        }
    
    async def handle_emergency(self, message):
        """Maneja mensajes de emergencia específicos"""
        emergency_data = message.content
        emergency_type = emergency_data.get("emergency_type", "unknown")
        
        self.logger.warning(f"Emergencia recibida por vehículo civil {self.agent_id}: {emergency_type}")
        
        if emergency_type == "traffic_jam":
            self.current_speed = min(self.current_speed, self.base_speed * 0.3)
        elif emergency_type == "accident":
            self.current_speed = 0.0
            self.movement_state = MovementState.EMERGENCY_STOP
        elif emergency_type == "weather_alert":
            # Ajustar comportamiento por alerta climática
            self.current_speed *= 0.7
        
        self.emergency_responses += 1
    
    async def handle_traffic_update(self, message):
        """Maneja actualizaciones de tráfico"""
        traffic_data = message.content
        
        # Actualizar memoria de tráfico
        if "congestion_update" in traffic_data:
            congestion_info = traffic_data["congestion_update"]
            for edge, level in congestion_info.items():
                self.traffic_memory[edge] = level
        
        # Responder a cambios en semáforos
        if "traffic_light_update" in traffic_data:
            light_update = traffic_data["traffic_light_update"]
            light_id = light_update.get("intersection_id")
            new_state = light_update.get("current_state")
            
            if light_id in self.visible_traffic_lights:
                self.visible_traffic_lights[light_id]["state"] = new_state
    
    async def next_step(self, environment_state: Dict[str, Any]):
        perception = await self.perceive(environment_state)
        decision = await self.decide(perception)
        if await self.act(decision):
            # Debug comentado para mejorar rendimiento
            # if(self.agent_id == 'vehicle0'):
            #     print(f"🚗 {self.agent_id}: pos=({self.lat:.6f}, {self.lon:.6f}), speed={self.current_speed:.1f}, progress={self.progress:.2f}")
            #     print(f"   Estado: {self.movement_state.value}, nodo actual: {self.current_node}, siguiente: {self.next_node}")
            pass
        else:
            # Solo mantener logs de errores críticos
            pass  # Comentado también para evitar spam: print(f"❌ No se pudo ejecutar acción para {self.agent_id}")
  
        

    
    def __str__(self) -> str:
        return (f"CivilianTrafficAgent(id={self.agent_id}, "
                f"type={self.vehicle_type}, behavior={self.behavior.value}, "
                f"state={self.movement_state.value}, speed={self.current_speed:.1f}), "
                f"current_node={self.current_node}, "
                f"next_node={self.next_node}, ")
