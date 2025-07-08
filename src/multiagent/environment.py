import sys
import os
import time
import random
import asyncio
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .civilian_traffic import CivilianTrafficAgent
from .Civilian_enums import *
from .Environment_enums import *

# Imports
sys.path.append("src")
sys.path.append("src/weather")
sys.path.append("src/traffic_events")
sys.path.append("src/crawler")
sys.path.append("src/multi_agent")

# BDI System imports
from .delivery_truck_bdi import DeliveryTruckBDI
from .communication_system import communication_manager

# Import communication manager - DISABLED
try:
    # from src.multi_agent.communication import communication_manager
    COMMUNICATION_AVAILABLE = True  # Habilitar comunicación BDI
    print("Communication manager BDI enabled")
except ImportError as e:
    print(f"Advertencia: Communication manager no disponible: {e}")
    COMMUNICATION_AVAILABLE = False

try:
    from weather.weather_impact_analyzer import WeatherImpactAnalyzer
    from traffic_events.traffic_events_analyzer import apply_traffic_weights
    from crawler.traffic_events_crawler import TrafficCrawler
    WEATHER_AVAILABLE = True
except ImportError as e:
    print(f"Advertencia: Módulos de clima/tráfico no disponibles: {e}")
    WEATHER_AVAILABLE = False



@dataclass
class WeatherState:
    """Estado del clima"""
    condition: WeatherCondition = WeatherCondition.CLEAR
    temperature: float = 25.0  # Celsius
    humidity: float = 60.0  # Porcentaje
    wind_speed: float = 10.0  # km/h
    precipitation: float = 0.0  # mm/h
    visibility: float = 10.0  # km
    pressure: float = 1013.25  # hPa
    # uv_index: float = 5.0 
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrafficEvent:
    """Evento de tráfico"""
    event_id: str
    event_type: TrafficEventType
    location: Tuple[float, float]  # (lat, lon)
    affected_streets: List[str]
    severity: int  # 1-5 (1=menor, 5=crítico)
    start_time: datetime
    estimated_duration: timedelta
    description: str
    impact_factor: float = 1.0  # Multiplicador de retraso
    is_active: bool = True


@dataclass
class VehicleState:
    """Estado de un vehículo"""
    vehicle_id: str
    vehicle_type: str = "delivery_truck"  # truck, van, car, motorcycle
    current_node: int = 0
    next_node: Optional[int] = None
    lat: float = 0.0
    lon: float = 0.0
    speed: float = 0.0  # km/h
    capacity: int = 100  # kg
    current_load: int = 0  # kg
    fuel_level: float = 100.0  # %
    driver_type: str = "normal"  # normal, aggressive, cautious
    route: List[int] = field(default_factory=list)
    progress: float = 0.0  # Progreso en arista actual (0-1)
    last_update: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    emergency_priority: bool = False
    maintenance_needed: bool = False


@dataclass
class RoadSegment:
    """Segmento de carretera"""
    edge_id: Tuple[int, int]
    road_type: str = "residential"  # motorway, trunk, primary, secondary, tertiary, residential
    condition: RoadCondition = RoadCondition.GOOD
    max_speed: float = 50.0  # km/h
    min_speed: float = 20.0  # km/h
    capacity: int = 10  # Número máximo de vehículos simultáneos
    current_vehicles: int = 0
    weather_factor: float = 1.0  # Multiplicador por clima
    traffic_factor: float = 1.0  # Multiplicador por tráfico
    construction_factor: float = 1.0  # Multiplicador por construcción
    slope: float = 0.0  # Pendiente en grados
    width: float = 3.5  # Ancho en metros
    surface_type: str = "asphalt"  # asphalt, concrete, gravel, dirt
    lighting: bool = True
    toll: bool = False
    toll_cost: float = 0.0


@dataclass
class TrafficLight:
    """Semáforo"""
    node_id: int
    state: str = "green"  # green, yellow, red
    green_duration: float = 30.0  # segundos
    yellow_duration: float = 5.0  # segundos
    red_duration: float = 25.0  # segundos
    last_change: datetime = field(default_factory=datetime.now)
    cycle_time: float = 60.0  # segundos
    is_adaptive: bool = False  # Se adapta al tráfico
    priority_override: bool = False  # Para vehículos de emergencia


class Environment:

    def __init__(self, street_graph: nx.Graph, num_vehicles:int = 20):
        """
        Args:
            street_graph: Grafo de NetworkX con la red de calles
            config: Configuración adicional del entorno
        """
        self.street_graph = street_graph
        self.num_vehicles = num_vehicles
        
        # Estados principales del entorno
        self.current_time = datetime.now()
        self.simulation_start_time = datetime.now()
        self.time_step = 1.0  # segundos por paso de simulación
        self.simulation_speed = 1.0  # Multiplicador de velocidad
        
        # Estado del clima
        self.weather_state = WeatherState()
        self.weather_forecast = []  # Pronóstico futuro
        
        # Vehículos en el sistema
        self.vehicles: Dict[str, CivilianTrafficAgent] = {}
        self.delivery_trucks: Dict[str, VehicleState] = {}
        # self.civilian_traffic: Dict[str, VehicleState] = {}
        
        # BDI Delivery Trucks - NUEVO SISTEMA
        self.bdi_delivery_trucks: Dict[str, DeliveryTruckBDI] = {}
        
        # Estado de la red vial
        self.road_segments: Dict[Tuple[int, int], RoadSegment] = {}
        self.traffic_lights: Dict[int, TrafficLight] = {}
        self.congestion_matrix: Dict[Tuple[int, int], float] = {}
        
        # Eventos de tráfico
        self.traffic_events: Dict[str, TrafficEvent] = {}
        self.active_events: List[TrafficEvent] = []
        
        # Métricas del sistema
        self.system_metrics = {
            "total_vehicles": 0,
            "active_deliveries": 0,
            "average_speed": 0.0,
            "congestion_level": 0.0,
            "completed_deliveries": 0,
            "failed_deliveries": 0,
            "total_distance_traveled": 0.0,
            "total_fuel_consumed": 0.0,
            "emergency_responses": 0,
            "weather_delays": 0,
            "traffic_violations": 0,
            # Nuevas métricas BDI
            "bdi_decisions_made": 0,
            "bdi_collaborations": 0,
            "bdi_intentions_executed": 0
        }
        
        # Zonas especiales
        self.emergency_zones = []
        self.restricted_zones = []
        self.construction_zones = []
        self.school_zones = []
        self.hospital_zones = []
        
        # Configuración de generadores
        self.weather_generator_config = {
            "update_interval": 300,  # segundos
            "seasonal_variation": True,
            "extreme_weather_probability": 0.05
        }
        
        self.traffic_generator_config = {
            "rush_hour_start": 7,  # 7 AM
            "rush_hour_end": 9,    # 9 AM
            "evening_rush_start": 17,  # 5 PM
            "evening_rush_end": 19,    # 7 PM
            "weekend_traffic_factor": 0.7,
            "event_probability": 0.02
        }
        
        # Inicializar módulos especializados
        self.weather_analyzer = None
        self.traffic_crawler = None
        if WEATHER_AVAILABLE:
            try:
                self.weather_analyzer = WeatherImpactAnalyzer()
                self.traffic_crawler = TrafficCrawler()
            except Exception as e:
                print(f"Error inicializando analizadores: {e}")
        
        # Inicializar el entorno
        self._initialize_environment()
        
        # Inicializar sistema de comunicación BDI
        if COMMUNICATION_AVAILABLE:
            self.communication_task = None
    
    def _initialize_environment(self):
        # Inicializar segmentos de carretera
        self._initialize_road_segments()
        
        # Inicializar semáforos
        self._initialize_traffic_lights()
        
        # Inicializar tráfico civil
        self._initialize_civilian_traffic()
        
        # Configurar clima inicial
        self._initialize_weather()
        
        
        # Imprimir estado inicial del entorno
        print(f"✅ Entorno inicializado: {len(self.road_segments)} segmentos, "
              f"{len(self.traffic_lights)} semáforos, {len(self.vehicles)} vehículos")
    
    def _initialize_road_segments(self):
        for edge in self.street_graph.edges(data=True):
            node1, node2, data = edge
            edge_id = (node1, node2)
            
            # Extraer datos de la arista
            road_type = data.get('highway', 'residential')
            max_speed = data.get('maxspeed', 50)
            
            # Convertir string speeds a float
            if isinstance(max_speed, str):
                try:
                    max_speed = float(max_speed.replace(' km/h', '').replace('mph', ''))
                except (ValueError, AttributeError):
                    max_speed = 50.0
            
            # Determinar capacidad basada en tipo de vía
            capacity_map = {
                'motorway': 20,
                'trunk': 15,
                'primary': 12,
                'secondary': 10,
                'tertiary': 8,
                'residential': 6,
                'service': 4
            }
            
            capacity = capacity_map.get(road_type, 6)
            
            self.road_segments[edge_id] = RoadSegment(
                edge_id=edge_id,
                road_type=road_type,
                max_speed=max_speed,
                min_speed=max(10, max_speed * 0.4),
                capacity=capacity,
                slope=data.get('slope', 0.0),
                surface_type=data.get('surface', 'asphalt'),
                lighting=data.get('lit', True),
                toll=data.get('toll', False),
                toll_cost=data.get('toll_cost', 0.0)
            )
    
    def _initialize_traffic_lights(self):
        """Inicializa los semáforos"""
        
        candidates = [n for n in self.street_graph.nodes() if len(list(self.street_graph.neighbors(n))) >= 4]
        selected = random.sample(candidates, min(1500, len(candidates)))

        # Identificar intersecciones importantes (nodos con múltiples conexiones)
        for node in selected:
            
            self.traffic_lights[node] = TrafficLight(
                node_id=node,
                state=random.choice(["green", "red"]),
                green_duration=random.uniform(25, 35),
                red_duration=random.uniform(20, 30),
                is_adaptive=random.choice([True, False])
            )

    def _initialize_civilian_traffic(self):
        """Inicializa el tráfico civil"""
        num_vehicles = self.num_vehicles
        all_nodes = list(self.street_graph.nodes())
        
        for i in range(num_vehicles):
            vehicle_id = "vehicle"+ str(i)
            start_node = random.choice(all_nodes)
            node_data = self.street_graph.nodes[start_node]
            behavior = random.choice([b for b in CivilianBehavior])
            v = CivilianTrafficAgent(vehicle_id=vehicle_id, initial_position=[node_data["lat"], node_data["lon"]], initial_node=start_node, behavior=behavior)
            
            # NUEVO: Dar acceso al grafo de calles para cálculos de posición
            v.street_graph = self.street_graph
            v._street_graph = self.street_graph
            
            # Seleccionar destino aleatorio diferente al nodo inicial
            target_node = random.choice([n for n in all_nodes if n != start_node])
            
            # Calcular ruta usando Dijkstra
            try:
                # NetworkX ya implementa Dijkstra con shortest_path
                route = nx.shortest_path(self.street_graph, source=start_node, target=target_node, weight='weight')
                
                # Asignar ruta y destino al vehículo
                destination_types = ["work", "home", "shopping", "recreation", "service"]
                destination_type = random.choice(destination_types)
                v.assign_route_and_destination(route, target_node, destination_type)
                
                # NUEVO: Inicializar velocidad para que se muevan inmediatamente
                v.current_speed = max(v.base_speed * 0.8, 20.0)  # Al menos 20 km/h para movimiento visible
                v.movement_state = MovementState.MOVING  # Cambiar estado a movimiento
                
                print(f"   🛣️ Ruta calculada: {route[:3]}{'...' if len(route) > 3 else ''} ({len(route)} nodos)")
            
            except nx.NetworkXNoPath:
                # Si no hay ruta posible, asignar un destino cercano
                try:
                    # Buscar nodos vecinos del start_node
                    neighbors = list(self.street_graph.neighbors(start_node))
                    if neighbors:
                        target_node = random.choice(neighbors)
                        route = [start_node, target_node]
                        v.assign_route_and_destination(route, target_node, "local")
                        
                        # NUEVO: Inicializar velocidad para movimiento inmediato
                        v.current_speed = max(v.base_speed * 0.8, 20.0)  # Al menos 20 km/h
                        v.movement_state = MovementState.MOVING
                    else:
                        # Si no hay vecinos, crear ruta mínima
                        route = [start_node]
                        v.assign_route_and_destination(route, start_node, "idle")
                except Exception as e:
                    print(f"Error asignando ruta alternativa para {vehicle_id}: {e}")
                    route = [start_node]
                    v.assign_route_and_destination(route, start_node, "idle")
            # print("la ruta: ")
            # print(route)
            
            self.vehicles[vehicle_id] = v
            
            # Register vehicle with communication manager
            if COMMUNICATION_AVAILABLE:
                try:
                    # Schedule registration for later when event loop is available
                    self._pending_registrations = getattr(self, '_pending_registrations', [])
                    self._pending_registrations.append(v)
                    print(f"📝 Programado registro de {vehicle_id} en communication manager")
                except Exception as e:
                    print(f"⚠️ Error preparando registro de {vehicle_id}: {e}")
            
            print(f"✅ Vehículo {vehicle_id} creado con velocidad {v.current_speed:.1f} km/h, estado: {v.movement_state.value}")
            print(f"   📍 Posición inicial: ({v.lat:.6f}, {v.lon:.6f}), nodo: {start_node}")
            print(f"   🎯 Destino: nodo {target_node}, ruta: {len(route)} nodos")
            

    def _initialize_weather(self):
        # Configurar estado inicial basado en estación y ubicación
        current_month = self.current_time.month
        
        if current_month in [6, 7, 8]:  # Verano
            self.weather_state.temperature = random.uniform(25, 35)
            self.weather_state.humidity = random.uniform(60, 85)
            self.weather_state.condition = random.choice([
                WeatherCondition.CLEAR, WeatherCondition.CLOUDY, 
                WeatherCondition.LIGHT_RAIN, WeatherCondition.EXTREME_HEAT
            ])
        elif current_month in [12, 1, 2]:  # Invierno
            self.weather_state.temperature = random.uniform(10, 25)
            self.weather_state.humidity = random.uniform(40, 70)
            self.weather_state.condition = random.choice([
                WeatherCondition.CLEAR, WeatherCondition.CLOUDY, 
                WeatherCondition.FOG, WeatherCondition.LIGHT_RAIN
            ])
        else:  # Primavera/Otoño
            self.weather_state.temperature = random.uniform(15, 30)
            self.weather_state.humidity = random.uniform(50, 80)
            self.weather_state.condition = random.choice([
                WeatherCondition.CLEAR, WeatherCondition.CLOUDY, 
                WeatherCondition.LIGHT_RAIN, WeatherCondition.STORM
            ])
    
    def add_delivery_truck(self, truck_id: str, start_node: int, 
                          capacity: int = 1000, **kwargs) -> bool:
        """
        Args:
            truck_id: ID único del camión
            start_node: Nodo de inicio
            capacity: Capacidad en kg
            **kwargs: Parámetros adicionales
            
        Returns:
            bool: True si se añadió exitosamente
        """
        if truck_id in self.vehicles:
            return False
        
        try:
            node_data = self.street_graph.nodes[start_node]
            
            truck = VehicleState(
                vehicle_id=truck_id,
                vehicle_type="delivery_truck",
                current_node=start_node,
                lat=node_data.get('lat', 0.0),
                lon=node_data.get('lon', 0.0),
                capacity=capacity,
                # current_speed=kwargs.get('speed, 45.0),
                speed=kwargs.get('speed', 45.0),
                fuel_level=kwargs.get('fuel_level', 100.0),
                driver_type=kwargs.get('driver_type', 'normal'),
                emergency_priority=kwargs.get('emergency_priority', False)
            )
            
            self.vehicles[truck_id] = truck
            self.delivery_trucks[truck_id] = truck
            self.system_metrics["total_vehicles"] += 1
            
            return True
            
        except KeyError:
            print(f"Error: Nodo {start_node} no existe en el grafo")
            return False
    
    def add_traffic_event(self, event: TrafficEvent):
        """Añade un evento de tráfico al sistema"""
        self.traffic_events[event.event_id] = event
        if event.is_active:
            self.active_events.append(event)
            
        # Aplicar impacto del evento a los segmentos afectados
        self._apply_traffic_event_impact(event)
    
    def _apply_traffic_event_impact(self, event: TrafficEvent):
        """Aplica el impacto de un evento de tráfico"""
        impact_factor = event.impact_factor
        
        # Aplicar a calles afectadas
        for street in event.affected_streets:
            for edge_id, segment in self.road_segments.items():
                # Simplificación: aplicar a segmentos que coincidan con el nombre
                if self._street_matches_segment(street, edge_id):
                    segment.traffic_factor *= impact_factor
                    
                    # Efectos específicos por tipo de evento
                    if event.event_type == TrafficEventType.ROAD_CLOSURE:
                        segment.condition = RoadCondition.CLOSED
                    elif event.event_type == TrafficEventType.CONSTRUCTION:
                        segment.construction_factor *= 1.5
                    elif event.event_type == TrafficEventType.FLOODING:
                        segment.condition = RoadCondition.BAD
                        segment.weather_factor *= 2.0
    
    def _street_matches_segment(self, street_name: str, edge_id: Tuple[int, int]) -> bool:
        """Verifica si un nombre de calle coincide con un segmento"""
        # Implementación simplificada - en práctica sería más compleja
        node1, node2 = edge_id
        edge_data = self.street_graph.get_edge_data(node1, node2)
        
        if edge_data:
            segment_name = edge_data.get('name', '').lower()
            return street_name.lower() in segment_name
        
        return False
    
    def update_weather(self, new_weather: Optional[WeatherState] = None):
        """Actualiza el estado del clima"""
        if new_weather:
            self.weather_state = new_weather
        else:
            # Generación automática basada en patrones
            self._generate_weather_evolution()
        
        # Aplicar impacto del clima a los segmentos
        self._apply_weather_impact()
    
    def _generate_weather_evolution(self):
        """Genera evolución natural del clima"""
        # Cambios graduales en temperatura y humedad
        temp_change = random.uniform(-2, 2)
        humidity_change = random.uniform(-5, 5)
        
        self.weather_state.temperature += temp_change
        self.weather_state.humidity += humidity_change
        
        # Limitar rangos
        self.weather_state.temperature = max(-10, min(50, self.weather_state.temperature))
        self.weather_state.humidity = max(0, min(100, self.weather_state.humidity))
        
        # Posibilidad de cambio de condición
        if random.random() < 0.1:  # 10% probabilidad
            self.weather_state.condition = random.choice(list(WeatherCondition))
        
        # Actualizar otros parámetros
        if self.weather_state.condition in [WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN]:
            self.weather_state.precipitation = random.uniform(1, 20)
        elif self.weather_state.condition == WeatherCondition.STORM:
            self.weather_state.precipitation = random.uniform(10, 50)
            self.weather_state.wind_speed = random.uniform(30, 80)
        else:
            self.weather_state.precipitation = 0.0
            self.weather_state.wind_speed = random.uniform(5, 25)
    
    def _apply_weather_impact(self):
        """Aplica el impacto del clima a todos los segmentos"""
        base_factor = 1.0
        
        # Factores por condición climática
        weather_factors = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.CLOUDY: 1.05,
            WeatherCondition.LIGHT_RAIN: 1.3,
            WeatherCondition.HEAVY_RAIN: 1.8,
            WeatherCondition.STORM: 2.5,
            WeatherCondition.FOG: 1.6,
            WeatherCondition.SNOW: 3.0,
            WeatherCondition.EXTREME_HEAT: 1.2
        }
        
        weather_factor = weather_factors.get(self.weather_state.condition, 1.0)
        
        # Aplicar a todos los segmentos
        for segment in self.road_segments.values():
            segment.weather_factor = weather_factor
            
            # Ajustes específicos por tipo de vía
            if segment.road_type in ['motorway', 'trunk']:
                segment.weather_factor *= 0.8  # Mejor drenaje
            elif segment.road_type == 'residential':
                segment.weather_factor *= 1.2  # Peor drenaje
    
    def update_traffic_lights(self):
        """Actualiza el estado de los semáforos"""
        current_time = time.time()
        
        for traffic_light in self.traffic_lights.values():
            if traffic_light.priority_override:
                continue  # Skip si tiene override de emergencia
            
            # Calcular tiempo transcurrido
            time_in_state = current_time - traffic_light.last_change.timestamp()
            
            # Cambiar estado según ciclo
            if traffic_light.state == "green" and time_in_state >= traffic_light.green_duration:
                traffic_light.state = "yellow"
                traffic_light.last_change = datetime.now()
                
            elif traffic_light.state == "yellow" and time_in_state >= traffic_light.yellow_duration:
                traffic_light.state = "red"
                traffic_light.last_change = datetime.now()
                
            elif traffic_light.state == "red" and time_in_state >= traffic_light.red_duration:
                traffic_light.state = "green"
                traffic_light.last_change = datetime.now()
    
    def update_congestion(self):
        """Actualiza la matriz de congestión"""
        for edge_id, segment in self.road_segments.items():
            # Calcular factor de congestión basado en ocupación
            occupancy_ratio = segment.current_vehicles / segment.capacity
            
            # Función logística para congestión
            congestion_factor = 1 + (occupancy_ratio ** 2) * 3
            
            # Aplicar factores ambientales
            total_factor = (congestion_factor * 
                          segment.weather_factor * 
                          segment.traffic_factor * 
                          segment.construction_factor)
            
            self.congestion_matrix[edge_id] = total_factor
    
    def get_environment_state(self) -> Dict[str, Any]:
        """
        Returns:
            Dict con todo el estado del entorno
        """
        return {
            # Tiempo y configuración
            "current_time": self.current_time,
            "simulation_time": (self.current_time - self.simulation_start_time).total_seconds(),
            "time_step": self.time_step,
            
            # Estado del clima
            "weather": {
                "condition": self.weather_state.condition.value,
                "temperature": self.weather_state.temperature,
                "humidity": self.weather_state.humidity,
                "precipitation": self.weather_state.precipitation,
                "wind_speed": self.weather_state.wind_speed,
                "visibility": self.weather_state.visibility
            },
            
            # Vehículos civiles
            "vehicles": {vid: {
                "type": v.vehicle_type,
                "lat": v.lat,
                "lon": v.lon,
                "speed": v.current_speed,
                "current_node": v.current_node,
                "next_node": v.next_node,
                "progress": v.progress,
                "fuel_level": v.fuel_level,
                "current_load": v.current_load,
                "capacity": v.capacity,
                "behavior": v.behavior.value,
                "movement_state": v.movement_state.value,
                "target_node": v.target_node,
                "route": v.route,
                "has_destination": v.has_destination,
                "destination_type": v.destination_type,
                "cooperation_level": v.cooperation_level,
                "risk_tolerance": v.risk_tolerance,
                "patience_level": v.patience_level,
                "emergency_priority": getattr(v, 'emergency_priority', False),
                "distance_traveled": v.distance_traveled,
                "emergency_responses": v.emergency_responses,
                "traffic_violations": v.traffic_violations
            } for vid, v in self.vehicles.items()},
            
            # Camiones de reparto específicamente
            "delivery_trucks": {tid: {
                "capacity": t.capacity,
                "current_load": t.current_load,
                "route": t.route,
                "driver_type": t.driver_type,
                "maintenance_needed": t.maintenance_needed
            } for tid, t in self.delivery_trucks.items()},
            
            # Camiones BDI - NUEVO SISTEMA
            "bdi_delivery_trucks": {tid: {
                "agent_id": t.agent_id,
                "current_node": t.current_node,
                "lat": t.lat,
                "lon": t.lon,
                "speed": t.current_speed,
                "capacity": t.capacity,
                "current_load": t.current_load,
                "fuel_level": t.fuel_level,
                "route": t.route,
                "delivery_locations": t.delivery_locations,
                "completed_deliveries": t.completed_deliveries,
                "progress": t.progress,
                "bdi_status": t.get_status(),
                "delivery_metrics": t.delivery_metrics
            } for tid, t in self.bdi_delivery_trucks.items()},
            
            # Estado de la red vial
            "road_network": {
                "congestion": self.congestion_matrix,
                "traffic_lights": {nid: {
                    "state": tl.state,
                    "cycle_time": tl.cycle_time,
                    "priority_override": tl.priority_override
                } for nid, tl in self.traffic_lights.items()},
                "road_conditions": {str(eid): {
                    "condition": rs.condition.value,
                    "weather_factor": rs.weather_factor,
                    "traffic_factor": rs.traffic_factor,
                    "construction_factor": rs.construction_factor,
                    "current_vehicles": rs.current_vehicles,
                    "capacity": rs.capacity
                } for eid, rs in self.road_segments.items()}
            },
            
            # Eventos activos
            "traffic_events": [{
                "event_id": event.event_id,
                "type": event.event_type.value,
                "severity": event.severity,
                "affected_streets": event.affected_streets,
                "impact_factor": event.impact_factor,
                "description": event.description
            } for event in self.active_events],
            
            # Zonas especiales
            "special_zones": {
                "emergency_zones": self.emergency_zones,
                "restricted_zones": self.restricted_zones,
                "construction_zones": self.construction_zones,
                "school_zones": self.school_zones,
                "hospital_zones": self.hospital_zones
            },
            
            # Métricas del sistema
            "system_metrics": self.system_metrics.copy(),
            
            # Información adicional para optimización
            "optimization_context": {
                "is_rush_hour": self._is_rush_hour(),
                "weekend": self.current_time.weekday() >= 5,
                "holiday": self._is_holiday(),
                "emergency_active": any(e.event_type == TrafficEventType.EMERGENCY 
                                     for e in self.active_events)
            }
        }
    
    def _is_rush_hour(self) -> bool:
        hour = self.current_time.hour
        return (self.traffic_generator_config["rush_hour_start"] <= hour <= 
                self.traffic_generator_config["rush_hour_end"] or
                self.traffic_generator_config["evening_rush_start"] <= hour <= 
                self.traffic_generator_config["evening_rush_end"])
    
    def _is_holiday(self) -> bool:
        # Implementación básica - en práctica sería más compleja
        return False
    
    async def step(self):
        """Avanza la simulación un paso"""
        # Register pending agents if needed
        if hasattr(self, '_pending_registrations') and self._pending_registrations:
            await self._register_pending_agents()
        
        # Actualizar tiempo
        self.current_time += timedelta(seconds=self.time_step)
        
        # Actualizar componentes del entorno
        await self._update_vehicle_positions()
        self.update_traffic_lights()
        self.update_congestion()
        
        # Actualizar clima periódicamente
        if (self.current_time.timestamp() % 
            self.weather_generator_config["update_interval"]) < self.time_step:
            self.update_weather()
        
        # Generar eventos de tráfico
        if random.random() < self.traffic_generator_config["event_probability"]:
            self._generate_random_traffic_event()
        
        # Actualizar métricas
        self._update_system_metrics()
        
        # Limpiar eventos expirados
        self._cleanup_expired_events()
    
    async def _register_pending_agents(self):
        """Registra agentes pendientes en el communication manager"""
        if not COMMUNICATION_AVAILABLE:
            # Clear pending registrations since communication is disabled
            self._pending_registrations.clear()
            return
            
        # Communication manager is disabled, just clear the list
        self._pending_registrations.clear()
        print("Communication manager disabled - skipping agent registration")
    
    async def _update_vehicle_positions(self):
        """Actualiza posiciones de todos los vehículos"""
        for vehicle in self.vehicles.values():
            await vehicle.next_step(self.get_environment_state())
    
    def _move_vehicle_to_next_node(self, vehicle: VehicleState):
        """Mueve un vehículo al siguiente nodo"""
        if vehicle.next_node is None:
            return
            
        # Decrementar congestión en arista actual
        old_edge = (vehicle.current_node, vehicle.next_node)
        if old_edge in self.road_segments:
            self.road_segments[old_edge].current_vehicles -= 1
        
        # Actualizar posición
        vehicle.current_node = vehicle.next_node
        node_data = self.street_graph.nodes[vehicle.current_node]
        vehicle.lat = node_data.get('lat', 0.0)
        vehicle.lon = node_data.get('lon', 0.0)
        
        # Asignar nuevo destino
        neighbors = list(self.street_graph.neighbors(vehicle.current_node))
        if neighbors:
            vehicle.next_node = random.choice(neighbors)
            vehicle.progress = 0.0
            
            # Incrementar congestión en nueva arista
            new_edge = (vehicle.current_node, vehicle.next_node)
            if new_edge in self.road_segments:
                self.road_segments[new_edge].current_vehicles += 1
        else:
            vehicle.next_node = None
    
    def _generate_random_traffic_event(self):
        """Genera un evento de tráfico aleatorio"""
        event_types = list(TrafficEventType)
        event_type = random.choice(event_types)
        
        # Seleccionar ubicación aleatoria
        nodes = list(self.street_graph.nodes())
        location_node = random.choice(nodes)
        node_data = self.street_graph.nodes[location_node]
        location = (node_data.get('lat', 0.0), node_data.get('lon', 0.0))
        
        # Obtener calles afectadas
        affected_streets = []
        for neighbor in self.street_graph.neighbors(location_node):
            edge_data = self.street_graph.get_edge_data(location_node, neighbor)
            if edge_data and 'name' in edge_data:
                affected_streets.append(edge_data['name'])
        
        # Crear evento
        event = TrafficEvent(
            event_id=f"event_{int(time.time())}_{random.randint(1000, 9999)}",
            event_type=event_type,
            location=location,
            affected_streets=affected_streets,
            severity=random.randint(1, 5),
            start_time=self.current_time,
            estimated_duration=timedelta(minutes=random.randint(15, 120)),
            description=f"Evento de {event_type.value} en {location}",
            impact_factor=random.uniform(1.2, 3.0)
        )
        
        self.add_traffic_event(event)
    
    def _update_system_metrics(self):
        """Actualiza las métricas del sistema"""
        # Contar vehículos activos
        active_vehicles = sum(1 for v in self.vehicles.values())
        self.system_metrics["total_vehicles"] = active_vehicles
        
        # Contar entregas activas
        active_deliveries = sum(1 for t in self.delivery_trucks.values() 
                              if t.is_active and t.current_load > 0)
        self.system_metrics["active_deliveries"] = active_deliveries
        
        # Calcular velocidad promedio
        if active_vehicles > 0:
            total_speed = sum(v.current_speed for v in self.vehicles.values())
            self.system_metrics["average_speed"] = total_speed / active_vehicles
        
        # Calcular nivel de congestión promedio
        if self.congestion_matrix:
            total_congestion = sum(self.congestion_matrix.values())
            self.system_metrics["congestion_level"] = total_congestion / len(self.congestion_matrix)
    
    def _cleanup_expired_events(self):
        """Limpia eventos expirados"""
        current_time = self.current_time
        expired_events = []
        
        for event in self.active_events:
            if current_time >= event.start_time + event.estimated_duration:
                expired_events.append(event)
                event.is_active = False
        
        # Remover eventos expirados
        for event in expired_events:
            self.active_events.remove(event)
            self._remove_traffic_event_impact(event)
    
    def _remove_traffic_event_impact(self, event: TrafficEvent):
        """Remueve el impacto de un evento de tráfico"""
        # Restaurar valores originales en segmentos afectados
        for street in event.affected_streets:
            for edge_id, segment in self.road_segments.items():
                if self._street_matches_segment(street, edge_id):
                    segment.traffic_factor = 1.0
                    segment.construction_factor = 1.0
                    
                    if segment.condition == RoadCondition.CLOSED:
                        segment.condition = RoadCondition.GOOD
    
    def get_route_optimization_context(self) -> Dict[str, Any]:
        """
        Retorna contexto específico para optimización de rutas
        
        Returns:
            Dict con información relevante para optimización
        """
        return {
            "weather_factors": self._get_weather_factors(),
            "traffic_factors": self._get_traffic_factors(),
            "congestion_matrix": self.congestion_matrix.copy(),
            "active_restrictions": self._get_active_restrictions(),
            "priority_zones": self._get_priority_zones(),
            "time_context": {
                "hour": self.current_time.hour,
                "day_of_week": self.current_time.weekday(),
                "is_rush_hour": self._is_rush_hour(),
                "weather_condition": self.weather_state.condition.value
            }
        }
    
    def _get_weather_factors(self) -> Dict[str, float]:
        """Retorna factores de impacto del clima por tipo de vía"""
        base_factor = 1.0
        
        if self.weather_state.condition == WeatherCondition.HEAVY_RAIN:
            return {
                "motorway": 1.3,
                "primary": 1.5,
                "secondary": 1.7,
                "residential": 2.0,
                "unpaved": 3.0
            }
        elif self.weather_state.condition == WeatherCondition.LIGHT_RAIN:
            return {
                "motorway": 1.1,
                "primary": 1.2,
                "secondary": 1.3,
                "residential": 1.5,
                "unpaved": 2.0
            }
        elif self.weather_state.condition == WeatherCondition.FOG:
            return {
                "motorway": 1.4,
                "primary": 1.3,
                "secondary": 1.2,
                "residential": 1.1,
                "unpaved": 1.1
            }
        else:
            return {
                "motorway": 1.0,
                "primary": 1.0,
                "secondary": 1.0,
                "residential": 1.0,
                "unpaved": 1.0
            }
    
    def _get_traffic_factors(self) -> Dict[Tuple[int, int], float]:
        """Retorna factores de tráfico por arista"""
        return {edge_id: segment.traffic_factor 
                for edge_id, segment in self.road_segments.items()}
    
    def _get_active_restrictions(self) -> List[Dict[str, Any]]:
        """Retorna restricciones activas"""
        restrictions = []
        
        for event in self.active_events:
            if event.event_type == TrafficEventType.ROAD_CLOSURE:
                restrictions.append({
                    "type": "road_closure",
                    "affected_streets": event.affected_streets,
                    "severity": event.severity
                })
        
        return restrictions
    
    def _get_priority_zones(self) -> List[Dict[str, Any]]:
        return [
            {"type": "emergency", "zones": self.emergency_zones},
            {"type": "hospital", "zones": self.hospital_zones},
            {"type": "school", "zones": self.school_zones}
        ]
    
    def __str__(self) -> str:
        return (f"Environment(vehicles={len(self.vehicles)}, "
                f"delivery_trucks={len(self.delivery_trucks)}, "
                f"road_segments={len(self.road_segments)}, "
                f"traffic_lights={len(self.traffic_lights)}, "
                f"active_events={len(self.active_events)}, "
                f"weather={self.weather_state.condition.value})")
    
    def get_vehicle_positions(self) -> List[Dict[str, Any]]:
        """
        Obtiene las posiciones actuales de todos los vehículos para la interfaz visual
        
        Returns:
            Lista de diccionarios con información de posición de cada vehículo
        """
        vehicle_positions = []
        
        for vehicle_id, vehicle in self.vehicles.items():
            # Asegurar que las coordenadas sean números válidos
            lat = float(vehicle.lat) if vehicle.lat is not None else 0.0
            lon = float(vehicle.lon) if vehicle.lon is not None else 0.0
            
            position_data = {
                "id": vehicle_id,
                "lat": lat,
                "lon": lon,
                "position": [lat, lon],  # Formato alternativo para deck.gl
                "speed": float(vehicle.current_speed) if vehicle.current_speed is not None else 0.0,
                "behavior": vehicle.behavior.value if hasattr(vehicle.behavior, 'value') else str(vehicle.behavior),
                "state": vehicle.movement_state.value if hasattr(vehicle.movement_state, 'value') else str(vehicle.movement_state),
                "type": vehicle.vehicle_type if isinstance(vehicle.vehicle_type, str) else (vehicle.vehicle_type.value if hasattr(vehicle.vehicle_type, 'value') else "car"),
                "current_node": vehicle.current_node,
                "next_node": vehicle.next_node,
                "progress": float(vehicle.progress) if vehicle.progress is not None else 0.0,
                "fuel_level": float(vehicle.fuel_level) if vehicle.fuel_level is not None else 100.0,
                "has_destination": bool(vehicle.has_destination),
                "target_node": vehicle.target_node,
                "trail": getattr(vehicle, 'trail', [])  # Add trail for visualization
            }
            vehicle_positions.append(position_data)
        
        return vehicle_positions
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado general de la simulación
        
        Returns:
            Diccionario con información del estado de la simulación
        """
        return {
            "is_running": True,  # Asumimos que está corriendo si estamos obteniendo el estado
            "simulation_time": (self.current_time - self.simulation_start_time).total_seconds(),
            "total_vehicles": len(self.vehicles),
            "total_delivery_trucks": len(self.delivery_trucks),
            "total_bdi_trucks": len(self.bdi_delivery_trucks),
            "total_road_segments": len(self.road_segments),
            "total_traffic_lights": len(self.traffic_lights),
            "active_events": len(self.active_events),
            "weather_condition": self.weather_state.condition.value if hasattr(self.weather_state.condition, 'value') else str(self.weather_state.condition),
            "average_speed": sum(v.current_speed for v in self.vehicles.values()) / len(self.vehicles) if self.vehicles else 0,
            "active_traffic_violations": sum(v.traffic_violations for v in self.vehicles.values()),
            "emergency_responses": sum(v.emergency_responses for v in self.vehicles.values())
        }
    
    def add_bdi_delivery_truck(self, truck_id: str, start_node: int,
                              capacity: int = 1000, delivery_locations: List[int] = None) -> bool:
        """
        Añade un camión de reparto BDI al sistema
        
        Args:
            truck_id: ID único del camión
            start_node: Nodo de inicio
            capacity: Capacidad en kg
            delivery_locations: Lista de ubicaciones de entrega
            
        Returns:
            bool: True si se añadió exitosamente
        """
        if truck_id in self.bdi_delivery_trucks:
            return False
        
        try:
            # Crear camión BDI
            truck = DeliveryTruckBDI(
                agent_id=truck_id,
                initial_node=start_node,
                capacity=capacity
            )
            
            # Configurar entorno y grafo
            truck.set_environment_reference(self, self.street_graph)
            
            # Asignar ubicaciones de entrega si se proporcionan
            if delivery_locations:
                truck.assign_delivery_route(delivery_locations)
            
            # Registrar en el sistema de comunicación
            if COMMUNICATION_AVAILABLE:
                communication_manager.register_agent(truck)
            
            # Añadir al entorno
            self.bdi_delivery_trucks[truck_id] = truck
            self.system_metrics["total_vehicles"] += 1
            self.system_metrics["active_deliveries"] += 1
            
            print(f"✅ Camión BDI {truck_id} añadido en nodo {start_node}")
            return True
            
        except Exception as e:
            print(f"Error añadiendo camión BDI {truck_id}: {e}")
            return False
    
    def remove_bdi_delivery_truck(self, truck_id: str) -> bool:
        """Remueve un camión BDI del sistema"""
        if truck_id not in self.bdi_delivery_trucks:
            return False
        
        try:
            # Desregistrar de comunicación
            if COMMUNICATION_AVAILABLE:
                communication_manager.unregister_agent(truck_id)
            
            # Remover del entorno
            del self.bdi_delivery_trucks[truck_id]
            self.system_metrics["total_vehicles"] -= 1
            self.system_metrics["active_deliveries"] -= 1
            
            print(f"Camión BDI {truck_id} removido del sistema")
            return True
            
        except Exception as e:
            print(f"Error removiendo camión BDI {truck_id}: {e}")
            return False
    
    async def update_bdi_trucks(self, delta_time: float):
        """Actualiza todos los camiones BDI"""
        try:
            # Obtener estado del entorno para los agentes BDI
            env_state = self.get_environment_state()
            
            # Procesar cada camión BDI
            for truck_id, truck in self.bdi_delivery_trucks.items():
                try:
                    # Ejecutar ciclo BDI
                    await truck.bdi_cycle(env_state)
                    
                    # Actualizar posición física
                    truck.update_position(delta_time)
                    
                    # Actualizar métricas del sistema
                    truck_metrics = truck.metrics
                    self.system_metrics["bdi_decisions_made"] += truck_metrics.get("decisions_made", 0)
                    self.system_metrics["bdi_intentions_executed"] += truck_metrics.get("intentions_executed", 0)
                    
                    # Resetear métricas del camión para evitar acumulación
                    truck.metrics["decisions_made"] = 0
                    truck.metrics["intentions_executed"] = 0
                    
                except Exception as e:
                    print(f"Error actualizando camión BDI {truck_id}: {e}")
            
            # Procesar comunicación entre agentes
            if COMMUNICATION_AVAILABLE:
                for truck in self.bdi_delivery_trucks.values():
                    communication_manager.process_agent_messages(truck.agent_id)
                    
        except Exception as e:
            print(f"Error en actualización de camiones BDI: {e}")
    
    def get_bdi_trucks_status(self) -> Dict[str, Any]:
        """Obtiene el estado de todos los camiones BDI"""
        status = {}
        
        for truck_id, truck in self.bdi_delivery_trucks.items():
            try:
                status[truck_id] = truck.get_delivery_status()
            except Exception as e:
                status[truck_id] = {"error": str(e)}
        
        return status
    
    def start_bdi_trucks_movement(self):
        """Inicia el movimiento de todos los camiones BDI"""
        for truck in self.bdi_delivery_trucks.values():
            try:
                truck.start_movement()
            except Exception as e:
                print(f"Error iniciando movimiento de {truck.agent_id}: {e}")
    
    async def start_communication_system(self):
        """Inicia el sistema de comunicación BDI"""
        if COMMUNICATION_AVAILABLE and self.communication_task is None:
            self.communication_task = asyncio.create_task(communication_manager.communication_loop())
            print("✅ Sistema de comunicación BDI iniciado")
    
    async def stop_communication_system(self):
        """Detiene el sistema de comunicación BDI"""
        if COMMUNICATION_AVAILABLE and self.communication_task:
            communication_manager.is_active = False
            self.communication_task.cancel()
            try:
                await self.communication_task
            except asyncio.CancelledError:
                pass
            self.communication_task = None
            print("Sistema de comunicación BDI detenido")
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de comunicación"""
        if COMMUNICATION_AVAILABLE:
            return communication_manager.get_communication_stats()
        return {"error": "Communication system not available"}

    def execute_bdi_agent_action(self, agent_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción solicitada por un agente BDI"""
        if agent_id not in self.bdi_delivery_trucks:
            return {"success": False, "error": f"Agent {agent_id} not found"}
        
        agent = self.bdi_delivery_trucks[agent_id]
        action_type = action.get("type", "")
        
        try:
            if action_type == "change_route":
                return self._execute_route_change(agent, action)
            elif action_type == "adjust_speed":
                return self._execute_speed_adjustment(agent, action)
            elif action_type == "emergency_stop":
                return self._execute_emergency_stop(agent, action)
            else:
                return {"success": False, "error": f"Unknown action type: {action_type}"}
        except Exception as e:
            return {"success": False, "error": f"Action execution failed: {str(e)}"}

    def _execute_route_change(self, agent, action: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta cambio de ruta"""
        new_route = action.get("new_route", [])
        if not new_route:
            return {"success": False, "error": "No route provided"}
        
        old_route = agent.route.copy()
        agent.route = new_route
        agent.next_node = new_route[1] if len(new_route) > 1 else None
        agent.progress = 0.0
        self.system_metrics["bdi_decisions_made"] += 1
        
        return {"success": True, "message": "Route changed", 
                "old_route_length": len(old_route), 
                "new_route_length": len(new_route)}

    def _execute_speed_adjustment(self, agent, action: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta ajuste de velocidad"""
        target_speed = action.get("target_speed", agent.base_speed)
        target_speed = max(10.0, min(target_speed, agent.max_speed))
        
        old_speed = agent.current_speed
        agent.current_speed = target_speed
        
        return {"success": True, "message": "Speed adjusted",
                "old_speed": old_speed, "new_speed": target_speed}

    def _execute_emergency_stop(self, agent, action: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta parada de emergencia"""
        agent.current_speed = 0.0
        agent.emergency_priority = True
        self.system_metrics["emergency_responses"] += 1
        
        return {"success": True, "message": "Emergency stop executed"}
    
    # BDI Agent Integration Methods
    
    def get_bdi_agent_perception(self, agent_id: str, perception_range: float = 1000.0) -> Dict[str, Any]:
        """
        Proporciona información del entorno para un agente BDI específico
        
        Args:
            agent_id: ID del agente BDI
            perception_range: Rango de percepción en metros
            
        Returns:
            Dict con información percibida del entorno
        """
        if agent_id not in self.bdi_delivery_trucks:
            return {"error": f"Agent {agent_id} not found"}
        
        agent = self.bdi_delivery_trucks[agent_id]
        
        # Información básica del entorno
        perception = {
            "timestamp": self.current_time.isoformat(),
            "agent_position": {
                "current_node": agent.current_node,
                "next_node": agent.next_node,
                "lat": agent.lat,
                "lon": agent.lon,
                "progress": agent.progress
            },
            "weather": {
                "condition": self.weather_state.condition.value,
                "temperature": self.weather_state.temperature,
                "visibility": self.weather_state.visibility,
                "precipitation": self.weather_state.precipitation,
                "wind_speed": self.weather_state.wind_speed
            },
            "traffic_conditions": self._get_local_traffic_info(agent.current_node, perception_range),
            "nearby_agents": self._get_nearby_bdi_agents(agent_id, perception_range),
            "road_conditions": self._get_road_conditions(agent.current_node, agent.next_node),
            "traffic_lights": self._get_nearby_traffic_lights(agent.current_node, perception_range),
            "emergency_events": self._get_nearby_emergency_events(agent.current_node, perception_range),
            "delivery_opportunities": self._get_delivery_opportunities(agent_id),
            "fuel_stations": self._get_nearby_fuel_stations(agent.current_node, perception_range),
            "congestion_forecast": self._get_congestion_forecast(agent.current_node)
        }
        
        return perception
    
    def _get_local_traffic_info(self, node: int, range_meters: float) -> Dict[str, Any]:
        """Obtiene información de tráfico local"""
        traffic_info = {
            "congestion_level": 0.0,
            "average_speed": 0.0,
            "vehicle_count": 0,
            "affected_routes": []
        }
        
        # Analizar congestión en nodos cercanos
        if node in self.street_graph.nodes:
            neighbors = list(self.street_graph.neighbors(node))
            total_congestion = 0.0
            total_vehicles = 0
            
            for neighbor in neighbors:
                edge = (node, neighbor)
                if edge in self.congestion_matrix:
                    congestion = self.congestion_matrix[edge]
                    total_congestion += congestion
                    
                if edge in self.road_segments:
                    total_vehicles += self.road_segments[edge].current_vehicles
            
            if neighbors:
                traffic_info["congestion_level"] = total_congestion / len(neighbors)
                traffic_info["vehicle_count"] = total_vehicles
                traffic_info["average_speed"] = 50.0 * (1.0 - traffic_info["congestion_level"])
        
        return traffic_info
    
    def _get_nearby_bdi_agents(self, agent_id: str, range_meters: float) -> List[Dict[str, Any]]:
        """Obtiene información de otros agentes BDI cercanos"""
        nearby_agents = []
        
        if agent_id not in self.bdi_delivery_trucks:
            return nearby_agents
        
        current_agent = self.bdi_delivery_trucks[agent_id]
        
        for other_id, other_agent in self.bdi_delivery_trucks.items():
            if other_id == agent_id:
                continue
            
            # Calcular distancia aproximada (simplificada)
            distance = self._calculate_node_distance(
                current_agent.current_node, 
                other_agent.current_node
            )
            
            if distance <= range_meters / 1000.0:  # Convertir a km
                agent_info = {
                    "agent_id": other_id,
                    "distance_km": distance,
                    "current_node": other_agent.current_node,
                    "speed": other_agent.current_speed,
                    "fuel_level": other_agent.fuel_level,
                    "current_load": other_agent.current_load,
                    "delivery_count": len(other_agent.completed_deliveries),
                    "available_for_collaboration": other_agent.fuel_level > 20.0 and other_agent.current_load < other_agent.capacity * 0.8
                }
                nearby_agents.append(agent_info)
        
        return nearby_agents
    
    def _get_road_conditions(self, current_node: int, next_node: Optional[int]) -> Dict[str, Any]:
        """Obtiene condiciones de la carretera actual"""
        road_info = {
            "current_segment": None,
            "next_segment": None,
            "surface_conditions": "good",
            "weather_impact": 1.0,
            "construction": False,
            "speed_limit": 50.0
        }
        
        if next_node and (current_node, next_node) in self.road_segments:
            segment = self.road_segments[(current_node, next_node)]
            road_info["current_segment"] = {
                "road_type": segment.road_type,
                "max_speed": segment.max_speed,
                "capacity": segment.capacity,
                "current_vehicles": segment.current_vehicles,
                "congestion_ratio": segment.current_vehicles / segment.capacity if segment.capacity > 0 else 0.0,
                "weather_factor": segment.weather_factor,
                "traffic_factor": segment.traffic_factor,
                "toll": segment.toll,
                "toll_cost": segment.toll_cost
            }
            road_info["speed_limit"] = segment.max_speed
            road_info["weather_impact"] = segment.weather_factor
        
        return road_info
    
    def _get_nearby_traffic_lights(self, node: int, range_meters: float) -> List[Dict[str, Any]]:
        """Obtiene información de semáforos cercanos"""
        nearby_lights = []
        
        # Buscar semáforos en el nodo actual y nodos vecinos
        nodes_to_check = [node] + list(self.street_graph.neighbors(node)) if node in self.street_graph.nodes else []
        
        for check_node in nodes_to_check:
            if check_node in self.traffic_lights:
                light = self.traffic_lights[check_node]
                light_info = {
                    "node_id": check_node,
                    "state": light.state,
                    "time_until_change": self._calculate_time_until_change(light),
                    "cycle_time": light.cycle_time,
                    "is_adaptive": light.is_adaptive
                }
                nearby_lights.append(light_info)
        
        return nearby_lights
    
    def _get_nearby_emergency_events(self, node: int, range_meters: float) -> List[Dict[str, Any]]:
        """Obtiene eventos de emergencia cercanos"""
        nearby_events = []
        
        for event in self.active_events:
            if not event.is_active:
                continue
            
            # Verificar si el evento afecta nodos cercanos
            event_info = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "severity": event.severity,
                "impact_factor": event.impact_factor,
                "estimated_duration": event.estimated_duration.total_seconds(),
                "description": event.description
            }
            nearby_events.append(event_info)
        
        return nearby_events
    
    def _get_delivery_opportunities(self, agent_id: str) -> Dict[str, Any]:
        """Obtiene oportunidades de entrega para el agente"""
        if agent_id not in self.bdi_delivery_trucks:
            return {"error": "Agent not found"}
        
        agent = self.bdi_delivery_trucks[agent_id]
        
        opportunities = {
            "pending_deliveries": len(agent.delivery_locations) - len(agent.completed_deliveries),
            "next_delivery_location": None,
            "estimated_delivery_time": 0.0,
            "potential_collaborations": []
        }
        
        # Identificar próxima entrega
        remaining_deliveries = [loc for loc in agent.delivery_locations if loc not in agent.completed_deliveries]
        if remaining_deliveries and agent.route:
            for delivery_loc in remaining_deliveries:
                if delivery_loc in agent.route:
                    opportunities["next_delivery_location"] = delivery_loc
                    break
        
        # Buscar oportunidades de colaboración
        for other_id, other_agent in self.bdi_delivery_trucks.items():
            if other_id == agent_id:
                continue
            
            # Verificar si hay entregas cercanas que podrían coordinarse
            common_areas = set(agent.delivery_locations) & set(other_agent.delivery_locations)
            if common_areas:
                collaboration = {
                    "agent_id": other_id,
                    "common_delivery_areas": list(common_areas),
                    "potential_savings": len(common_areas) * 0.1  # Estimación simple
                }
                opportunities["potential_collaborations"].append(collaboration)
        
        return opportunities
    
    def _get_nearby_fuel_stations(self, node: int, range_meters: float) -> List[Dict[str, Any]]:
        """Obtiene estaciones de combustible cercanas (simuladas)"""
        # En una implementación real, esto consultaría una base de datos de estaciones
        fuel_stations = []
        
        # Simular algunas estaciones de combustible
        if node % 10 == 0:  # Cada 10 nodos aproximadamente
            station_info = {
                "station_id": f"fuel_station_{node}",
                "node_id": node,
                "fuel_types": ["diesel", "gasoline"],
                "price_per_liter": 1.2 + random.uniform(-0.2, 0.2),
                "availability": True,
                "wait_time_minutes": random.randint(2, 8)
            }
            fuel_stations.append(station_info)
        
        return fuel_stations
    
    def _get_congestion_forecast(self, node: int) -> Dict[str, Any]:
        """Obtiene pronóstico de congestión"""
        # Simulación simple de pronóstico
        current_hour = self.current_time.hour
        
        # Patrones típicos de tráfico
        rush_hour_morning = 7 <= current_hour <= 9
        rush_hour_evening = 17 <= current_hour <= 19
        
        forecast = {
            "next_hour_congestion": 0.3,
            "peak_hour_expected": False,
            "recommended_routes": [],
            "traffic_trend": "stable"
        }
        
        if rush_hour_morning or rush_hour_evening:
            forecast["next_hour_congestion"] = 0.7
            forecast["peak_hour_expected"] = True
            forecast["traffic_trend"] = "increasing"
        elif 22 <= current_hour or current_hour <= 6:
            forecast["next_hour_congestion"] = 0.1
            forecast["traffic_trend"] = "decreasing"
        
        return forecast
    
    def _calculate_time_until_change(self, traffic_light: TrafficLight) -> float:
        """Calcula tiempo hasta el próximo cambio de semáforo"""
        time_since_change = (self.current_time - traffic_light.last_change).total_seconds()
        
        if traffic_light.state == "green":
            return max(0, traffic_light.green_duration - time_since_change)
        elif traffic_light.state == "yellow":
            return max(0, traffic_light.yellow_duration - time_since_change)
        else:  # red
            return max(0, traffic_light.red_duration - time_since_change)
    
    def _calculate_node_distance(self, node1: int, node2: int) -> float:
        """Calcula distancia aproximada entre dos nodos"""
        if node1 not in self.street_graph.nodes or node2 not in self.street_graph.nodes:
            return float('inf')
        
        try:
            path_length = nx.shortest_path_length(self.street_graph, node1, node2, weight='weight')
            return path_length
        except nx.NetworkXNoPath:
            return float('inf')