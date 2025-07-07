import sys
import os
import time
import random
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Imports
sys.path.append("src")
sys.path.append("src/weather")
sys.path.append("src/traffic_events")
sys.path.append("src/crawler")

try:
    from weather.weather_impact_analyzer import WeatherImpactAnalyzer
    from traffic_events.traffic_events_analyzer import apply_traffic_weights
    from crawler.traffic_events_crawler import TrafficCrawler
    WEATHER_AVAILABLE = True
except ImportError as e:
    print(f"Advertencia: Módulos de clima/tráfico no disponibles: {e}")
    WEATHER_AVAILABLE = False


class WeatherCondition(Enum):
    """Condiciones climáticas posibles"""
    CLEAR = "despejado"
    CLOUDY = "nublado"
    LIGHT_RAIN = "lluvia_ligera"
    HEAVY_RAIN = "lluvia_fuerte"
    STORM = "tormenta"
    FOG = "niebla"
    EXTREME_HEAT = "calor_extremo"


class RoadCondition(Enum):
    """Condiciones de las vías"""
    EXCELLENT = "excelente"
    GOOD = "buena"
    REGULAR = "regular"
    BAD = "mala"
    CLOSED = "cerrada"


class TrafficEventType(Enum):
    """Tipos de eventos de tráfico"""
    ACCIDENT = "accidente"
    CONSTRUCTION = "construccion"
    PROTEST = "protesta"
    SPECIAL_EVENT = "evento_especial"
    VEHICLE_BREAKDOWN = "averia_vehiculo"
    EMERGENCY = "emergencia"
    ROAD_CLOSURE = "cierre_vial"
    FLOODING = "inundacion"


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
        self.vehicles: Dict[str, VehicleState] = {}
        self.delivery_trucks: Dict[str, VehicleState] = {}
        self.civilian_traffic: Dict[str, VehicleState] = {}
        
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
            "traffic_violations": 0
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
    
    def _initialize_environment(self):
        # Inicializar segmentos de carretera
        self._initialize_road_segments()
        
        # Inicializar semáforos
        self._initialize_traffic_lights()
        
        # Inicializar tráfico civil
        self._initialize_civilian_traffic()
        
        # Configurar clima inicial
        self._initialize_weather()
        
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
            vehicle_id = f"civilian_{i}"
            start_node = random.choice(all_nodes)
            node_data = self.street_graph.nodes[start_node]
            
            vehicle = VehicleState(
                vehicle_id=vehicle_id,
                vehicle_type=random.choice(["car", "motorcycle", "van"]),
                current_node=start_node,
                lat=node_data.get('lat', 0.0),
                lon=node_data.get('lon', 0.0),
                speed=random.uniform(30, 60),
                capacity=random.randint(400, 1000),
                fuel_level=random.uniform(20, 100),
                driver_type=random.choice(["normal", "aggressive", "cautious"])
            )
            
            self.vehicles[vehicle_id] = vehicle
            self.civilian_traffic[vehicle_id] = vehicle
    
    def _initialize_weather(self):
        """Inicializa el estado del clima"""
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
            
            # Vehículos
            "vehicles": {vid: {
                "type": v.vehicle_type,
                "lat": v.lat,
                "lon": v.lon,
                "speed": v.speed,
                "current_node": v.current_node,
                "next_node": v.next_node,
                "progress": v.progress,
                "fuel_level": v.fuel_level,
                "current_load": v.current_load,
                "is_active": v.is_active,
                "emergency_priority": v.emergency_priority
            } for vid, v in self.vehicles.items()},
            
            # Camiones de reparto específicamente
            "delivery_trucks": {tid: {
                "capacity": t.capacity,
                "current_load": t.current_load,
                "route": t.route,
                "driver_type": t.driver_type,
                "maintenance_needed": t.maintenance_needed
            } for tid, t in self.delivery_trucks.items()},
            
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
    
    def step(self):
        """Avanza la simulación un paso"""
        # Actualizar tiempo
        self.current_time += timedelta(seconds=self.time_step)
        
        # Actualizar componentes del entorno
        self._update_vehicle_positions()
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
    
    def _update_vehicle_positions(self):
        """Actualiza posiciones de todos los vehículos"""
        for vehicle in self.vehicles.values():
            if not vehicle.is_active:
                continue
                
            # Lógica de movimiento simplificada
            if vehicle.next_node is not None:
                # Calcular velocidad efectiva
                edge_id = (vehicle.current_node, vehicle.next_node)
                congestion_factor = self.congestion_matrix.get(edge_id, 1.0)
                
                effective_speed = vehicle.speed / congestion_factor
                
                # Actualizar progreso
                vehicle.progress += effective_speed * self.time_step * 0.0001
                
                # Verificar si llegó al siguiente nodo
                if vehicle.progress >= 1.0:
                    self._move_vehicle_to_next_node(vehicle)
    
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
        active_vehicles = sum(1 for v in self.vehicles.values() if v.is_active)
        self.system_metrics["total_vehicles"] = active_vehicles
        
        # Contar entregas activas
        active_deliveries = sum(1 for t in self.delivery_trucks.values() 
                              if t.is_active and t.current_load > 0)
        self.system_metrics["active_deliveries"] = active_deliveries
        
        # Calcular velocidad promedio
        if active_vehicles > 0:
            total_speed = sum(v.speed for v in self.vehicles.values() if v.is_active)
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