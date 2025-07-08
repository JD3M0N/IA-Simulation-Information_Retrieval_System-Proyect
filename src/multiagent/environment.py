import sys
import os
import time
import random
import asyncio
import math
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats

from src.multiagent.civilian_traffic import CivilianTrafficAgent
from src.multiagent.Civilian_enums import *
from src.multiagent.Environment_enums import *

# Imports
sys.path.append("src")
sys.path.append("src/weather")
sys.path.append("src/traffic_events")
sys.path.append("src/crawler")
sys.path.append("src/multi_agent")

# Import communication manager - DISABLED
try:
    # from src.multi_agent.communication import communication_manager
    COMMUNICATION_AVAILABLE = False
    print("Communication manager disabled - using standalone simulation")
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


class AdvancedRandomGenerator:
    """
    Generador de variables aleatorias usando métodos estadísticos fundamentales
    Implementa distribuciones usando transformada inversa y otros métodos clásicos
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Semilla para reproducibilidad de la simulación
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Cache para optimización de cálculos
        self._normal_cache = None
        self._has_cached_normal = False
    
    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """
        Distribución uniforme usando generador básico
        """
        return low + (high - low) * random.random()
    
    def exponential(self, lam: float = 1.0) -> float:
        """
        Distribución exponencial usando transformada inversa
        F(x) = 1 - e^(-λx)
        F^(-1)(u) = -ln(1-u)/λ
        """
        u = random.random()
        return -math.log(1 - u) / lam
    
    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        Distribución normal usando método Box-Muller
        Genera dos valores normales independientes, cachea uno
        """
        if self._has_cached_normal:
            self._has_cached_normal = False
            return self._normal_cache * sigma + mu
        
        # Método Box-Muller
        u1 = random.random()
        u2 = random.random()
        
        # Evitar log(0)
        while u1 == 0:
            u1 = random.random()
        
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        
        # Cachear z1 para la próxima llamada
        self._normal_cache = z1
        self._has_cached_normal = True
        
        return z0 * sigma + mu
    
    def poisson(self, lam: float) -> int:
        """
        Distribución de Poisson usando algoritmo de Knuth
        Para λ grandes usa aproximación normal
        """
        if lam > 30:
            # Aproximación normal para λ grandes
            return max(0, int(self.normal(lam, math.sqrt(lam)) + 0.5))
        
        # Algoritmo de Knuth para λ pequeñas
        L = math.exp(-lam)
        k = 0
        p = 1.0
        
        while p > L:
            k += 1
            p *= random.random()
        
        return k - 1
    
    def lognormal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        Distribución log-normal
        Si Y ~ Normal(μ, σ²), entonces X = e^Y ~ LogNormal(μ, σ²)
        """
        return math.exp(self.normal(mu, sigma))
    
    def gamma(self, shape: float, scale: float = 1.0) -> float:
        """
        Distribución Gamma usando método de Marsaglia-Tsang
        Para shape < 1, usa transformación
        """
        if shape < 1:
            # Para α < 1, usar transformación: Gamma(α) = Gamma(α+1) * U^(1/α)
            return self.gamma(shape + 1, scale) * (random.random() ** (1.0 / shape))
        
        # Método de Marsaglia-Tsang para α ≥ 1
        d = shape - 1.0/3.0
        c = 1.0 / math.sqrt(9.0 * d)
        
        while True:
            x = self.normal(0, 1)
            v = (1.0 + c * x) ** 3
            
            if v > 0:
                u = random.random()
                if u < 1 - 0.0331 * (x ** 4):
                    return d * v * scale
                elif math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)):
                    return d * v * scale
    
    def beta(self, alpha: float, beta_param: float) -> float:
        """
        Distribución Beta usando dos variables Gamma
        Beta(α, β) = Gamma(α) / (Gamma(α) + Gamma(β))
        """
        x = self.gamma(alpha)
        y = self.gamma(beta_param)
        return x / (x + y)
    
    def weibull(self, shape: float, scale: float = 1.0) -> float:
        """
        Distribución Weibull usando transformada inversa
        F^(-1)(u) = λ * (-ln(1-u))^(1/k)
        """
        u = random.random()
        return scale * ((-math.log(1 - u)) ** (1.0 / shape))
    
    def triangular(self, low: float, high: float, mode: float) -> float:
        """
        Distribución triangular usando transformada inversa
        """
        u = random.random()
        c = (mode - low) / (high - low)
        
        if u < c:
            return low + math.sqrt(u * (high - low) * (mode - low))
        else:
            return high - math.sqrt((1 - u) * (high - low) * (high - mode))
    
    def binomial(self, n: int, p: float) -> int:
        """
        Distribución binomial
        Para n grande usa aproximación normal
        """
        if n * p > 10 and n * (1 - p) > 10:
            # Aproximación normal con corrección de continuidad
            mu = n * p
            sigma = math.sqrt(n * p * (1 - p))
            return max(0, min(n, int(self.normal(mu, sigma) + 0.5)))
        
        # Método directo para n pequeño
        count = 0
        for _ in range(n):
            if random.random() < p:
                count += 1
        return count
    
    def choice_weighted(self, choices: List, weights: List[float]):
        """
        Selección aleatoria con pesos usando búsqueda binaria
        """
        if len(choices) != len(weights):
            raise ValueError("choices y weights deben tener la misma longitud")
        
        # Normalizar pesos
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(choices)
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Crear distribución acumulativa
        cumulative = []
        cumsum = 0
        for weight in normalized_weights:
            cumsum += weight
            cumulative.append(cumsum)
        
        # Seleccionar usando transformada inversa
        u = random.random()
        for i, cum_weight in enumerate(cumulative):
            if u <= cum_weight:
                return choices[i]
        
        return choices[-1]  # Fallback
    
    def pareto(self, scale: float, shape: float) -> float:
        """
        Distribución de Pareto usando transformada inversa
        F^(-1)(u) = x_m * (1-u)^(-1/α)
        """
        u = random.random()
        return scale * ((1 - u) ** (-1.0 / shape))


# Data classes and enums

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

    def __init__(self, street_graph: nx.Graph, num_vehicles:int = 20, random_seed: Optional[int] = None):
        """
        Args:
            street_graph: Grafo de NetworkX con la red de calles
            num_vehicles: Número de vehículos civiles
            random_seed: Semilla para reproducibilidad de la simulación
        """
        self.street_graph = street_graph
        self.num_vehicles = num_vehicles
        
        # Inicializar generador avanzado de variables aleatorias
        self.rng = AdvancedRandomGenerator(seed=random_seed)
        
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
        """Inicializa los semáforos con distribuciones realistas"""
        
        candidates = [n for n in self.street_graph.nodes() if len(list(self.street_graph.neighbors(n))) >= 4]
        # Usar distribución de Poisson para número de semáforos basado en intersecciones
        num_lights = min(self.rng.poisson(len(candidates) * 0.3), len(candidates))
        selected = random.sample(candidates, num_lights)

        # Identificar intersecciones importantes (nodos con múltiples conexiones)
        for node in selected:
            # Duración de verde usando distribución normal con límites
            green_duration = max(20, min(45, self.rng.normal(30, 5)))
            
            # Duración de rojo correlacionada con tráfico esperado
            red_duration = max(15, min(40, self.rng.normal(25, 4)))
            
            # Probabilidad de semáforo adaptativo basada en importancia del nodo
            node_degree = len(list(self.street_graph.neighbors(node)))
            adaptive_prob = min(0.8, node_degree / 10.0)  # Más conexiones = más probabilidad adaptativo
            
            self.traffic_lights[node] = TrafficLight(
                node_id=node,
                state=self.rng.choice_weighted(["green", "red"], [0.6, 0.4]),
                green_duration=green_duration,
                red_duration=red_duration,
                is_adaptive=self.rng.binomial(1, adaptive_prob) == 1
            )

    def _initialize_civilian_traffic(self):
        """Inicializa el tráfico civil con distribuciones realistas"""
        num_vehicles = self.num_vehicles
        all_nodes = list(self.street_graph.nodes())
        
        for i in range(num_vehicles):
            vehicle_id = "vehicle"+ str(i)
            start_node = random.choice(all_nodes)
            node_data = self.street_graph.nodes[start_node]
            
            # Comportamiento basado en distribución realista
            behavior_weights = [0.1, 0.6, 0.15, 0.1, 0.05]  # Conservative, Normal, Aggressive, Cautious, Reckless
            behavior = self.rng.choice_weighted(list(CivilianBehavior), behavior_weights)
            
            v = CivilianTrafficAgent(vehicle_id=vehicle_id, initial_position=[node_data["lat"], node_data["lon"]], initial_node=start_node, behavior=behavior)
            
            # NUEVO: Dar acceso al grafo de calles para cálculos de posición
            v.street_graph = self.street_graph
            v._street_graph = self.street_graph
            
            # Seleccionar destino usando distribución de distancias realista
            # Usar distribución gamma para distancias (cola larga = algunos viajes muy largos)
            max_distance = min(20, len(all_nodes) // 10)
            target_distance = int(self.rng.gamma(2, 2))  # Forma=2, escala=2
            target_distance = min(max_distance, max(1, target_distance))
            
            # Encontrar nodos a la distancia objetivo
            possible_targets = []
            try:
                distances = nx.single_source_shortest_path_length(self.street_graph, start_node, cutoff=target_distance+2)
                possible_targets = [n for n, d in distances.items() if abs(d - target_distance) <= 1 and n != start_node]
            except:
                possible_targets = [n for n in all_nodes if n != start_node]
            
            if not possible_targets:
                possible_targets = [n for n in all_nodes if n != start_node]
            
            target_node = random.choice(possible_targets)
            
            # Calcular ruta usando Dijkstra
            try:
                # NetworkX ya implementa Dijkstra con shortest_path
                route = nx.shortest_path(self.street_graph, source=start_node, target=target_node, weight='weight')
                
                # Asignar ruta y destino al vehículo con tipos realistas
                destination_weights = [0.35, 0.25, 0.15, 0.15, 0.1]  # trabajo, casa, compras, recreo, servicio
                destination_types = ["work", "home", "shopping", "recreation", "service"]
                destination_type = self.rng.choice_weighted(destination_types, destination_weights)
                v.assign_route_and_destination(route, target_node, destination_type)
                
                # NUEVO: Velocidad inicial con distribución normal truncada
                base_speed_factor = max(0.6, min(1.4, self.rng.normal(1.0, 0.2)))
                v.current_speed = max(v.base_speed * base_speed_factor, 15.0)  # Al menos 15 km/h
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
                        v.current_speed = max(v.base_speed * self.rng.normal(0.9, 0.1), 15.0)
                        v.movement_state = MovementState.MOVING
                    else:
                        # Si no hay vecinos, crear ruta mínima
                        route = [start_node]
                        v.assign_route_and_destination(route, start_node, "idle")
                except Exception as e:
                    print(f"Error asignando ruta alternativa para {vehicle_id}: {e}")
                    route = [start_node]
                    v.assign_route_and_destination(route, start_node, "idle")
            
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
            # Usar distribución beta para temperatura (más concentrada hacia el centro)
            temp_beta = self.rng.beta(2, 2)  # Beta(2,2) centrada en 0.5
            self.weather_state.temperature = 25 + temp_beta * 10  # 25-35°C
            
            # Humedad usando distribución triangular (más común en valores medios)
            self.weather_state.humidity = self.rng.triangular(60, 85, 72)
            
            # Condiciones climáticas con pesos estacionales
            summer_conditions = [WeatherCondition.CLEAR, WeatherCondition.CLOUDY, 
                               WeatherCondition.LIGHT_RAIN, WeatherCondition.EXTREME_HEAT]
            summer_weights = [0.6, 0.25, 0.1, 0.05]
            self.weather_state.condition = self.rng.choice_weighted(summer_conditions, summer_weights)
            
        elif current_month in [12, 1, 2]:  # Invierno
            # Temperatura con distribución normal truncada
            temp = max(10, min(25, self.rng.normal(17.5, 3)))
            self.weather_state.temperature = temp
            
            # Humedad con distribución log-normal (sesgo hacia valores bajos)
            humidity_raw = min(70, self.rng.lognormal(3.9, 0.3))  # ln(50) ≈ 3.9
            self.weather_state.humidity = max(40, humidity_raw)
            
            winter_conditions = [WeatherCondition.CLEAR, WeatherCondition.CLOUDY, 
                               WeatherCondition.FOG, WeatherCondition.LIGHT_RAIN]
            winter_weights = [0.4, 0.35, 0.15, 0.1]
            self.weather_state.condition = self.rng.choice_weighted(winter_conditions, winter_weights)
            
        else:  # Primavera/Otoño
            # Temperatura con distribución triangular
            self.weather_state.temperature = self.rng.triangular(15, 30, 22)
            
            # Humedad con distribución gamma (forma flexible)
            humidity_gamma = self.rng.gamma(2, 15)  # Forma=2, escala=15
            self.weather_state.humidity = min(80, max(50, humidity_gamma))
            
            transitional_conditions = [WeatherCondition.CLEAR, WeatherCondition.CLOUDY, 
                                     WeatherCondition.LIGHT_RAIN, WeatherCondition.STORM]
            transitional_weights = [0.45, 0.3, 0.2, 0.05]
            self.weather_state.condition = self.rng.choice_weighted(transitional_conditions, transitional_weights)
    
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
        """Genera evolución natural del clima usando distribuciones realistas"""
        # Cambios graduales en temperatura usando distribución normal centrada en 0
        temp_change = self.rng.normal(0, 1.5)  # Cambios más suaves que uniforme
        humidity_change = self.rng.normal(0, 3)  # Cambios graduales en humedad
        
        self.weather_state.temperature += temp_change
        self.weather_state.humidity += humidity_change
        
        # Limitar rangos usando funciones suaves
        self.weather_state.temperature = max(-10, min(50, self.weather_state.temperature))
        self.weather_state.humidity = max(0, min(100, self.weather_state.humidity))
        
        # Posibilidad de cambio de condición usando distribución exponencial para estabilidad
        change_rate = self.rng.exponential(10)  # Promedio de 10 pasos antes de cambio
        if change_rate < 1.0:  # Si el valor es bajo, cambiar condición
            # Usar pesos basados en la estación actual
            current_month = self.current_time.month
            if current_month in [6, 7, 8]:  # Verano
                conditions = [WeatherCondition.CLEAR, WeatherCondition.CLOUDY, 
                            WeatherCondition.LIGHT_RAIN, WeatherCondition.EXTREME_HEAT]
                weights = [0.6, 0.25, 0.1, 0.05]
            elif current_month in [12, 1, 2]:  # Invierno
                conditions = [WeatherCondition.CLEAR, WeatherCondition.CLOUDY, 
                            WeatherCondition.FOG, WeatherCondition.LIGHT_RAIN]
                weights = [0.4, 0.35, 0.15, 0.1]
            else:  # Primavera/Otoño
                conditions = [WeatherCondition.CLEAR, WeatherCondition.CLOUDY, 
                            WeatherCondition.LIGHT_RAIN, WeatherCondition.STORM]
                weights = [0.45, 0.3, 0.2, 0.05]
            
            self.weather_state.condition = self.rng.choice_weighted(conditions, weights)
        
        # Actualizar otros parámetros con distribuciones apropiadas
        if self.weather_state.condition in [WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN]:
            # Precipitación usando distribución gamma (valores bajos más comunes)
            self.weather_state.precipitation = self.rng.gamma(1.5, 8)  # Media ~12mm/h
        elif self.weather_state.condition == WeatherCondition.STORM:
            # Tormentas con distribución log-normal (eventos extremos)
            self.weather_state.precipitation = self.rng.lognormal(3, 0.5)  # Valores altos ocasionales
            self.weather_state.wind_speed = max(30, self.rng.weibull(2, 60))  # Weibull para vientos extremos
        else:
            self.weather_state.precipitation = 0.0
            # Viento normal con distribución gamma
            self.weather_state.wind_speed = self.rng.gamma(2, 8)  # Vientos suaves promedio
    
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
        
        # Generar eventos de tráfico con distribución de eventos más realista
        # Usar distribución de Poisson para eventos por hora
        events_per_hour = 2.0  # Promedio de eventos por hora
        event_probability = events_per_hour * (self.time_step / 3600.0)  # Convertir a probabilidad por step
        
        if self.rng.poisson(event_probability) > 0:
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
        
        # Asignar nuevo destino con distribución realista
        neighbors = list(self.street_graph.neighbors(vehicle.current_node))
        if neighbors:
            # Usar distribución de Pareto para modelar preferencia por rutas principales
            # Los vehículos tienden a preferir ciertas rutas (80/20 rule)
            if len(neighbors) > 1:
                # Calcular pesos basados en tipo de vía y tráfico histórico
                weights = []
                for neighbor in neighbors:
                    edge_data = self.street_graph.get_edge_data(vehicle.current_node, neighbor)
                    road_type = edge_data.get('highway', 'residential') if edge_data else 'residential'
                    
                    # Pesos por tipo de vía (principales más probables)
                    type_weights = {
                        'motorway': 5.0,
                        'trunk': 4.0,
                        'primary': 3.0,
                        'secondary': 2.0,
                        'tertiary': 1.5,
                        'residential': 1.0,
                        'service': 0.5
                    }
                    
                    base_weight = type_weights.get(road_type, 1.0)
                    # Añadir factor de congestión (menos peso si hay mucho tráfico)
                    edge_id = (vehicle.current_node, neighbor)
                    congestion_factor = 1.0 / (1.0 + self.congestion_matrix.get(edge_id, 1.0))
                    
                    weights.append(base_weight * congestion_factor)
                
                vehicle.next_node = self.rng.choice_weighted(neighbors, weights)
            else:
                vehicle.next_node = neighbors[0]
                
            vehicle.progress = 0.0
            
            # Incrementar congestión en nueva arista
            new_edge = (vehicle.current_node, vehicle.next_node)
            if new_edge in self.road_segments:
                self.road_segments[new_edge].current_vehicles += 1
        else:
            vehicle.next_node = None
    
    def _generate_random_traffic_event(self):
        """Genera un evento de tráfico aleatorio usando distribuciones realistas"""
        # Tipos de eventos con probabilidades realistas
        event_types = list(TrafficEventType)
        event_weights = [0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02]  # Pesos según frecuencia real
        
        # Asegurar que tenemos pesos para todos los tipos
        if len(event_weights) < len(event_types):
            event_weights.extend([0.01] * (len(event_types) - len(event_weights)))
        elif len(event_weights) > len(event_types):
            event_weights = event_weights[:len(event_types)]
        
        event_type = self.rng.choice_weighted(event_types, event_weights)
        
        # Seleccionar ubicación con sesgo hacia intersecciones importantes
        all_nodes = list(self.street_graph.nodes())
        
        # Calcular importancia de nodos (más conexiones = más probable)
        node_importance = {}
        for node in all_nodes:
            degree = len(list(self.street_graph.neighbors(node)))
            node_importance[node] = degree
        
        # Normalizar importancias y usar como pesos
        max_importance = max(node_importance.values()) if node_importance else 1
        node_weights = [node_importance[node] / max_importance for node in all_nodes]
        
        location_node = self.rng.choice_weighted(all_nodes, node_weights)
        node_data = self.street_graph.nodes[location_node]
        location = (node_data.get('lat', 0.0), node_data.get('lon', 0.0))
        
        # Obtener calles afectadas
        affected_streets = []
        for neighbor in self.street_graph.neighbors(location_node):
            edge_data = self.street_graph.get_edge_data(location_node, neighbor)
            if edge_data and 'name' in edge_data:
                affected_streets.append(edge_data['name'])
        
        # Severidad usando distribución gamma (más eventos leves que severos)
        severity_raw = self.rng.gamma(1.5, 1.5)  # Sesgo hacia valores bajos
        severity = max(1, min(5, int(severity_raw + 0.5)))
        
        # Duración usando distribución log-normal (mayoría cortos, algunos muy largos)
        duration_minutes = max(15, int(self.rng.lognormal(3.5, 0.8)))  # Media ~45 min
        
        # Factor de impacto correlacionado con severidad usando distribución exponencial
        base_impact = 1.0 + self.rng.exponential(severity * 0.3)
        impact_factor = min(5.0, base_impact)  # Limitar impacto máximo
        
        # Crear evento
        event = TrafficEvent(
            event_id=f"event_{int(time.time())}_{self.rng.binomial(9999, 0.5)}",
            event_type=event_type,
            location=location,
            affected_streets=affected_streets,
            severity=severity,
            start_time=self.current_time,
            estimated_duration=timedelta(minutes=duration_minutes),
            description=f"Evento de {event_type.value} en {location} (severidad {severity})",
            impact_factor=impact_factor
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
            "total_road_segments": len(self.road_segments),
            "total_traffic_lights": len(self.traffic_lights),
            "active_events": len(self.active_events),
            "weather_condition": self.weather_state.condition.value if hasattr(self.weather_state.condition, 'value') else str(self.weather_state.condition),
            "average_speed": sum(v.current_speed for v in self.vehicles.values()) / len(self.vehicles) if self.vehicles else 0,
            "active_traffic_violations": sum(v.traffic_violations for v in self.vehicles.values()),
            "emergency_responses": sum(v.emergency_responses for v in self.vehicles.values())
        }
    
    def generate_delivery_demand(self, base_demand: float = 50.0) -> int:
        """
        Genera demanda de paquetes usando distribución de Poisson
        con variaciones por hora del día y día de la semana
        
        Args:
            base_demand: Demanda base promedio por período
            
        Returns:
            Número de paquetes a entregar
        """
        # Factores de ajuste temporal
        hour = self.current_time.hour
        weekday = self.current_time.weekday()
        
        # Factor por hora del día (picos en horas comerciales)
        if 9 <= hour <= 17:  # Horas comerciales
            hour_factor = 1.5
        elif 18 <= hour <= 20:  # Horas pico de entrega residencial
            hour_factor = 2.0
        elif 7 <= hour <= 8 or 21 <= hour <= 22:  # Horas de transición
            hour_factor = 1.2
        else:  # Horas nocturnas
            hour_factor = 0.3
        
        # Factor por día de la semana
        if weekday < 5:  # Lunes a viernes
            weekday_factor = 1.0
        elif weekday == 5:  # Sábado
            weekday_factor = 1.3
        else:  # Domingo
            weekday_factor = 0.7
        
        # Demanda ajustada
        adjusted_demand = base_demand * hour_factor * weekday_factor
        
        # Generar usando Poisson
        return self.rng.poisson(adjusted_demand)
    
    def generate_delivery_time(self, distance_km: float, traffic_factor: float = 1.0) -> float:
        """
        Genera tiempo de entrega usando distribución normal con factores de tráfico
        
        Args:
            distance_km: Distancia en kilómetros
            traffic_factor: Factor de tráfico (1.0 = normal, >1.0 = más lento)
            
        Returns:
            Tiempo estimado en minutos
        """
        # Tiempo base: 30 km/h promedio en ciudad
        base_time = (distance_km / 30.0) * 60.0  # minutos
        
        # Variabilidad usando distribución normal (±20% de variación)
        time_variation = self.rng.normal(1.0, 0.2)
        time_variation = max(0.5, min(2.0, time_variation))  # Limitar variación
        
        # Aplicar factores
        total_time = base_time * time_variation * traffic_factor
        
        # Añadir retrasos ocasionales usando distribución exponencial
        if self.rng.binomial(1, 0.15):  # 15% probabilidad de retraso
            delay = self.rng.exponential(10)  # Retraso promedio de 10 min
            total_time += delay
        
        return max(5.0, total_time)  # Mínimo 5 minutos
    
    def generate_vehicle_failure_probability(self, vehicle_age_years: float = 3.0, 
                                           maintenance_score: float = 0.8) -> bool:
        """
        Genera probabilidad de fallo de vehículo usando distribución Weibull
        
        Args:
            vehicle_age_years: Edad del vehículo en años
            maintenance_score: Puntaje de mantenimiento (0-1, 1=perfecto)
            
        Returns:
            True si el vehículo falla
        """
        # Parámetros Weibull para modelar fallos
        shape = 1.5  # Factor de forma (>1 indica mayor probabilidad con edad)
        scale = 10.0 / vehicle_age_years  # Factor de escala ajustado por edad
        
        # Tiempo hasta fallo (en años)
        time_to_failure = self.rng.weibull(shape, scale)
        
        # Ajustar por mantenimiento
        time_to_failure *= maintenance_score
        
        # Convertir a probabilidad por día
        daily_failure_prob = 1.0 / (time_to_failure * 365.25)
        
        # Generar fallo
        return self.rng.binomial(1, min(0.1, daily_failure_prob)) == 1
    
    def generate_fuel_consumption(self, distance_km: float, vehicle_type: str = "truck",
                                weather_condition: Optional[WeatherCondition] = None) -> float:
        """
        Genera consumo de combustible usando distribución gamma
        
        Args:
            distance_km: Distancia recorrida
            vehicle_type: Tipo de vehículo
            weather_condition: Condición climática
            
        Returns:
            Litros de combustible consumidos
        """
        # Consumo base por tipo de vehículo (litros/100km)
        base_consumption = {
            "truck": 25.0,
            "van": 15.0,
            "car": 8.0,
            "motorcycle": 4.0
        }
        
        base_rate = base_consumption.get(vehicle_type, 15.0)
        
        # Factor climático
        weather_factor = 1.0
        if weather_condition:
            weather_factors = {
                WeatherCondition.CLEAR: 1.0,
                WeatherCondition.LIGHT_RAIN: 1.1,
                WeatherCondition.HEAVY_RAIN: 1.25,
                WeatherCondition.STORM: 1.4,
                WeatherCondition.FOG: 1.15,
                WeatherCondition.EXTREME_HEAT: 1.2,
                WeatherCondition.SNOW: 1.5
            }
            weather_factor = weather_factors.get(weather_condition, 1.0)
        
        # Variabilidad usando distribución gamma (sesgo hacia valores bajos)
        consumption_variation = self.rng.gamma(2, 0.4)  # Media ~0.8, sesgo hacia eficiencia
        
        # Calcular consumo total
        consumption_per_100km = base_rate * weather_factor * consumption_variation
        total_consumption = (distance_km / 100.0) * consumption_per_100km
        
        return max(0.1, total_consumption)
    
    def simulate_traffic_congestion_factor(self, hour: int, road_type: str = "residential") -> float:
        """
        Simula factor de congestión usando distribuciones apropiadas
        
        Args:
            hour: Hora del día (0-23)
            road_type: Tipo de vía
            
        Returns:
            Factor de congestión (1.0 = normal, >1.0 = congestionado)
        """
        # Factores base por hora usando distribución beta para curvas realistas
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Horas pico
            # Beta(2,5) da una distribución sesgada hacia valores altos
            peak_factor = 1.0 + self.rng.beta(2, 5) * 3.0  # 1.0 a 4.0
        elif 10 <= hour <= 16:  # Horas normales
            # Beta(5,2) da una distribución sesgada hacia valores medios
            peak_factor = 1.0 + self.rng.beta(5, 2) * 1.5  # 1.0 a 2.5
        elif 20 <= hour <= 22:  # Noche temprana
            peak_factor = 1.0 + self.rng.beta(3, 7) * 1.0  # 1.0 a 2.0
        else:  # Madrugada
            peak_factor = 1.0 + self.rng.beta(8, 2) * 0.3  # 1.0 a 1.3
        
        # Ajustar por tipo de vía
        road_factors = {
            "motorway": 0.8,    # Menos congestión relativa
            "primary": 1.0,     # Congestión normal
            "secondary": 1.1,   # Algo más de congestión
            "residential": 1.2,  # Más congestión local
            "service": 0.9      # Menos tráfico
        }
        
        road_factor = road_factors.get(road_type, 1.0)
        
        return peak_factor * road_factor