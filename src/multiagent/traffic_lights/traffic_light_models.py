"""
Modelos de datos para el sistema de semáforos
Define las estructuras de datos centrales para el manejo de semáforos
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging


class TrafficLightPhase(Enum):
    """Fases del semáforo"""
    GREEN = "green"
    YELLOW = "yellow" 
    RED = "red"
    FLASHING_RED = "flashing_red"
    FLASHING_YELLOW = "flashing_yellow"
    OFF = "off"


class IntersectionType(Enum):
    """Tipos de intersección"""
    SIMPLE_CROSS = "simple_cross"  # Cruz simple
    T_JUNCTION = "t_junction"      # Intersección en T
    ROUNDABOUT = "roundabout"      # Rotonda
    MULTI_WAY = "multi_way"        # Múltiples direcciones
    HIGHWAY_RAMP = "highway_ramp"  # Rampa de autopista


class TrafficDirection(Enum):
    """Direcciones del tráfico"""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    SOUTHEAST = "southeast" 
    SOUTHWEST = "southwest"


class PriorityLevel(Enum):
    """Niveles de prioridad"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    EMERGENCY = 4
    CRITICAL = 5


@dataclass
class TrafficFlow:
    """Información del flujo de tráfico en una dirección"""
    direction: TrafficDirection
    vehicle_count: int = 0
    average_speed: float = 0.0
    queue_length: int = 0
    wait_time: float = 0.0
    density: float = 0.0
    flow_rate: float = 0.0  # vehículos por minuto
    emergency_vehicles: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PhaseConfiguration:
    """Configuración de una fase del semáforo"""
    phase: TrafficLightPhase
    duration: float  # segundos
    allowed_directions: List[TrafficDirection]
    priority: PriorityLevel = PriorityLevel.NORMAL
    min_duration: float = 5.0
    max_duration: float = 120.0
    is_adaptive: bool = True


@dataclass
class TrafficLightData:
    """Datos completos de un semáforo"""
    # Identificación
    light_id: str
    node_id: int
    intersection_id: str = ""
    
    # Ubicación
    latitude: float = 0.0
    longitude: float = 0.0
    
    # Estado actual
    current_phase: TrafficLightPhase = TrafficLightPhase.RED
    phase_start_time: datetime = field(default_factory=datetime.now)
    time_in_phase: float = 0.0
    
    # Configuración de fases
    phase_config: Dict[TrafficLightPhase, PhaseConfiguration] = field(default_factory=dict)
    cycle_order: List[TrafficLightPhase] = field(
        default_factory=lambda: [
            TrafficLightPhase.GREEN,
            TrafficLightPhase.YELLOW, 
            TrafficLightPhase.RED
        ]
    )
    
    # Control adaptativo
    is_adaptive: bool = True
    adaptation_factor: float = 1.0
    min_cycle_time: float = 60.0
    max_cycle_time: float = 180.0
    
    # Estado operacional
    is_operational: bool = True
    maintenance_mode: bool = False
    emergency_override: bool = False
    manual_override: bool = False
    
    # Métricas
    cycles_completed: int = 0
    total_vehicles_served: int = 0
    average_wait_time: float = 0.0
    efficiency_score: float = 0.0
    
    # Historial
    phase_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_current_phase_duration(self) -> float:
        """Obtiene la duración configurada para la fase actual"""
        if self.current_phase in self.phase_config:
            return self.phase_config[self.current_phase].duration
        return 30.0  # Por defecto
    
    def get_next_phase(self) -> TrafficLightPhase:
        """Obtiene la siguiente fase en el ciclo"""
        try:
            current_index = self.cycle_order.index(self.current_phase)
            next_index = (current_index + 1) % len(self.cycle_order)
            return self.cycle_order[next_index]
        except ValueError:
            return TrafficLightPhase.RED
    
    def add_phase_record(self, phase: TrafficLightPhase, duration: float, 
                        traffic_served: int = 0):
        """Añade un registro al historial de fases"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase.value,
            "duration": duration,
            "traffic_served": traffic_served,
            "efficiency": self.efficiency_score
        }
        self.phase_history.append(record)
        
        # Mantener solo los últimos 100 registros
        if len(self.phase_history) > 100:
            self.phase_history.pop(0)


@dataclass
class IntersectionData:
    """Datos completos de una intersección"""
    # Identificación
    intersection_id: str
    node_id: int
    
    # Ubicación y geometría
    latitude: float
    longitude: float
    intersection_type: IntersectionType = IntersectionType.SIMPLE_CROSS
    
    # Semáforos controlados
    traffic_lights: Dict[str, TrafficLightData] = field(default_factory=dict)
    
    # Flujos de tráfico
    traffic_flows: Dict[TrafficDirection, TrafficFlow] = field(default_factory=dict)
    
    # Configuración operacional
    coordination_enabled: bool = True
    priority_preemption: bool = True
    pedestrian_phases: bool = False
    
    # Estado del entorno
    weather_condition: str = "clear"
    visibility: float = 1.0  # 0-1
    road_conditions: str = "good"
    
    # Eventos especiales
    active_events: List[Dict[str, Any]] = field(default_factory=list)
    emergency_vehicles_present: bool = False
    
    # Métricas de rendimiento
    total_throughput: int = 0
    average_delay: float = 0.0
    level_of_service: str = "A"  # A-F rating
    
    # Optimización
    optimization_enabled: bool = True
    last_optimization: datetime = field(default_factory=datetime.now)
    optimization_interval: timedelta = timedelta(minutes=15)
    
    def get_total_vehicle_count(self) -> int:
        """Obtiene el conteo total de vehículos en todas las direcciones"""
        return sum(flow.vehicle_count for flow in self.traffic_flows.values())
    
    def get_max_queue_length(self) -> int:
        """Obtiene la longitud máxima de cola en cualquier dirección"""
        return max(flow.queue_length for flow in self.traffic_flows.values()) if self.traffic_flows else 0
    
    def has_emergency_vehicles(self) -> bool:
        """Verifica si hay vehículos de emergencia presentes"""
        return any(flow.emergency_vehicles > 0 for flow in self.traffic_flows.values())
    
    def update_traffic_flow(self, direction: TrafficDirection, 
                          vehicle_count: int, queue_length: int = 0,
                          average_speed: float = 0.0, emergency_count: int = 0):
        """Actualiza el flujo de tráfico para una dirección"""
        if direction not in self.traffic_flows:
            self.traffic_flows[direction] = TrafficFlow(direction)
        
        flow = self.traffic_flows[direction]
        flow.vehicle_count = vehicle_count
        flow.queue_length = queue_length
        flow.average_speed = average_speed
        flow.emergency_vehicles = emergency_count
        flow.last_updated = datetime.now()
        
        # Calcular densidad y tasa de flujo
        if average_speed > 0:
            flow.density = vehicle_count / average_speed
            flow.flow_rate = vehicle_count * (60.0 / max(1.0, average_speed))


@dataclass
class TrafficLightEvent:
    """Evento relacionado con semáforos"""
    event_id: str
    timestamp: datetime
    event_type: str  # "phase_change", "override", "emergency", "failure"
    intersection_id: str
    light_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    severity: PriorityLevel = PriorityLevel.NORMAL
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class TrafficLightPerformanceMetrics:
    """Métricas de rendimiento de semáforos"""
    intersection_id: str
    measurement_period: timedelta
    start_time: datetime
    end_time: datetime
    
    # Métricas de flujo
    total_vehicles_processed: int = 0
    average_vehicles_per_hour: float = 0.0
    peak_hour_volume: int = 0
    
    # Métricas de tiempo
    average_wait_time: float = 0.0
    max_wait_time: float = 0.0
    average_cycle_time: float = 0.0
    
    # Métricas de eficiencia
    green_utilization: float = 0.0  # % de tiempo verde utilizado efectivamente
    intersection_efficiency: float = 0.0  # 0-1
    level_of_service_score: float = 0.0
    
    # Métricas ambientales
    estimated_fuel_consumption: float = 0.0
    estimated_emissions: float = 0.0
    
    # Eventos
    emergency_preemptions: int = 0
    system_failures: int = 0
    manual_interventions: int = 0
    
    def calculate_efficiency_score(self) -> float:
        """Calcula un score de eficiencia general"""
        # Fórmula simplificada que combina varios factores
        wait_factor = max(0, 1 - (self.average_wait_time / 120.0))  # Normalizar a 2 min máx
        utilization_factor = self.green_utilization
        los_factor = self.level_of_service_score
        
        # Promedio ponderado
        efficiency = (wait_factor * 0.4 + utilization_factor * 0.3 + los_factor * 0.3)
        return min(1.0, max(0.0, efficiency))
