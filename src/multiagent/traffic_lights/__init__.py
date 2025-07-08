"""
Sistema de Semáforos Modular para el Entorno Multi-Agente
Módulo independiente para manejo de semáforos con lógica inteligente
"""

from .traffic_light_agent import TrafficLightAgent
from .traffic_light_controller import TrafficLightController
from .traffic_light_optimization import TrafficLightOptimizer
from .traffic_light_models import (
    TrafficLightData, IntersectionData, TrafficFlow,
    TrafficLightPhase, TrafficDirection, PriorityLevel
)
from .traffic_light_utils import calculate_phase_timing, detect_traffic_patterns
from .integration import TrafficLightIntegration, initialize_traffic_light_system, get_traffic_light_data_for_vehicle
from .server_integration import (
    ServerTrafficLightManager, server_traffic_manager,
    initialize_server_traffic_lights, get_server_traffic_lights_data,
    modify_server_traffic_light, get_server_traffic_metrics
)

# Utilidades de testing (importación opcional)
try:
    from .testing_utils import TrafficLightTester, TrafficLightDebugger, quick_test, quick_debug
    TESTING_UTILS_AVAILABLE = True
except ImportError:
    TESTING_UTILS_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Sistema Multi-Agente CVRP - Traffic Lights Module"

__all__ = [
    # Agente principal
    "TrafficLightAgent",
    
    # Controlador centralizado
    "TrafficLightController",
    
    # Optimización
    "TrafficLightOptimizer",
    
    # Modelos de datos
    "TrafficLightData",
    "IntersectionData", 
    "TrafficFlow",
    "TrafficLightPhase",
    "TrafficDirection",
    "PriorityLevel",
    
    # Utilidades
    "calculate_phase_timing",
    "detect_traffic_patterns",
    
    # Integración básica
    "TrafficLightIntegration",
    "initialize_traffic_light_system",
    "get_traffic_light_data_for_vehicle",
    
    # Integración para servidor
    "ServerTrafficLightManager",
    "server_traffic_manager",
    "initialize_server_traffic_lights",
    "get_server_traffic_lights_data",
    "modify_server_traffic_light",
    "get_server_traffic_metrics"
]
