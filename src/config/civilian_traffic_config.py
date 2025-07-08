"""
Configuración y Mejores Prácticas para CivilianTrafficAgent
Guía para la integración del agente de tráfico civil con Environment
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import random

from multiagent.civilian_traffic import CivilianTrafficAgent, CivilianBehavior
from multiagent.environment import Environment, WeatherCondition, TrafficEventType

@dataclass
class CivilianTrafficConfig:
    """Configuración para la simulación de tráfico civil"""
    
    # Parámetros de población
    num_civilian_vehicles: int = 50
    behavior_distribution: Dict[CivilianBehavior, float] = None
    vehicle_type_distribution: Dict[str, float] = None
    
    # Parámetros de simulación
    simulation_area: Tuple[float, float, float] = (40.7128, -74.0060, 5.0)  # lat, lon, radius_km
    update_frequency: float = 1.0  # segundos
    
    # Parámetros de comportamiento
    cooperation_enabled: bool = True
    emergency_response_enabled: bool = True
    weather_adaptation_enabled: bool = True
    
    # Parámetros de eventos
    traffic_event_probability: float = 0.1  # Por paso de simulación
    weather_change_probability: float = 0.05
    emergency_vehicle_probability: float = 0.02
    
    def __post_init__(self):
        """Inicializa valores por defecto"""
        if self.behavior_distribution is None:
            self.behavior_distribution = {
                CivilianBehavior.CONSERVATIVE: 0.15,
                CivilianBehavior.NORMAL: 0.50,
                CivilianBehavior.AGGRESSIVE: 0.20,
                CivilianBehavior.CAUTIOUS: 0.10,
                CivilianBehavior.RECKLESS: 0.05
            }
        
        if self.vehicle_type_distribution is None:
            self.vehicle_type_distribution = {
                "car": 0.70,
                "motorcycle": 0.15,
                "van": 0.10,
                "bus": 0.05
            }

class CivilianTrafficManager:
    """
    Administrador para múltiples agentes de tráfico civil
    Facilita la creación, configuración y gestión de agentes
    """
    
    def __init__(self, config: CivilianTrafficConfig):
        self.config = config
        self.agents: List[CivilianTrafficAgent] = []
        self.agent_statistics = {}
        
    def create_civilian_agents(self, street_graph, environment: Environment) -> List[CivilianTrafficAgent]:
        """
        Crea múltiples agentes de tráfico civil según la configuración
        """
        agents = []
        
        for i in range(self.config.num_civilian_vehicles):
            # Seleccionar comportamiento según distribución
            behavior = self._select_behavior()
            
            # Seleccionar tipo de vehículo
            vehicle_type = self._select_vehicle_type()
            
            # Generar posición inicial
            initial_position = self._generate_initial_position()
            
            # Crear agente
            agent = CivilianTrafficAgent(
                vehicle_id=f"civilian_{i}",
                initial_position=initial_position,
                behavior=behavior
            )
            
            # Configurar parámetros específicos
            agent.vehicle_type = vehicle_type
            agent.current_node = self._select_initial_node(street_graph)
            
            # Establecer destino inicial
            destination = self._select_destination_node(street_graph, agent.current_node)
            route = self._generate_route(street_graph, agent.current_node, destination)
            
            agent.set_destination_and_route(destination, route)
            
            agents.append(agent)
        
        self.agents = agents
        return agents
    
    def _select_behavior(self) -> CivilianBehavior:
        """Selecciona comportamiento según distribución configurada"""
        behaviors = list(self.config.behavior_distribution.keys())
        probabilities = list(self.config.behavior_distribution.values())
        
        return random.choices(behaviors, weights=probabilities)[0]
    
    def _select_vehicle_type(self) -> str:
        """Selecciona tipo de vehículo según distribución"""
        vehicle_types = list(self.config.vehicle_type_distribution.keys())
        probabilities = list(self.config.vehicle_type_distribution.values())
        
        return random.choices(vehicle_types, weights=probabilities)[0]
    
    def _generate_initial_position(self) -> Tuple[float, float]:
        """Genera posición inicial dentro del área de simulación"""
        center_lat, center_lon, radius = self.config.simulation_area
        
        # Generar posición aleatoria dentro del radio
        angle = random.uniform(0, 2 * 3.14159)
        distance = random.uniform(0, radius)
        
        # Aproximación simple para coordenadas
        lat_offset = distance * 0.009 * random.cos(angle)  # ~1km = 0.009 degrees
        lon_offset = distance * 0.009 * random.sin(angle)
        
        return (center_lat + lat_offset, center_lon + lon_offset)
    
    def _select_initial_node(self, street_graph) -> int:
        """Selecciona nodo inicial aleatorio"""
        return random.choice(list(street_graph.nodes))
    
    def _select_destination_node(self, street_graph, current_node: int) -> int:
        """Selecciona nodo destino diferente al actual"""
        nodes = list(street_graph.nodes)
        nodes.remove(current_node)
        return random.choice(nodes)
    
    def _generate_route(self, street_graph, start: int, end: int) -> List[int]:
        """Genera ruta simple entre dos nodos"""
        try:
            import networkx as nx
            path = nx.shortest_path(street_graph, start, end)
            return path
        except:
            # Fallback a ruta directa
            return [start, end]
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de todos los agentes"""
        if not self.agents:
            return {}
        
        stats = {
            "total_agents": len(self.agents),
            "behavior_distribution": {},
            "vehicle_type_distribution": {},
            "average_speed": 0.0,
            "total_distance": 0.0,
            "total_stops": 0,
            "total_route_changes": 0,
            "total_emergency_responses": 0
        }
        
        # Calcular distribuciones
        for agent in self.agents:
            behavior = agent.behavior.value
            vehicle_type = agent.vehicle_type
            
            stats["behavior_distribution"][behavior] = stats["behavior_distribution"].get(behavior, 0) + 1
            stats["vehicle_type_distribution"][vehicle_type] = stats["vehicle_type_distribution"].get(vehicle_type, 0) + 1
            
            # Acumular métricas
            agent_status = agent.get_vehicle_status()
            metrics = agent_status["metrics"]
            
            stats["average_speed"] += agent_status["current_speed"]
            stats["total_distance"] += metrics["distance_traveled"]
            stats["total_stops"] += metrics["stops_count"]
            stats["total_route_changes"] += metrics["route_changes"]
            stats["total_emergency_responses"] += metrics["emergency_responses"]
        
        # Calcular promedio de velocidad
        if len(self.agents) > 0:
            stats["average_speed"] /= len(self.agents)
        
        return stats
    
    def update_agent_destinations(self, street_graph):
        """Actualiza destinos para agentes que han llegado"""
        for agent in self.agents:
            if not agent.has_destination or agent.movement_state.value == "parado":
                # Asignar nuevo destino
                new_destination = self._select_destination_node(street_graph, agent.current_node)
                new_route = self._generate_route(street_graph, agent.current_node, new_destination)
                agent.set_destination_and_route(new_destination, new_route)

class CivilianTrafficIntegration:
    """
    Clase de integración que facilita el uso conjunto de Environment y CivilianTrafficAgent
    """
    
    @staticmethod
    def create_realistic_traffic_scenario(street_graph, num_vehicles: int = 30) -> Tuple[Environment, List[CivilianTrafficAgent]]:
        """
        Crea un escenario realista de tráfico con Environment y agentes civiles
        """
        # Crear configuración realista
        config = CivilianTrafficConfig(
            num_civilian_vehicles=num_vehicles,
            behavior_distribution={
                CivilianBehavior.CONSERVATIVE: 0.20,
                CivilianBehavior.NORMAL: 0.45,
                CivilianBehavior.AGGRESSIVE: 0.25,
                CivilianBehavior.CAUTIOUS: 0.08,
                CivilianBehavior.RECKLESS: 0.02
            },
            traffic_event_probability=0.15,
            weather_change_probability=0.08
        )
        
        # Crear Environment
        environment = Environment(street_graph, num_vehicles=num_vehicles)
        
        # Crear agentes civiles
        manager = CivilianTrafficManager(config)
        agents = manager.create_civilian_agents(street_graph, environment)
        
        return environment, agents
    
    @staticmethod
    def setup_environment_vehicle_sync(environment: Environment, agents: List[CivilianTrafficAgent]):
        """
        Configura sincronización entre Environment y agentes de vehículos
        """
        # Registrar vehículos en el environment
        for agent in agents:
            vehicle_state = {
                "vehicle_id": agent.agent_id,
                "vehicle_type": agent.vehicle_type,
                "current_node": agent.current_node,
                "lat": agent.position[0],
                "lon": agent.position[1],
                "speed": agent.current_speed,
                "capacity": agent.capacity,
                "current_load": 0,  # Vehículos civiles generalmente no cargan
                "fuel_level": agent.fuel_level,
                "driver_type": agent.behavior.value,
                "route": agent.route if hasattr(agent, 'route') else [],
                "progress": agent.progress,
                "is_active": True,
                "emergency_priority": False
            }
            
            environment.vehicles[agent.agent_id] = vehicle_state
    
    @staticmethod
    def process_environment_feedback(environment: Environment, agents: List[CivilianTrafficAgent]):
        """
        Procesa retroalimentación del Environment hacia los agentes
        """
        for agent in agents:
            # Actualizar información del agente en el environment
            if agent.agent_id in environment.vehicles:
                vehicle_state = environment.vehicles[agent.agent_id]
                
                # Sincronizar posición
                vehicle_state["lat"] = agent.position[0]
                vehicle_state["lon"] = agent.position[1]
                vehicle_state["speed"] = agent.current_speed
                vehicle_state["current_node"] = agent.current_node
                vehicle_state["fuel_level"] = agent.fuel_level
                vehicle_state["progress"] = agent.progress
                
                # Actualizar ruta si ha cambiado
                if hasattr(agent, 'route'):
                    vehicle_state["route"] = agent.route

# Ejemplo de uso práctico
def example_usage():
    """
    Ejemplo de uso completo del sistema integrado
    """
    import networkx as nx
    import asyncio
    
    async def run_integrated_simulation():
        # Crear grafo de calles
        street_graph = nx.grid_2d_graph(5, 5)
        street_graph = nx.convert_node_labels_to_integers(street_graph)
        
        # Crear escenario realista
        environment, agents = CivilianTrafficIntegration.create_realistic_traffic_scenario(
            street_graph, num_vehicles=20
        )
        
        # Configurar sincronización
        CivilianTrafficIntegration.setup_environment_vehicle_sync(environment, agents)
        
        # Ejecutar simulación
        for step in range(50):
            # Actualizar environment
            environment.update_state()
            environment_state = environment.get_perception_for_agent("civilian")
            
            # Procesar cada agente
            for agent in agents:
                perception = await agent.perceive(environment_state)
                decision = await agent.decide(perception)
                await agent.act(decision)
            
            # Procesar retroalimentación
            CivilianTrafficIntegration.process_environment_feedback(environment, agents)
            
            # Generar eventos ocasionales
            if random.random() < 0.1:
                environment.generate_traffic_event(
                    event_type=random.choice(list(TrafficEventType)),
                    location=(40.7128, -74.0060),
                    severity=random.randint(1, 4)
                )
            
            await asyncio.sleep(0.1)
        
        # Obtener estadísticas finales
        manager = CivilianTrafficManager(CivilianTrafficConfig())
        manager.agents = agents
        stats = manager.get_agent_statistics()
        
        print("Estadísticas finales:")
        print(f"Total agentes: {stats['total_agents']}")
        print(f"Velocidad promedio: {stats['average_speed']:.2f} km/h")
        print(f"Distancia total: {stats['total_distance']:.2f} km")
        print(f"Paradas totales: {stats['total_stops']}")
    
    # Ejecutar ejemplo
    asyncio.run(run_integrated_simulation())

# Configuraciones predefinidas
URBAN_TRAFFIC_CONFIG = CivilianTrafficConfig(
    num_civilian_vehicles=100,
    behavior_distribution={
        CivilianBehavior.CONSERVATIVE: 0.15,
        CivilianBehavior.NORMAL: 0.55,
        CivilianBehavior.AGGRESSIVE: 0.20,
        CivilianBehavior.CAUTIOUS: 0.08,
        CivilianBehavior.RECKLESS: 0.02
    },
    traffic_event_probability=0.12,
    weather_change_probability=0.06
)

HIGHWAY_TRAFFIC_CONFIG = CivilianTrafficConfig(
    num_civilian_vehicles=200,
    behavior_distribution={
        CivilianBehavior.CONSERVATIVE: 0.10,
        CivilianBehavior.NORMAL: 0.45,
        CivilianBehavior.AGGRESSIVE: 0.35,
        CivilianBehavior.CAUTIOUS: 0.05,
        CivilianBehavior.RECKLESS: 0.05
    },
    vehicle_type_distribution={
        "car": 0.80,
        "motorcycle": 0.05,
        "van": 0.10,
        "bus": 0.05
    },
    traffic_event_probability=0.08,
    weather_change_probability=0.04
)

RESIDENTIAL_TRAFFIC_CONFIG = CivilianTrafficConfig(
    num_civilian_vehicles=30,
    behavior_distribution={
        CivilianBehavior.CONSERVATIVE: 0.30,
        CivilianBehavior.NORMAL: 0.50,
        CivilianBehavior.AGGRESSIVE: 0.10,
        CivilianBehavior.CAUTIOUS: 0.08,
        CivilianBehavior.RECKLESS: 0.02
    },
    traffic_event_probability=0.05,
    weather_change_probability=0.03
)

if __name__ == "__main__":
    # Ejecutar ejemplo
    example_usage()
