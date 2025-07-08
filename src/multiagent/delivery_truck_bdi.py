import sys
import random
import math
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from bdi_core import (
    BDIAgent, Belief, Desire, BeliefType, DesireType
)
from delivery_intentions import (
    OptimizeFuelConsumptionIntention,
    MinimizeTravelTimeIntention, 
    MaximizeDeliveriesIntention,
    CoordinateWithOthersIntention,
    AvoidTrafficIntention
)

class DeliveryTruckBDI(BDIAgent):
    """Agente BDI para camión de reparto"""
    
    def __init__(self, agent_id: str, initial_node: int, capacity: int = 1000,
                 initial_fuel: float = 100.0):
        super().__init__(agent_id, "delivery_truck_bdi")
        
        # Propiedades físicas del camión
        self.current_node = initial_node
        self.next_node = None
        self.lat = 0.0
        self.lon = 0.0
        self.capacity = capacity  # kg
        self.current_load = 0  # kg
        self.fuel_level = initial_fuel  # %
        self.fuel_capacity = 100.0  # litros
        
        # Propiedades de movimiento
        self.current_speed = 0.0  # km/h
        self.max_speed = 60.0  # km/h
        self.base_speed = 45.0  # km/h
        self.progress = 0.0  # Progreso en arista actual (0-1)
        
        # Información de ruta y entregas
        self.route: List[int] = []
        self.delivery_locations: List[int] = []
        self.completed_deliveries: List[int] = []
        self.delivery_schedule: Dict[int, Dict[str, Any]] = {}
        
        # Referencias al entorno
        self.street_graph: Optional[nx.Graph] = None
        self.environment_reference = None
        
        # Estado operacional
        self.is_active = True
        self.driver_type = "normal"  # normal, aggressive, cautious
        self.emergency_priority = False
        self.maintenance_needed = False
        
        # Métricas específicas de entrega
        self.delivery_metrics = {
            "total_distance": 0.0,
            "fuel_consumed": 0.0,
            "deliveries_completed": 0,
            "deliveries_failed": 0,
            "average_delivery_time": 0.0,
            "on_time_deliveries": 0,
            "late_deliveries": 0
        }
        
        # Inicializar componentes BDI
        self._initialize_desires()
        self._initialize_intentions()
        self._initialize_base_beliefs()
    
    def _initialize_desires(self):
        """Inicializa los deseos base del agente"""
        desires = [
            Desire(
                desire_id="save_fuel",
                desire_type=DesireType.SAVE_FUEL,
                priority=0.7,
                target_value=90.0,  # Mantener al menos 90% de eficiencia
                current_value=100.0
            ),
            Desire(
                desire_id="save_time", 
                desire_type=DesireType.SAVE_TIME,
                priority=0.8,
                target_value=1.0,  # Ratio ideal tiempo planificado/tiempo real
                current_value=1.0
            ),
            Desire(
                desire_id="maximize_deliveries",
                desire_type=DesireType.MAXIMIZE_DELIVERIES,
                priority=0.9,
                target_value=100.0,  # 100% de entregas exitosas
                current_value=0.0
            ),
            Desire(
                desire_id="avoid_traffic",
                desire_type=DesireType.AVOID_TRAFFIC,
                priority=0.6,
                target_value=0.3,  # Mantener congestión bajo 30%
                current_value=0.0
            ),
            Desire(
                desire_id="collaborate",
                desire_type=DesireType.COLLABORATE,
                priority=0.5,
                target_value=3.0,  # Colaborar con al menos 3 agentes
                current_value=0.0
            )
        ]
        
        for desire in desires:
            self.desire_set.add_desire(desire)
    
    def _initialize_intentions(self):
        """Inicializa las intenciones disponibles"""
        intentions = [
            OptimizeFuelConsumptionIntention(priority=0.7),
            MinimizeTravelTimeIntention(priority=0.8),
            MaximizeDeliveriesIntention(priority=0.9),
            CoordinateWithOthersIntention(priority=0.6),
            AvoidTrafficIntention(priority=0.75)
        ]
        
        for intention in intentions:
            self.intention_stack.add_intention(intention)
    
    def _initialize_base_beliefs(self):
        """Inicializa creencias base"""
        # Creencia sobre estado del vehículo
        vehicle_belief = Belief(
            belief_id="vehicle_state",
            belief_type=BeliefType.VEHICLE_INFO,
            content={
                "capacity": self.capacity,
                "current_load": self.current_load,
                "max_speed": self.max_speed,
                "driver_type": self.driver_type
            }
        )
        self.belief_base.add_belief(vehicle_belief)
        
        # Creencia sobre combustible
        fuel_belief = Belief(
            belief_id="fuel_state",
            belief_type=BeliefType.FUEL_INFO,
            content={
                "fuel_level": self.fuel_level,
                "fuel_capacity": self.fuel_capacity,
                "consumption_rate": 8.0  # L/100km estimado
            }
        )
        self.belief_base.add_belief(fuel_belief)
        
        # Creencia sobre entregas
        delivery_belief = Belief(
            belief_id="delivery_state",
            belief_type=BeliefType.DELIVERY_INFO,
            content={
                "pending_deliveries": len(self.delivery_locations),
                "completed_deliveries": len(self.completed_deliveries),
                "delivery_locations": self.delivery_locations.copy(),
                "capacity_utilization": self.current_load / self.capacity
            }
        )
        self.belief_base.add_belief(delivery_belief)
    
    def set_environment_reference(self, environment, street_graph):
        """Establece referencia al entorno y grafo de calles"""
        self.environment_reference = environment
        self.street_graph = street_graph
        
        # Actualizar posición inicial si hay datos del grafo
        if street_graph and self.current_node in street_graph.nodes:
            node_data = street_graph.nodes[self.current_node]
            self.lat = node_data.get('lat', 0.0)
            self.lon = node_data.get('lon', 0.0)
    
    def assign_delivery_route(self, delivery_locations: List[int], 
                            delivery_schedule: Optional[Dict[int, Dict[str, Any]]] = None):
        """Asigna ubicaciones de entrega y calcula ruta inicial"""
        self.delivery_locations = delivery_locations.copy()
        self.delivery_schedule = delivery_schedule or {}
        
        # Calcular ruta inicial simple
        if self.street_graph and delivery_locations:
            try:
                # Crear ruta que visite todas las ubicaciones
                route = [self.current_node]
                remaining_locations = delivery_locations.copy()
                current = self.current_node
                
                while remaining_locations:
                    # Encontrar la ubicación más cercana
                    closest = min(remaining_locations, 
                                key=lambda x: self._estimate_distance(current, x))
                    
                    # Calcular ruta hasta la ubicación más cercana
                    try:
                        path = nx.shortest_path(self.street_graph, current, closest, weight='weight')
                        route.extend(path[1:])  # Excluir el nodo actual
                        current = closest
                        remaining_locations.remove(closest)
                    except nx.NetworkXNoPath:
                        # Si no hay ruta, remover esta ubicación
                        remaining_locations.remove(closest)
                        print(f"No se puede llegar a ubicación {closest}")
                
                self.route = route
                
                # Actualizar creencias
                self._update_route_beliefs()
                self._update_delivery_beliefs()
                
            except Exception as e:
                print(f"Error calculando ruta: {e}")
                self.route = [self.current_node]
    
    def _estimate_distance(self, node1: int, node2: int) -> float:
        """Estima distancia entre dos nodos"""
        if not self.street_graph:
            return abs(node2 - node1)  # Fallback simple
        
        try:
            return nx.shortest_path_length(self.street_graph, node1, node2, weight='weight')
        except nx.NetworkXNoPath:
            return float('inf')
    
    def _update_beliefs_from_environment(self, env_data: Dict[str, Any]):
        """Actualiza creencias basadas en datos del entorno"""
        
        print(self.get_delivery_status())
        # Actualizar creencias de tráfico
        if "road_network" in env_data:
            road_data = env_data["road_network"]
            congestion = road_data.get("congestion", {})
            
            # Calcular nivel de congestión promedio en la ruta
            route_congestion = 0.0
            if self.route and len(self.route) > 1:
                congestion_values = []
                for i in range(len(self.route) - 1):
                    edge = (self.route[i], self.route[i + 1])
                    edge_congestion = congestion.get(str(edge), 1.0)
                    congestion_values.append(edge_congestion)
                
                if congestion_values:
                    route_congestion = sum(congestion_values) / len(congestion_values)
            
            traffic_belief = Belief(
                belief_id="current_traffic",
                belief_type=BeliefType.TRAFFIC_INFO,
                content={
                    "congestion_level": route_congestion,
                    "congested_areas": [k for k, v in congestion.items() if v > 1.5],
                    "traffic_lights": road_data.get("traffic_lights", {})
                }
            )
            self.belief_base.add_belief(traffic_belief)
        
        # Actualizar creencias climáticas
        if "weather" in env_data:
            weather_data = env_data["weather"]
            weather_belief = Belief(
                belief_id="current_weather",
                belief_type=BeliefType.WEATHER_INFO,
                content=weather_data
            )
            self.belief_base.add_belief(weather_belief)
        
        # Actualizar creencias sobre otros vehículos
        if "vehicles" in env_data:
            nearby_vehicles = []
            for vehicle_id, vehicle_data in env_data["vehicles"].items():
                if vehicle_id != self.agent_id:
                    distance = self._calculate_distance_to_vehicle(vehicle_data)
                    if distance < 1000:  # Dentro de 1km
                        nearby_vehicles.append({
                            "vehicle_id": vehicle_id,
                            "distance": distance,
                            "data": vehicle_data
                        })
            
            if nearby_vehicles:
                comm_belief = Belief(
                    belief_id="nearby_vehicles",
                    belief_type=BeliefType.AGENT_COMMUNICATION,
                    content={"nearby_vehicles": nearby_vehicles}
                )
                self.belief_base.add_belief(comm_belief)
    
    def _calculate_distance_to_vehicle(self, vehicle_data: Dict[str, Any]) -> float:
        """Calcula distancia a otro vehículo"""
        other_lat = vehicle_data.get("lat", 0.0)
        other_lon = vehicle_data.get("lon", 0.0)
        
        # Distancia euclidiana simple
        lat_diff = self.lat - other_lat
        lon_diff = self.lon - other_lon
        return math.sqrt(lat_diff**2 + lon_diff**2) * 111000  # Conversión aproximada a metros
    
    def _update_route_beliefs(self):
        """Actualiza creencias sobre la ruta"""
        route_belief = Belief(
            belief_id="current_route_info",
            belief_type=BeliefType.ROUTE_INFO,
            content={
                "current_route": self.route,
                "route_length": len(self.route),
                "current_node": self.current_node,
                "next_node": self.next_node,
                "progress": self.progress
            }
        )
        self.belief_base.add_belief(route_belief)
    
    def _update_delivery_beliefs(self):
        """Actualiza creencias sobre entregas"""
        delivery_belief = Belief(
            belief_id="delivery_status",
            belief_type=BeliefType.DELIVERY_INFO,
            content={
                "pending_deliveries": len(self.delivery_locations),
                "completed_deliveries": len(self.completed_deliveries),
                "delivery_locations": self.delivery_locations.copy(),
                "capacity_utilization": self.current_load / self.capacity,
                "schedule": self.delivery_schedule
            }
        )
        self.belief_base.add_belief(delivery_belief)
    
    def _update_fuel_beliefs(self):
        """Actualiza creencias sobre combustible"""
        fuel_belief = Belief(
            belief_id="fuel_status",
            belief_type=BeliefType.FUEL_INFO,
            content={
                "fuel_level": self.fuel_level,
                "fuel_capacity": self.fuel_capacity,
                "estimated_range": self.fuel_level * 5,  # Estimación simple
                "low_fuel_warning": self.fuel_level < 25.0
            }
        )
        self.belief_base.add_belief(fuel_belief)
    
    def update_position(self, delta_time: float):
        """Actualiza la posición del camión en la ruta"""
        if not self.route or len(self.route) < 2:
            return
        
        # Encontrar posición actual en la ruta
        current_index = None
        for i, node in enumerate(self.route):
            if node == self.current_node:
                current_index = i
                break
        
        if current_index is None or current_index >= len(self.route) - 1:
            return
        
        # Configurar siguiente nodo si no está configurado
        if self.next_node is None:
            self.next_node = self.route[current_index + 1]
        
        # Calcular movimiento
        if self.current_speed > 0:
            # Convertir velocidad de km/h a progreso por segundo
            # Asumiendo que cada arista representa ~100m
            speed_per_second = (self.current_speed / 3.6) / 100.0  # Progreso por segundo
            
            self.progress += speed_per_second * delta_time
            
            # Si completamos la arista, movernos al siguiente nodo
            if self.progress >= 1.0:
                self.current_node = self.next_node
                self.progress = 0.0
                
                # Configurar siguiente nodo
                current_index = None
                for i, node in enumerate(self.route):
                    if node == self.current_node:
                        current_index = i
                        break
                
                if current_index is not None and current_index < len(self.route) - 1:
                    self.next_node = self.route[current_index + 1]
                else:
                    self.next_node = None
                    self.current_speed = 0.0  # Detenerse al final de la ruta
                
                # Verificar si llegamos a una ubicación de entrega
                if self.current_node in self.delivery_locations:
                    self._handle_delivery_arrival()
        
        # Actualizar coordenadas geográficas
        self._update_geographic_position()
        
        # Actualizar métricas
        distance_traveled = (self.current_speed / 3.6) * delta_time / 1000.0  # km
        self.delivery_metrics["total_distance"] += distance_traveled
        
        # Debug - mostrar métricas ocasionalmente
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 1
            
        if self._debug_counter % 50 == 0:  # Cada 50 actualizaciones
            print(f"[DEBUG {self.agent_id}] Speed: {self.current_speed:.1f} km/h, Distance: {self.delivery_metrics['total_distance']:.3f} km, Deliveries: {self.delivery_metrics['deliveries_completed']}")
        
        # Simular consumo de combustible
        fuel_consumption_rate = 8.0 / 100.0  # L/km
        fuel_consumed = distance_traveled * fuel_consumption_rate
        self.fuel_level = max(0.0, self.fuel_level - (fuel_consumed / self.fuel_capacity) * 100.0)
        self.delivery_metrics["fuel_consumed"] += fuel_consumed
        
        # Actualizar creencias
        self._update_route_beliefs()
        self._update_fuel_beliefs()
    
    def _update_geographic_position(self):
        """Actualiza posición geográfica basada en nodos y progreso"""
        if not self.street_graph or self.current_node not in self.street_graph.nodes:
            return
        
        current_node_data = self.street_graph.nodes[self.current_node]
        current_lat = current_node_data.get('lat', 0.0)
        current_lon = current_node_data.get('lon', 0.0)
        
        if self.next_node and self.next_node in self.street_graph.nodes:
            next_node_data = self.street_graph.nodes[self.next_node]
            next_lat = next_node_data.get('lat', 0.0)
            next_lon = next_node_data.get('lon', 0.0)
            
            # Interpolación lineal basada en progreso
            self.lat = current_lat + (next_lat - current_lat) * self.progress
            self.lon = current_lon + (next_lon - current_lon) * self.progress
        else:
            self.lat = current_lat
            self.lon = current_lon
    
    def _handle_delivery_arrival(self):
        """Maneja la llegada a una ubicación de entrega"""
        if self.current_node in self.delivery_locations:
            # Simular tiempo de entrega
            delivery_time = random.uniform(5, 15)  # 5-15 minutos
            
            # Marcar entrega como completada
            self.delivery_locations.remove(self.current_node)
            self.completed_deliveries.append(self.current_node)
            
            # Actualizar carga
            delivery_weight = random.uniform(20, 100)  # kg
            self.current_load = max(0, self.current_load - delivery_weight)
            
            # Actualizar métricas
            self.delivery_metrics["deliveries_completed"] += 1
            
            # Verificar si la entrega fue a tiempo
            if self.current_node in self.delivery_schedule:
                scheduled_time = self.delivery_schedule[self.current_node].get("scheduled_time")
                if scheduled_time:
                    # Simplificación: asumir que llegamos a tiempo si no hay retrasos significativos
                    self.delivery_metrics["on_time_deliveries"] += 1
            
            # Actualizar creencias
            self._update_delivery_beliefs()
            
            print(f"[{self.agent_id}] Entrega completada en nodo {self.current_node}")
    
    def start_movement(self):
        """Inicia el movimiento del camión"""
        if self.route and len(self.route) > 1:
            self.current_speed = self.base_speed
            if len(self.route) > 1:
                self.next_node = self.route[1]
            print(f"[{self.agent_id}] Iniciando movimiento hacia entregas")
    
    def get_delivery_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual de entregas"""
        return {
            "agent_id": self.agent_id,
            "current_node": self.current_node,
            "next_node": self.next_node,
            "position": {"lat": self.lat, "lon": self.lon},
            "route": self.route,
            "delivery_locations": self.delivery_locations,
            "completed_deliveries": self.completed_deliveries,
            "current_load": self.current_load,
            "capacity": self.capacity,
            "fuel_level": self.fuel_level,
            "current_speed": self.current_speed,
            "progress": self.progress,
            "metrics": self.delivery_metrics.copy(),
            "bdi_status": self.get_status()
        }
    
    def _update_beliefs_from_environment(self, env_data: Dict[str, Any]):
        """Actualiza creencias basadas en datos del entorno usando la nueva interfaz"""
        if self.environment_reference:
            try:
                perception = self.environment_reference.get_bdi_agent_perception(
                    self.agent_id, perception_range=1000.0
                )
                if "error" not in perception:
                    self._process_environment_perception(perception)
                else:
                    print(f"[{self.agent_id}] Error obteniendo percepción: {perception['error']}")
            except Exception as e:
                print(f"[{self.agent_id}] Error en percepción del entorno: {e}")
        
        self._process_basic_environment_data(env_data)
    
    def _process_environment_perception(self, perception: Dict[str, Any]):
        """Procesa la percepción detallada del entorno"""
        if "weather" in perception:
            weather_belief = Belief(
                belief_id="current_weather",
                belief_type=BeliefType.WEATHER_INFO,
                content=perception["weather"],
                confidence=0.9
            )
            self.belief_base.add_belief(weather_belief)
        
        if "traffic_conditions" in perception:
            traffic_belief = Belief(
                belief_id="local_traffic",
                belief_type=BeliefType.TRAFFIC_INFO,
                content=perception["traffic_conditions"],
                confidence=0.8
            )
            self.belief_base.add_belief(traffic_belief)
    
    def _process_basic_environment_data(self, env_data: Dict[str, Any]):
        """Procesa datos básicos del entorno (fallback)"""
        if "vehicles" in env_data:
            vehicles_belief = Belief(
                belief_id="nearby_vehicles",
                belief_type=BeliefType.TRAFFIC_INFO,
                content={"vehicle_count": len(env_data["vehicles"])},
                confidence=0.6
            )
            self.belief_base.add_belief(vehicles_belief)
    
    def execute_environment_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción en el entorno usando la nueva interfaz"""
        if not self.environment_reference:
            return {"success": False, "error": "No environment reference available"}
        
        try:
            result = self.environment_reference.execute_bdi_agent_action(self.agent_id, action)
            if result.get("success", False):
                print(f"[{self.agent_id}] Acción '{action.get('type', 'unknown')}' ejecutada exitosamente")
            return result
        except Exception as e:
            return {"success": False, "error": f"Error ejecutando acción: {str(e)}"}
