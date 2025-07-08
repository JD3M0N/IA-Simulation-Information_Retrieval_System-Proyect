import sys
import asyncio
import random
import math
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from bdi_core import Intention, IntentionType, Belief, Desire, BeliefType, DesireType

class OptimizeFuelConsumptionIntention(Intention):
    """Intención individual para optimizar el consumo de combustible"""
    
    def __init__(self, priority: float = 0.7):
        super().__init__(
            intention_id="optimize_fuel_consumption",
            intention_type=IntentionType.INDIVIDUAL,
            priority=priority
        )
        self.fuel_threshold = 30.0  # Umbral de combustible bajo
    
    def evaluate(self, beliefs: Dict[str, Belief], desires: Dict[str, Desire]) -> float:
        """Evalúa si debe optimizar combustible"""
        # Verificar deseo de ahorrar combustible
        fuel_desire = desires.get("save_fuel")
        if not fuel_desire or not fuel_desire.is_active:
            return 0.0
        
        # Verificar nivel de combustible actual
        fuel_beliefs = [b for b in beliefs.values() 
                       if b.belief_type == BeliefType.FUEL_INFO]
        
        if not fuel_beliefs:
            return 0.1  # Activación baja si no hay información
        
        current_fuel = fuel_beliefs[0].content.get("fuel_level", 100.0)
        
        # Mayor puntuación si el combustible está bajo
        if current_fuel < self.fuel_threshold:
            return min(1.0, (self.fuel_threshold - current_fuel) / self.fuel_threshold + 0.5)
        
        return fuel_desire.priority * 0.3
    
    async def execute(self, agent_context: Any) -> bool:
        """Ejecuta optimización de combustible"""
        try:
            # Reducir velocidad para ahorrar combustible
            if hasattr(agent_context, 'current_speed'):
                agent_context.current_speed *= 0.9  # Reducir 10%
            
            # Actualizar creencia sobre estrategia de combustible
            fuel_strategy_belief = Belief(
                belief_id="fuel_strategy_active",
                belief_type=BeliefType.VEHICLE_INFO,
                content={"strategy": "fuel_optimization", "active": True},
                confidence=1.0
            )
            agent_context.belief_base.add_belief(fuel_strategy_belief)
            
            # print(f"[{agent_context.agent_id}] Optimizando consumo de combustible")  # Comentado
            return True
            
        except Exception as e:
            print(f"Error en optimización de combustible: {e}")
            return False

class MinimizeTravelTimeIntention(Intention):
    """Intención individual para minimizar tiempo de viaje"""
    
    def __init__(self, priority: float = 0.8):
        super().__init__(
            intention_id="minimize_travel_time",
            intention_type=IntentionType.INDIVIDUAL,
            priority=priority
        )
    
    def evaluate(self, beliefs: Dict[str, Belief], desires: Dict[str, Desire]) -> float:
        """Evalúa si debe minimizar tiempo de viaje"""
        time_desire = desires.get("save_time")
        if not time_desire or not time_desire.is_active:
            return 0.0
        
        # Verificar si hay información de tráfico
        traffic_beliefs = [b for b in beliefs.values() 
                          if b.belief_type == BeliefType.TRAFFIC_INFO]
        
        schedule_beliefs = [b for b in beliefs.values() 
                           if b.belief_type == BeliefType.DELIVERY_INFO]
        
        score = time_desire.priority * 0.5
        
        # Aumentar puntuación si hay congestión
        if traffic_beliefs:
            congestion_level = traffic_beliefs[0].content.get("congestion_level", 0.0)
            score += congestion_level * 0.3
        
        # Aumentar puntuación si hay retrasos en entregas
        if schedule_beliefs:
            delays = schedule_beliefs[0].content.get("delays", 0)
            score += min(0.4, delays * 0.1)
        
        return min(1.0, score)
    
    async def execute(self, agent_context: Any) -> bool:
        """Ejecuta minimización de tiempo de viaje"""
        try:
            # Buscar rutas alternativas más rápidas
            route_beliefs = [b for b in agent_context.belief_base.beliefs.values() 
                           if b.belief_type == BeliefType.ROUTE_INFO]
            
            if route_beliefs and hasattr(agent_context, 'street_graph'):
                route_info = route_beliefs[0].content
                current_route = route_info.get("current_route", [])
                
                if len(current_route) > 2:
                    # Intentar encontrar ruta más rápida
                    start = current_route[0]
                    end = current_route[-1]
                    
                    try:
                        # Usar algoritmo de ruta más corta
                        new_route = nx.shortest_path(
                            agent_context.street_graph, 
                            start, end, 
                            weight='weight'
                        )
                        
                        if len(new_route) < len(current_route):
                            # Actualizar ruta si es más eficiente
                            if hasattr(agent_context, 'route'):
                                agent_context.route = new_route
                            
                            # Actualizar creencia de ruta
                            route_belief = Belief(
                                belief_id="optimized_route",
                                belief_type=BeliefType.ROUTE_INFO,
                                content={"current_route": new_route, "optimized": True}
                            )
                            agent_context.belief_base.add_belief(route_belief)
                            
                    except nx.NetworkXNoPath:
                        pass
            
            # Aumentar velocidad si es seguro
            if hasattr(agent_context, 'current_speed') and hasattr(agent_context, 'max_speed'):
                max_safe_speed = getattr(agent_context, 'max_speed', 60.0)
                agent_context.current_speed = min(max_safe_speed, agent_context.current_speed * 1.1)
            
            # print(f"[{agent_context.agent_id}] Minimizando tiempo de viaje")  # Comentado
            return True
            
        except Exception as e:
            print(f"Error en minimización de tiempo: {e}")
            return False

class MaximizeDeliveriesIntention(Intention):
    """Intención individual para maximizar entregas"""
    
    def __init__(self, priority: float = 0.9):
        super().__init__(
            intention_id="maximize_deliveries",
            intention_type=IntentionType.INDIVIDUAL,
            priority=priority
        )
    
    def evaluate(self, beliefs: Dict[str, Belief], desires: Dict[str, Desire]) -> float:
        """Evalúa si debe maximizar entregas"""
        delivery_desire = desires.get("maximize_deliveries")
        if not delivery_desire or not delivery_desire.is_active:
            return 0.0
        
        # Verificar información de entregas pendientes
        delivery_beliefs = [b for b in beliefs.values() 
                           if b.belief_type == BeliefType.DELIVERY_INFO]
        
        if not delivery_beliefs:
            return 0.2
        
        delivery_info = delivery_beliefs[0].content
        pending_deliveries = delivery_info.get("pending_deliveries", 0)
        completed_deliveries = delivery_info.get("completed_deliveries", 0)
        capacity_utilization = delivery_info.get("capacity_utilization", 0.0)
        
        # Mayor puntuación si hay muchas entregas pendientes o baja utilización
        score = delivery_desire.priority * 0.6
        
        if pending_deliveries > 5:
            score += 0.3
        
        if capacity_utilization < 0.7:  # Menos del 70% de capacidad utilizada
            score += 0.2
        
        return min(1.0, score)
    
    async def execute(self, agent_context: Any) -> bool:
        """Ejecuta maximización de entregas"""
        try:
            # Reordenar entregas para optimizar la secuencia
            delivery_beliefs = [b for b in agent_context.belief_base.beliefs.values() 
                               if b.belief_type == BeliefType.DELIVERY_INFO]
            
            if delivery_beliefs:
                delivery_info = delivery_beliefs[0].content
                pending_deliveries = delivery_info.get("delivery_locations", [])
                
                if len(pending_deliveries) > 1:
                    # Reordenar por proximidad (algoritmo del viajante simple)
                    if hasattr(agent_context, 'current_node'):
                        current_pos = agent_context.current_node
                        optimized_sequence = self._optimize_delivery_sequence(
                            current_pos, pending_deliveries
                        )
                        
                        # Actualizar creencia con secuencia optimizada
                        optimized_belief = Belief(
                            belief_id="optimized_delivery_sequence",
                            belief_type=BeliefType.DELIVERY_INFO,
                            content={
                                "delivery_sequence": optimized_sequence,
                                "optimized": True
                            }
                        )
                        agent_context.belief_base.add_belief(optimized_belief)
            
            # print(f"[{agent_context.agent_id}] Maximizando entregas")  # Comentado
            return True
            
        except Exception as e:
            print(f"Error en maximización de entregas: {e}")
            return False
    
    def _optimize_delivery_sequence(self, start_pos: int, 
                                  delivery_locations: List[int]) -> List[int]:
        """Optimiza la secuencia de entregas usando algoritmo greedy"""
        if not delivery_locations:
            return []
        
        remaining = delivery_locations.copy()
        sequence = []
        current = start_pos
        
        while remaining:
            # Encontrar la entrega más cercana
            closest = min(remaining, key=lambda x: abs(x - current))
            sequence.append(closest)
            remaining.remove(closest)
            current = closest
        
        return sequence

class CoordinateWithOthersIntention(Intention):
    """Intención colectiva para coordinar con otros agentes"""
    
    def __init__(self, priority: float = 0.6):
        super().__init__(
            intention_id="coordinate_with_others",
            intention_type=IntentionType.COLLECTIVE,
            priority=priority
        )
        self.coordination_radius = 1000  # metros
    
    def evaluate(self, beliefs: Dict[str, Belief], desires: Dict[str, Desire]) -> float:
        """Evalúa si debe coordinar con otros agentes"""
        collaborate_desire = desires.get("collaborate")
        if not collaborate_desire or not collaborate_desire.is_active:
            return 0.0
        
        # Verificar si hay otros agentes cerca
        comm_beliefs = [b for b in beliefs.values() 
                       if b.belief_type == BeliefType.AGENT_COMMUNICATION]
        
        if not comm_beliefs:
            return 0.1  # Baja puntuación si no hay comunicación
        
        # Contar agentes cercanos
        nearby_agents = 0
        for belief in comm_beliefs:
            message = belief.content
            if message.get("distance", float('inf')) < self.coordination_radius:
                nearby_agents += 1
        
        # Mayor puntuación si hay más agentes cercanos
        score = collaborate_desire.priority * 0.4
        score += min(0.5, nearby_agents * 0.1)
        
        return min(1.0, score)
    
    async def execute(self, agent_context: Any) -> bool:
        """Ejecuta coordinación con otros agentes"""
        try:
            # Compartir información de ruta y entregas
            share_info = {
                "agent_id": agent_context.agent_id,
                "position": getattr(agent_context, 'current_node', None),
                "route": getattr(agent_context, 'route', []),
                "deliveries_remaining": self._get_remaining_deliveries(agent_context),
                "fuel_level": self._get_fuel_level(agent_context),
                "timestamp": datetime.now().isoformat()
            }
            
            # En un sistema real, esto se enviaría a otros agentes
            # Por ahora, creamos una creencia de coordinación
            coordination_belief = Belief(
                belief_id="coordination_info_shared",
                belief_type=BeliefType.AGENT_COMMUNICATION,
                content=share_info,
                source="self"
            )
            agent_context.belief_base.add_belief(coordination_belief)
            
            # Buscar oportunidades de colaboración
            comm_beliefs = [b for b in agent_context.belief_base.beliefs.values() 
                           if b.belief_type == BeliefType.AGENT_COMMUNICATION and b.source != "self"]
            
            for belief in comm_beliefs:
                other_agent_info = belief.content
                if self._can_collaborate(agent_context, other_agent_info):
                    await self._propose_collaboration(agent_context, other_agent_info)
            
            # print(f"[{agent_context.agent_id}] Coordinando con otros agentes")  # Comentado
            return True
            
        except Exception as e:
            print(f"Error en coordinación: {e}")
            return False
    
    def _get_remaining_deliveries(self, agent_context: Any) -> int:
        """Obtiene el número de entregas restantes"""
        delivery_beliefs = [b for b in agent_context.belief_base.beliefs.values() 
                           if b.belief_type == BeliefType.DELIVERY_INFO]
        
        if delivery_beliefs:
            return delivery_beliefs[0].content.get("pending_deliveries", 0)
        return 0
    
    def _get_fuel_level(self, agent_context: Any) -> float:
        """Obtiene el nivel de combustible actual"""
        fuel_beliefs = [b for b in agent_context.belief_base.beliefs.values() 
                       if b.belief_type == BeliefType.FUEL_INFO]
        
        if fuel_beliefs:
            return fuel_beliefs[0].content.get("fuel_level", 100.0)
        return 100.0
    
    def _can_collaborate(self, agent_context: Any, other_agent_info: Dict[str, Any]) -> bool:
        """Verifica si puede colaborar con otro agente"""
        # Lógica simple: colaborar si están cerca y tienen entregas en áreas similares
        own_position = getattr(agent_context, 'current_node', None)
        other_position = other_agent_info.get("position")
        
        if not own_position or not other_position:
            return False
        
        # Colaborar si están relativamente cerca
        distance = abs(own_position - other_position)
        return distance < 50  # Umbral de distancia arbitrario
    
    async def _propose_collaboration(self, agent_context: Any, other_agent_info: Dict[str, Any]):
        """Propone colaboración con otro agente"""
        collaboration_proposal = {
            "type": "collaboration_proposal",
            "from_agent": agent_context.agent_id,
            "to_agent": other_agent_info.get("agent_id"),
            "proposal": "route_sharing",
            "timestamp": datetime.now().isoformat()
        }
        
        # Crear creencia sobre la propuesta
        proposal_belief = Belief(
            belief_id=f"collab_proposal_{other_agent_info.get('agent_id')}",
            belief_type=BeliefType.AGENT_COMMUNICATION,
            content=collaboration_proposal,
            source="self"
        )
        agent_context.belief_base.add_belief(proposal_belief)

class AvoidTrafficIntention(Intention):
    """Intención individual para evitar tráfico"""
    
    def __init__(self, priority: float = 0.75):
        super().__init__(
            intention_id="avoid_traffic",
            intention_type=IntentionType.INDIVIDUAL,
            priority=priority
        )
        self.congestion_threshold = 0.6
    
    def evaluate(self, beliefs: Dict[str, Belief], desires: Dict[str, Desire]) -> float:
        """Evalúa si debe evitar tráfico"""
        traffic_desire = desires.get("avoid_traffic")
        if not traffic_desire or not traffic_desire.is_active:
            return 0.0
        
        # Verificar información de tráfico
        traffic_beliefs = [b for b in beliefs.values() 
                          if b.belief_type == BeliefType.TRAFFIC_INFO]
        
        if not traffic_beliefs:
            return 0.1
        
        traffic_info = traffic_beliefs[0].content
        congestion_level = traffic_info.get("congestion_level", 0.0)
        
        # Mayor puntuación si hay mucha congestión
        if congestion_level > self.congestion_threshold:
            return min(1.0, congestion_level + 0.2)
        
        return traffic_desire.priority * 0.3
    
    async def execute(self, agent_context: Any) -> bool:
        """Ejecuta evitación de tráfico"""
        try:
            # Buscar rutas alternativas con menos tráfico
            traffic_beliefs = [b for b in agent_context.belief_base.beliefs.values() 
                              if b.belief_type == BeliefType.TRAFFIC_INFO]
            
            if traffic_beliefs and hasattr(agent_context, 'street_graph'):
                traffic_info = traffic_beliefs[0].content
                congested_areas = traffic_info.get("congested_areas", [])
                route_changed = False
                
                # Intentar evitar áreas congestionadas solo si hay congestión
                if hasattr(agent_context, 'route') and congested_areas:
                    current_route = getattr(agent_context, 'route', [])
                    if current_route and len(current_route) > 2:
                        # Crear una versión modificada del grafo sin aristas congestionadas
                        modified_graph = agent_context.street_graph.copy()
                        
                        for congested_area in congested_areas:
                            if congested_area in modified_graph.nodes():
                                # Aumentar peso de aristas congestionadas
                                for neighbor in modified_graph.neighbors(congested_area):
                                    if modified_graph.has_edge(congested_area, neighbor):
                                        edge_data = modified_graph[congested_area][neighbor]
                                        edge_data['weight'] = edge_data.get('weight', 1) * 2.0
                        
                        try:
                            # Recalcular ruta evitando tráfico
                            start = current_route[0]
                            end = current_route[-1]
                            alternative_route = nx.shortest_path(
                                modified_graph, start, end, weight='weight'
                            )
                            
                            if len(alternative_route) <= len(current_route) * 1.2:  # No más del 20% más larga
                                if alternative_route != current_route:  # Solo si la ruta realmente cambió
                                    agent_context.route = alternative_route
                                    route_changed = True
                                    
                                    # Actualizar creencia de ruta
                                    route_belief = Belief(
                                        belief_id="traffic_avoiding_route",
                                        belief_type=BeliefType.ROUTE_INFO,
                                        content={"current_route": alternative_route, "traffic_avoided": True}
                                    )
                                    agent_context.belief_base.add_belief(route_belief)
                                    
                        except nx.NetworkXNoPath:
                            pass
            
            # Solo imprimir si realmente se evitó tráfico o hay actividad relevante
            if route_changed:
                print(f"[{agent_context.agent_id}] Evitando tráfico - Ruta recalculada")
            # Comentado para evitar logs verbosos
            # else:
            #     # Determinar estado actual más específico
            #     current_congestion = 0.0
            #     if traffic_beliefs:
            #         traffic_info = traffic_beliefs[0].content
            #         current_congestion = traffic_info.get("congestion_level", 0.0)
            #     
            #     if current_congestion > self.congestion_threshold:
            #         print(f"[{agent_context.agent_id}] Monitoreando tráfico - Congestión: {current_congestion:.2f}")
            #     elif hasattr(agent_context, 'route') and agent_context.route:
            #         # Verificar si tiene entregas pendientes
            #         if hasattr(agent_context, 'delivery_locations') and agent_context.delivery_locations:
            #             print(f"[{agent_context.agent_id}] En ruta - {len(agent_context.delivery_locations)} entregas pendientes")
            #         else:
            #             print(f"[{agent_context.agent_id}] Patrullando - Ruta activa")
            #     else:
            #         print(f"[{agent_context.agent_id}] En espera - Sin ruta asignada")
            
            return True
            
        except Exception as e:
            print(f"[{agent_context.agent_id}] Error en análisis de tráfico: {e}")
            return False
