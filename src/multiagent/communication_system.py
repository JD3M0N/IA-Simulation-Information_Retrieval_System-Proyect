"""
Sistema de comunicación entre agentes BDI
Permite intercambio de información entre camiones de reparto
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class MessageType(Enum):
    """Tipos de mensajes entre agentes"""
    ROUTE_INFO = "route_info"
    TRAFFIC_UPDATE = "traffic_update"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    EMERGENCY_ALERT = "emergency_alert"
    FUEL_STATUS = "fuel_status"
    DELIVERY_STATUS = "delivery_status"
    POSITION_UPDATE = "position_update"

@dataclass
class Message:
    """Mensaje entre agentes"""
    message_id: str
    sender_id: str
    receiver_id: str  # "broadcast" para envío masivo
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1-5, donde 5 es más prioritario
    ttl: int = 300  # Time to live en segundos
    
    def is_expired(self) -> bool:
        """Verifica si el mensaje ha expirado"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl

class CommunicationManager:
    """Gestor de comunicación entre agentes"""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}  # agent_id -> agent_reference
        self.message_queue: Dict[str, List[Message]] = {}  # agent_id -> messages
        self.broadcast_messages: List[Message] = []
        self.message_handlers: Dict[str, Callable] = {}
        self.communication_range = 1000.0  # metros
        self.is_active = True
        
        # Estadísticas
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_dropped": 0,
            "broadcast_messages": 0
        }
    
    def register_agent(self, agent):
        """Registra un agente en el sistema de comunicación"""
        if hasattr(agent, 'agent_id'):
            self.agents[agent.agent_id] = agent
            self.message_queue[agent.agent_id] = []
            print(f"Agente {agent.agent_id} registrado en comunicación")
    
    def unregister_agent(self, agent_id: str):
        """Desregistra un agente del sistema"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if agent_id in self.message_queue:
                del self.message_queue[agent_id]
            print(f"Agente {agent_id} desregistrado de comunicación")
    
    def send_message(self, sender_id: str, receiver_id: str, 
                    message_type: MessageType, content: Dict[str, Any],
                    priority: int = 1, ttl: int = 300) -> bool:
        """Envía un mensaje entre agentes"""
        if not self.is_active:
            return False
        
        # Verificar que el remitente existe
        if sender_id not in self.agents:
            return False
        
        # Crear mensaje
        message = Message(
            message_id=f"{sender_id}_{datetime.now().timestamp()}",
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            priority=priority,
            ttl=ttl
        )
        
        # Verificar rango de comunicación
        if receiver_id != "broadcast" and not self._in_communication_range(sender_id, receiver_id):
            self.stats["messages_dropped"] += 1
            return False
        
        # Enviar mensaje
        if receiver_id == "broadcast":
            self.broadcast_messages.append(message)
            self.stats["broadcast_messages"] += 1
        else:
            if receiver_id in self.message_queue:
                self.message_queue[receiver_id].append(message)
                self.stats["messages_delivered"] += 1
            else:
                self.stats["messages_dropped"] += 1
                return False
        
        self.stats["messages_sent"] += 1
        return True
    
    def _in_communication_range(self, sender_id: str, receiver_id: str) -> bool:
        """Verifica si dos agentes están en rango de comunicación"""
        if sender_id not in self.agents or receiver_id not in self.agents:
            return False
        
        sender = self.agents[sender_id]
        receiver = self.agents[receiver_id]
        
        # Calcular distancia
        if hasattr(sender, 'lat') and hasattr(receiver, 'lat'):
            distance = self._calculate_distance(
                sender.lat, sender.lon,
                receiver.lat, receiver.lon
            )
            return distance <= self.communication_range
        
        return True  # Asumir que están en rango si no hay coordenadas
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calcula distancia entre dos puntos geográficos"""
        import math
        
        # Conversión simple a metros (aproximada)
        lat_diff = lat1 - lat2
        lon_diff = lon1 - lon2
        return math.sqrt(lat_diff**2 + lon_diff**2) * 111000  # Aproximación
    
    def get_messages(self, agent_id: str) -> List[Message]:
        """Obtiene mensajes para un agente específico"""
        if agent_id not in self.message_queue:
            return []
        
        # Combinar mensajes directos y broadcast
        direct_messages = self.message_queue[agent_id].copy()
        broadcast_for_agent = [msg for msg in self.broadcast_messages 
                             if msg.sender_id != agent_id and not msg.is_expired()]
        
        all_messages = direct_messages + broadcast_for_agent
        
        # Limpiar mensajes expirados
        self.message_queue[agent_id] = [msg for msg in direct_messages if not msg.is_expired()]
        self.broadcast_messages = [msg for msg in self.broadcast_messages if not msg.is_expired()]
        
        # Ordenar por prioridad y timestamp
        all_messages.sort(key=lambda x: (-x.priority, x.timestamp))
        
        return all_messages
    
    def process_agent_messages(self, agent_id: str):
        """Procesa mensajes para un agente y actualiza sus creencias"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        messages = self.get_messages(agent_id)
        
        for message in messages:
            try:
                # Convertir mensaje a creencia del agente
                if hasattr(agent, 'add_communication_belief'):
                    agent.add_communication_belief(message.sender_id, {
                        "message_type": message.message_type.value,
                        "content": message.content,
                        "timestamp": message.timestamp.isoformat(),
                        "priority": message.priority
                    })
                
                # Procesar mensaje específico según tipo
                self._process_message_by_type(agent, message)
                
            except Exception as e:
                print(f"Error procesando mensaje para {agent_id}: {e}")
    
    def _process_message_by_type(self, agent, message: Message):
        """Procesa mensaje según su tipo específico"""
        if message.message_type == MessageType.ROUTE_INFO:
            self._process_route_info_message(agent, message)
        elif message.message_type == MessageType.TRAFFIC_UPDATE:
            self._process_traffic_update_message(agent, message)
        elif message.message_type == MessageType.COLLABORATION_REQUEST:
            self._process_collaboration_request(agent, message)
        elif message.message_type == MessageType.EMERGENCY_ALERT:
            self._process_emergency_alert(agent, message)
    
    def _process_route_info_message(self, agent, message: Message):
        """Procesa mensaje de información de ruta"""
        route_info = message.content
        
        # Si el agente tiene una ruta similar, puede coordinarse
        if hasattr(agent, 'route') and 'route' in route_info:
            other_route = route_info['route']
            if agent.route and other_route:
                # Buscar nodos en común
                common_nodes = set(agent.route) & set(other_route)
                if len(common_nodes) > 2:  # Rutas con solapamiento significativo
                    # Crear oportunidad de colaboración
                    collaboration_content = {
                        "type": "route_optimization",
                        "common_nodes": list(common_nodes),
                        "suggestion": "coordinate_timing"
                    }
                    
                    # Enviar respuesta de colaboración
                    self.send_message(
                        agent.agent_id,
                        message.sender_id,
                        MessageType.COLLABORATION_REQUEST,
                        collaboration_content
                    )
    
    def _process_traffic_update_message(self, agent, message: Message):
        """Procesa actualización de tráfico"""
        traffic_info = message.content
        
        # Actualizar creencias de tráfico del agente
        if hasattr(agent, 'belief_base'):
            from bdi_core import Belief, BeliefType
            
            traffic_belief = Belief(
                belief_id=f"traffic_from_{message.sender_id}",
                belief_type=BeliefType.TRAFFIC_INFO,
                content=traffic_info,
                source=message.sender_id,
                confidence=0.8
            )
            agent.belief_base.add_belief(traffic_belief)
    
    def _process_collaboration_request(self, agent, message: Message):
        """Procesa solicitud de colaboración"""
        request_content = message.content
        
        # Evaluar si puede colaborar
        can_collaborate = self._evaluate_collaboration_feasibility(agent, request_content)
        
        # Enviar respuesta
        response_content = {
            "request_id": message.message_id,
            "accepted": can_collaborate,
            "agent_status": {
                "fuel_level": getattr(agent, 'fuel_level', 100.0),
                "current_load": getattr(agent, 'current_load', 0),
                "deliveries_remaining": len(getattr(agent, 'delivery_locations', []))
            }
        }
        
        self.send_message(
            agent.agent_id,
            message.sender_id,
            MessageType.COLLABORATION_RESPONSE,
            response_content
        )
    
    def _evaluate_collaboration_feasibility(self, agent, request_content: Dict[str, Any]) -> bool:
        """Evalúa si un agente puede colaborar"""
        # Verificar capacidad de combustible
        if hasattr(agent, 'fuel_level') and agent.fuel_level < 30.0:
            return False
        
        # Verificar carga de trabajo
        if hasattr(agent, 'delivery_locations') and len(agent.delivery_locations) > 10:
            return False
        
        # Verificar tipo de colaboración
        collab_type = request_content.get("type", "")
        if collab_type == "route_optimization":
            return True  # Generalmente beneficioso
        elif collab_type == "emergency_assistance":
            return True  # Prioridad alta
        elif collab_type == "load_sharing":
            # Verificar capacidad disponible
            if hasattr(agent, 'capacity') and hasattr(agent, 'current_load'):
                available_capacity = agent.capacity - agent.current_load
                return available_capacity > 100  # Al menos 100kg disponible
        
        return False
    
    def _process_emergency_alert(self, agent, message: Message):
        """Procesa alerta de emergencia"""
        emergency_info = message.content
        
        # Las alertas de emergencia tienen prioridad máxima
        if hasattr(agent, 'emergency_priority'):
            agent.emergency_priority = True
        
        # Actualizar creencias sobre emergencia
        if hasattr(agent, 'belief_base'):
            from bdi_core import Belief, BeliefType
            
            emergency_belief = Belief(
                belief_id=f"emergency_{message.message_id}",
                belief_type=BeliefType.TRAFFIC_INFO,  # Usar tipo más general
                content=emergency_info,
                source=message.sender_id,
                confidence=1.0  # Alta confianza en emergencias
            )
            agent.belief_base.add_belief(emergency_belief)
    
    def broadcast_traffic_update(self, sender_id: str, traffic_data: Dict[str, Any]):
        """Difunde actualización de tráfico a todos los agentes"""
        self.send_message(
            sender_id,
            "broadcast",
            MessageType.TRAFFIC_UPDATE,
            traffic_data,
            priority=3
        )
    
    def broadcast_emergency(self, sender_id: str, emergency_data: Dict[str, Any]):
        """Difunde alerta de emergencia"""
        self.send_message(
            sender_id,
            "broadcast", 
            MessageType.EMERGENCY_ALERT,
            emergency_data,
            priority=5  # Máxima prioridad
        )
    
    async def communication_loop(self):
        """Bucle principal de comunicación"""
        while self.is_active:
            try:
                # Procesar mensajes para todos los agentes
                for agent_id in list(self.agents.keys()):
                    self.process_agent_messages(agent_id)
                
                # Limpiar mensajes expirados cada minuto
                current_time = datetime.now()
                if hasattr(self, '_last_cleanup'):
                    if (current_time - self._last_cleanup).total_seconds() > 60:
                        self._cleanup_expired_messages()
                        self._last_cleanup = current_time
                else:
                    self._last_cleanup = current_time
                
                # Pausa antes del siguiente ciclo
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"Error en bucle de comunicación: {e}")
                await asyncio.sleep(5.0)
    
    def _cleanup_expired_messages(self):
        """Limpia mensajes expirados"""
        for agent_id in self.message_queue:
            self.message_queue[agent_id] = [
                msg for msg in self.message_queue[agent_id] 
                if not msg.is_expired()
            ]
        
        self.broadcast_messages = [
            msg for msg in self.broadcast_messages 
            if not msg.is_expired()
        ]
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de comunicación"""
        active_agents = len(self.agents)
        total_messages = sum(len(queue) for queue in self.message_queue.values())
        total_broadcasts = len(self.broadcast_messages)
        
        return {
            "active_agents": active_agents,
            "total_queued_messages": total_messages,
            "total_broadcast_messages": total_broadcasts,
            "communication_range": self.communication_range,
            "stats": self.stats.copy()
        }

# Instancia global del gestor de comunicación
communication_manager = CommunicationManager()
