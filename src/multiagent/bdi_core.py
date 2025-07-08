import sys
import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import random
import math

# Base enums for BDI system
class BeliefType(Enum):
    """Tipos de creencias del agente"""
    ROUTE_INFO = "route_info"
    TRAFFIC_INFO = "traffic_info"  
    VEHICLE_INFO = "vehicle_info"
    WEATHER_INFO = "weather_info"
    FUEL_INFO = "fuel_info"
    DELIVERY_INFO = "delivery_info"
    AGENT_COMMUNICATION = "agent_communication"

class DesireType(Enum):
    """Tipos de deseos del agente"""
    SAVE_FUEL = "save_fuel"
    SAVE_TIME = "save_time"
    MAXIMIZE_DELIVERIES = "maximize_deliveries"
    AVOID_TRAFFIC = "avoid_traffic"
    MAINTAIN_SCHEDULE = "maintain_schedule"
    COLLABORATE = "collaborate"

class IntentionType(Enum):
    """Tipos de intenciones del agente"""
    INDIVIDUAL = "individual"
    COLLECTIVE = "collective"

# Core BDI Classes
@dataclass
class Belief:
    """Representación de una creencia del agente"""
    belief_id: str
    belief_type: BeliefType
    content: Any
    confidence: float = 1.0  # 0.0 a 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "self"  # "self", "communication", "observation"
    
    def is_expired(self, max_age_seconds: int = 300) -> bool:
        """Verifica si la creencia ha expirado"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > max_age_seconds

@dataclass  
class Desire:
    """Representación de un deseo del agente"""
    desire_id: str
    desire_type: DesireType
    priority: float = 1.0  # 0.0 a 1.0
    target_value: float = 0.0
    current_value: float = 0.0
    is_active: bool = True
    
    def satisfaction_level(self) -> float:
        """Calcula el nivel de satisfacción del deseo"""
        if self.target_value == 0:
            return 1.0
        return min(1.0, self.current_value / self.target_value)

class Intention(ABC):
    """Clase base abstracta para intenciones"""
    
    def __init__(self, intention_id: str, intention_type: IntentionType, 
                 priority: float = 1.0):
        self.intention_id = intention_id
        self.intention_type = intention_type
        self.priority = priority
        self.is_active = True
        self.created_at = datetime.now()
        
    @abstractmethod
    def evaluate(self, beliefs: Dict[str, Belief], desires: Dict[str, Desire]) -> float:
        """
        Evalúa si la intención debe ejecutarse
        
        Args:
            beliefs: Creencias actuales del agente
            desires: Deseos actuales del agente
            
        Returns:
            float: Puntuación de activación (0.0 a 1.0)
        """
        pass
    
    @abstractmethod
    async def execute(self, agent_context: Any) -> bool:
        """
        Ejecuta la intención
        
        Args:
            agent_context: Contexto del agente (referencia al agente BDI)
            
        Returns:
            bool: True si la ejecución fue exitosa
        """
        pass

class BeliefBase:
    """Base de conocimientos del agente"""
    
    def __init__(self):
        self.beliefs: Dict[str, Belief] = {}
        self.max_age_seconds = 300  # 5 minutos por defecto
    
    def add_belief(self, belief: Belief):
        """Añade o actualiza una creencia"""
        self.beliefs[belief.belief_id] = belief
    
    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """Obtiene una creencia específica"""
        belief = self.beliefs.get(belief_id)
        if belief and belief.is_expired(self.max_age_seconds):
            del self.beliefs[belief_id]
            return None
        return belief
    
    def get_beliefs_by_type(self, belief_type: BeliefType) -> List[Belief]:
        """Obtiene todas las creencias de un tipo específico"""
        return [b for b in self.beliefs.values() 
                if b.belief_type == belief_type and not b.is_expired(self.max_age_seconds)]
    
    def remove_expired_beliefs(self):
        """Limpia creencias expiradas"""
        expired_ids = [bid for bid, belief in self.beliefs.items() 
                      if belief.is_expired(self.max_age_seconds)]
        for bid in expired_ids:
            del self.beliefs[bid]
    
    def update_belief_content(self, belief_id: str, new_content: Any, 
                            confidence: float = 1.0):
        """Actualiza el contenido de una creencia existente"""
        if belief_id in self.beliefs:
            self.beliefs[belief_id].content = new_content
            self.beliefs[belief_id].confidence = confidence
            self.beliefs[belief_id].timestamp = datetime.now()

class DesireSet:
    """Conjunto de deseos del agente"""
    
    def __init__(self):
        self.desires: Dict[str, Desire] = {}
    
    def add_desire(self, desire: Desire):
        """Añade un nuevo deseo"""
        self.desires[desire.desire_id] = desire
    
    def get_desire(self, desire_id: str) -> Optional[Desire]:
        """Obtiene un deseo específico"""
        return self.desires.get(desire_id)
    
    def get_active_desires(self) -> List[Desire]:
        """Obtiene todos los deseos activos ordenados por prioridad"""
        active = [d for d in self.desires.values() if d.is_active]
        return sorted(active, key=lambda x: x.priority, reverse=True)
    
    def update_desire_value(self, desire_id: str, current_value: float):
        """Actualiza el valor actual de un deseo"""
        if desire_id in self.desires:
            self.desires[desire_id].current_value = current_value

class IntentionStack:
    """Pila de intenciones del agente"""
    
    def __init__(self):
        self.intentions: List[Intention] = []
        self.executing_intention: Optional[Intention] = None
    
    def add_intention(self, intention: Intention):
        """Añade una nueva intención"""
        self.intentions.append(intention)
        self._sort_by_priority()
    
    def _sort_by_priority(self):
        """Ordena intenciones por prioridad"""
        self.intentions.sort(key=lambda x: x.priority, reverse=True)
    
    def get_next_intention(self) -> Optional[Intention]:
        """Obtiene la siguiente intención a ejecutar"""
        active_intentions = [i for i in self.intentions if i.is_active]
        return active_intentions[0] if active_intentions else None
    
    def remove_intention(self, intention_id: str):
        """Remueve una intención"""
        self.intentions = [i for i in self.intentions if i.intention_id != intention_id]
    
    def clear_completed_intentions(self):
        """Limpia intenciones completadas"""
        self.intentions = [i for i in self.intentions if i.is_active]

class BDIAgent:
    """Agente BDI base"""
    
    def __init__(self, agent_id: str, agent_type: str = "delivery_truck"):
        self.agent_id = agent_id
        self.agent_type = agent_type
        
        # Componentes BDI
        self.belief_base = BeliefBase()
        self.desire_set = DesireSet()
        self.intention_stack = IntentionStack()
        
        # Estado del agente
        self.is_active = True
        self.cycle_count = 0
        self.last_update = datetime.now()
        
        # Métricas
        self.metrics = {
            "decisions_made": 0,
            "intentions_executed": 0,
            "beliefs_updated": 0,
            "communication_received": 0
        }
    
    def perceive(self, environment_data: Dict[str, Any]):
        """Percibe el entorno y actualiza creencias"""
        self._update_beliefs_from_environment(environment_data)
        self.belief_base.remove_expired_beliefs()
        self.metrics["beliefs_updated"] += 1
    
    def deliberate(self) -> Optional[Intention]:
        """Proceso de deliberación para seleccionar intenciones"""
        active_desires = self.desire_set.get_active_desires()
        
        if not active_desires:
            return None
        
        # Evaluar todas las intenciones disponibles
        best_intention = None
        best_score = 0.0
        
        for intention in self.intention_stack.intentions:
            if not intention.is_active:
                continue
                
            score = intention.evaluate(self.belief_base.beliefs, self.desire_set.desires)
            if score > best_score:
                best_score = score
                best_intention = intention
        
        self.metrics["decisions_made"] += 1
        return best_intention
    
    async def execute_intention(self, intention: Intention) -> bool:
        """Ejecuta una intención específica"""
        if not intention or not intention.is_active:
            return False
        
        self.intention_stack.executing_intention = intention
        
        try:
            success = await intention.execute(self)
            if success:
                self.metrics["intentions_executed"] += 1
            return success
        except Exception as e:
            print(f"Error ejecutando intención {intention.intention_id}: {e}")
            return False
        finally:
            self.intention_stack.executing_intention = None
    
    async def bdi_cycle(self, environment_data: Dict[str, Any]):
        """Ciclo principal BDI"""
        self.cycle_count += 1
        
        # 1. Percibir
        self.perceive(environment_data)
        
        # 2. Deliberar
        selected_intention = self.deliberate()
        
        # 3. Ejecutar
        if selected_intention:
            await self.execute_intention(selected_intention)
        
        # 4. Limpiar
        self.intention_stack.clear_completed_intentions()
        
        self.last_update = datetime.now()
    
    def _update_beliefs_from_environment(self, env_data: Dict[str, Any]):
        """Actualiza creencias basadas en datos del entorno"""
        # Esta implementación será específica para cada tipo de agente
        pass
    
    def add_communication_belief(self, source_agent: str, message: Dict[str, Any]):
        """Añade creencias basadas en comunicación con otros agentes"""
        belief_id = f"comm_{source_agent}_{datetime.now().timestamp()}"
        belief = Belief(
            belief_id=belief_id,
            belief_type=BeliefType.AGENT_COMMUNICATION,
            content=message,
            source=source_agent,
            confidence=0.8  # Confianza moderada en comunicaciones
        )
        self.belief_base.add_belief(belief)
        self.metrics["communication_received"] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del agente BDI"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "is_active": self.is_active,
            "cycle_count": self.cycle_count,
            "beliefs_count": len(self.belief_base.beliefs),
            "desires_count": len(self.desire_set.desires),
            "intentions_count": len(self.intention_stack.intentions),
            "executing_intention": self.intention_stack.executing_intention.intention_id if self.intention_stack.executing_intention else None,
            "metrics": self.metrics.copy(),
            "last_update": self.last_update.isoformat()
        }
