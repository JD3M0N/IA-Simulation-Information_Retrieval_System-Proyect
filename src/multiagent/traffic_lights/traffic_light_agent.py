"""
Agente de Semáforo Individual
Maneja la lógica de un semáforo específico con comportamiento inteligente
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import random
import math

from .traffic_light_models import (
    TrafficLightData, TrafficLightPhase, IntersectionData, 
    TrafficFlow, TrafficDirection, PriorityLevel, PhaseConfiguration
)


class TrafficLightAgent:
    """
    Agente inteligente que controla un semáforo individual
    Implementa lógica adaptativa y respuesta a emergencias
    """
    
    def __init__(self, light_id: str, node_id: int, 
                 latitude: float = 0.0, longitude: float = 0.0,
                 intersection_id: str = ""):
        """
        Inicializa el agente de semáforo
        
        Args:
            light_id: Identificador único del semáforo
            node_id: ID del nodo en el grafo de calles
            latitude: Latitud de la ubicación
            longitude: Longitud de la ubicación  
            intersection_id: ID de la intersección (si aplica)
        """
        self.logger = logging.getLogger(f"TrafficLight-{light_id}")
        
        # Inicializar datos del semáforo
        self.data = TrafficLightData(
            light_id=light_id,
            node_id=node_id,
            intersection_id=intersection_id or f"intersection_{node_id}",
            latitude=latitude,
            longitude=longitude
        )
        
        # Configurar fases por defecto
        self._setup_default_phases()
        
        # Estado interno del agente
        self.is_running = False
        self.last_update = datetime.now()
        self.update_interval = 1.0  # segundos
        
        # Percepción del entorno
        self.perceived_traffic: Dict[TrafficDirection, TrafficFlow] = {}
        self.nearby_vehicles: List[Dict[str, Any]] = []
        self.weather_factor = 1.0
        self.visibility_factor = 1.0
        
        # Control adaptativo
        self.adaptation_enabled = True
        self.learning_rate = 0.1
        self.historical_patterns: Dict[str, Any] = {}
        
        # Sistema de eventos
        self.event_queue: List[Dict[str, Any]] = []
        self.priority_override_active = False
        self.override_end_time: Optional[datetime] = None
        
        self.logger.info(f"Agente de semáforo inicializado: {light_id} en nodo {node_id}")
    
    def _setup_default_phases(self):
        """Configura las fases por defecto del semáforo"""
        # Configuración estándar para intersección simple
        self.data.phase_config = {
            TrafficLightPhase.GREEN: PhaseConfiguration(
                phase=TrafficLightPhase.GREEN,
                duration=30.0,
                allowed_directions=[TrafficDirection.NORTH, TrafficDirection.SOUTH],
                min_duration=15.0,
                max_duration=60.0,
                is_adaptive=True
            ),
            TrafficLightPhase.YELLOW: PhaseConfiguration(
                phase=TrafficLightPhase.YELLOW,
                duration=5.0,
                allowed_directions=[],
                min_duration=3.0,
                max_duration=8.0,
                is_adaptive=False
            ),
            TrafficLightPhase.RED: PhaseConfiguration(
                phase=TrafficLightPhase.RED,
                duration=25.0,
                allowed_directions=[TrafficDirection.EAST, TrafficDirection.WEST],
                min_duration=10.0,
                max_duration=90.0,
                is_adaptive=True
            )
        }
    
    async def start_operation(self):
        """Inicia la operación del semáforo"""
        if self.is_running:
            return
        
        self.is_running = True
        self.data.is_operational = True
        self.data.phase_start_time = datetime.now()
        
        self.logger.info(f"Semáforo {self.data.light_id} iniciado")
        
        # Ejecutar el ciclo principal en background
        asyncio.create_task(self._main_cycle())
    
    async def stop_operation(self):
        """Detiene la operación del semáforo"""
        self.is_running = False
        self.data.is_operational = False
        self.logger.info(f"Semáforo {self.data.light_id} detenido")
    
    async def _main_cycle(self):
        """Ciclo principal de operación del semáforo"""
        while self.is_running:
            try:
                # Actualizar percepción del entorno
                await self._perceive_environment()
                
                # Procesar eventos de alta prioridad
                await self._process_priority_events()
                
                # Tomar decisión sobre cambio de fase
                decision = await self._decide_phase_action()
                
                # Ejecutar acción decidida
                await self._execute_action(decision)
                
                # Actualizar métricas y estado
                await self._update_metrics()
                
                # Esperar hasta la próxima actualización
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error en ciclo principal: {e}")
                await asyncio.sleep(5.0)  # Espera de seguridad
    
    async def _perceive_environment(self):
        """Percibe el estado del entorno circundante"""
        try:
            # Actualizar tiempo en fase actual
            current_time = datetime.now()
            self.data.time_in_phase = (current_time - self.data.phase_start_time).total_seconds()
            
            # Detectar vehículos cercanos (simulado)
            await self._detect_nearby_vehicles()
            
            # Analizar flujos de tráfico
            await self._analyze_traffic_flows()
            
            # Evaluar condiciones ambientales
            await self._assess_environmental_conditions()
            
        except Exception as e:
            self.logger.error(f"Error en percepción: {e}")
    
    async def _detect_nearby_vehicles(self):
        """Detecta vehículos cercanos al semáforo"""
        # Esta función será integrada con el sistema de vehículos del entorno
        # Por ahora simulamos la detección
        
        # Simular detección de vehículos
        vehicle_count = random.randint(0, 10)
        self.nearby_vehicles = []
        
        for i in range(vehicle_count):
            vehicle = {
                "id": f"vehicle_{i}",
                "distance": random.uniform(5, 100),  # metros
                "speed": random.uniform(0, 50),  # km/h
                "direction": random.choice(list(TrafficDirection)),
                "emergency": random.random() < 0.05,  # 5% probabilidad
                "waiting_time": random.uniform(0, 120)  # segundos
            }
            self.nearby_vehicles.append(vehicle)
    
    async def _analyze_traffic_flows(self):
        """Analiza los flujos de tráfico en cada dirección"""
        # Agrupar vehículos por dirección
        direction_counts = {direction: 0 for direction in TrafficDirection}
        direction_queues = {direction: 0 for direction in TrafficDirection}
        direction_speeds = {direction: [] for direction in TrafficDirection}
        
        for vehicle in self.nearby_vehicles:
            direction = vehicle["direction"]
            direction_counts[direction] += 1
            
            if vehicle["speed"] < 5.0:  # Vehículo en cola
                direction_queues[direction] += 1
            
            direction_speeds[direction].append(vehicle["speed"])
        
        # Actualizar flujos de tráfico percibidos
        for direction in TrafficDirection:
            avg_speed = sum(direction_speeds[direction]) / max(1, len(direction_speeds[direction]))
            emergency_count = sum(1 for v in self.nearby_vehicles 
                                if v["direction"] == direction and v["emergency"])
            
            self.perceived_traffic[direction] = TrafficFlow(
                direction=direction,
                vehicle_count=direction_counts[direction],
                queue_length=direction_queues[direction],
                average_speed=avg_speed,
                emergency_vehicles=emergency_count,
                density=direction_counts[direction] / 100.0  # Simplified density
            )
    
    async def _assess_environmental_conditions(self):
        """Evalúa las condiciones ambientales"""
        # Simular factores ambientales
        self.weather_factor = random.uniform(0.7, 1.0)  # Factor climático
        self.visibility_factor = random.uniform(0.8, 1.0)  # Factor de visibilidad
        
        # Ajustar tiempos según condiciones
        if self.weather_factor < 0.8:  # Mal clima
            # Extender tiempos para mayor seguridad
            for phase_config in self.data.phase_config.values():
                if phase_config.phase == TrafficLightPhase.YELLOW:
                    phase_config.duration = max(5.0, phase_config.duration * 1.2)
    
    async def _process_priority_events(self):
        """Procesa eventos de alta prioridad"""
        current_time = datetime.now()
        
        # Verificar si hay override activo que debe terminar
        if (self.priority_override_active and self.override_end_time and 
            current_time >= self.override_end_time):
            await self._end_priority_override()
        
        # Procesar nuevos eventos de emergencia
        emergency_vehicles = any(v["emergency"] for v in self.nearby_vehicles)
        
        if emergency_vehicles and not self.priority_override_active:
            await self._activate_emergency_preemption()
        
        # Procesar cola de eventos
        while self.event_queue:
            event = self.event_queue.pop(0)
            await self._handle_event(event)
    
    async def _decide_phase_action(self):
        """Decide la acción a tomar respecto al cambio de fase"""
        if not self.data.is_operational:
            return {"action": "maintain"}
        
        # Si hay override activo, mantener estado actual
        if self.priority_override_active:
            return {"action": "maintain"}
        
        current_phase = self.data.current_phase
        time_in_phase = self.data.time_in_phase
        phase_config = self.data.phase_config.get(current_phase)
        
        if not phase_config:
            return {"action": "maintain"}
        
        # Lógica adaptativa para cambio de fase
        if self.adaptation_enabled:
            return await self._adaptive_phase_decision(phase_config, time_in_phase)
        else:
            return await self._fixed_time_decision(phase_config, time_in_phase)
    
    async def _adaptive_phase_decision(self, phase_config: PhaseConfiguration, 
                                     time_in_phase: float) -> Dict[str, Any]:
        """Toma decisión adaptativa basada en tráfico"""
        current_phase = self.data.current_phase
        
        # Verificar tiempo mínimo
        if time_in_phase < phase_config.min_duration:
            return {"action": "maintain"}
        
        # Analizar demanda de tráfico
        total_waiting = sum(flow.queue_length for flow in self.perceived_traffic.values())
        current_direction_traffic = self._get_current_direction_traffic()
        cross_direction_traffic = self._get_cross_direction_traffic()
        
        # Lógica difusa para decisión
        change_urgency = 0.0
        
        # Factor 1: Tiempo en fase actual
        time_factor = min(1.0, time_in_phase / phase_config.duration)
        
        # Factor 2: Demanda en dirección cruzada
        if cross_direction_traffic > 0:
            cross_demand = cross_direction_traffic / max(1, current_direction_traffic)
            cross_demand = min(2.0, cross_demand)
        else:
            cross_demand = 0.0
        
        # Factor 3: Presencia de colas
        queue_pressure = min(1.0, total_waiting / 20.0)  # Normalizar a 20 vehículos máx
        
        # Combinar factores
        if current_phase == TrafficLightPhase.GREEN:
            # Evaluar si cambiar a amarillo
            change_urgency = (time_factor * 0.4 + cross_demand * 0.4 + queue_pressure * 0.2)
            
            if change_urgency > 0.7 or time_in_phase >= phase_config.max_duration:
                return {"action": "change_phase", "new_phase": "yellow"}
            elif time_in_phase >= phase_config.duration and cross_demand > 0.5:
                return {"action": "change_phase", "new_phase": "yellow"}
                
        elif current_phase == TrafficLightPhase.YELLOW:
            # Amarillo siempre cambia a rojo después del tiempo mínimo
            if time_in_phase >= phase_config.duration:
                return {"action": "change_phase", "new_phase": "red"}
                
        elif current_phase == TrafficLightPhase.RED:
            # Evaluar si cambiar a verde
            if time_in_phase >= phase_config.min_duration:
                green_urgency = (time_factor * 0.3 + cross_demand * 0.5 + queue_pressure * 0.2)
                
                if green_urgency > 0.6 or time_in_phase >= phase_config.max_duration:
                    return {"action": "change_phase", "new_phase": "green"}
        
        return {"action": "maintain"}
    
    async def _fixed_time_decision(self, phase_config: PhaseConfiguration, 
                                 time_in_phase: float) -> Dict[str, Any]:
        """Decisión basada en tiempos fijos"""
        if time_in_phase >= phase_config.duration:
            next_phase = self.data.get_next_phase()
            return {"action": "change_phase", "new_phase": next_phase.value}
        
        return {"action": "maintain"}
    
    def _get_current_direction_traffic(self) -> int:
        """Obtiene el tráfico en las direcciones permitidas actualmente"""
        if self.data.current_phase not in self.data.phase_config:
            return 0
        
        allowed_directions = self.data.phase_config[self.data.current_phase].allowed_directions
        return sum(self.perceived_traffic.get(direction, TrafficFlow(direction)).vehicle_count 
                  for direction in allowed_directions)
    
    def _get_cross_direction_traffic(self) -> int:
        """Obtiene el tráfico en las direcciones cruzadas (no permitidas actualmente)"""
        if self.data.current_phase not in self.data.phase_config:
            return 0
        
        allowed_directions = self.data.phase_config[self.data.current_phase].allowed_directions
        cross_directions = [d for d in TrafficDirection if d not in allowed_directions]
        
        return sum(self.perceived_traffic.get(direction, TrafficFlow(direction)).vehicle_count 
                  for direction in cross_directions)
    
    async def _execute_action(self, decision: Dict[str, Any]):
        """Ejecuta la acción decidida"""
        action = decision.get("action", "maintain")
        
        if action == "change_phase":
            new_phase_str = decision.get("new_phase")
            if new_phase_str:
                try:
                    new_phase = TrafficLightPhase(new_phase_str)
                    await self._change_phase(new_phase)
                except ValueError:
                    self.logger.error(f"Fase inválida: {new_phase_str}")
        
        elif action == "maintain":
            # Mantener fase actual, no hacer nada
            pass
    
    async def _change_phase(self, new_phase: TrafficLightPhase):
        """Cambia la fase del semáforo"""
        old_phase = self.data.current_phase
        
        # Registrar cambio en historial
        self.data.add_phase_record(
            old_phase, 
            self.data.time_in_phase, 
            self._get_current_direction_traffic()
        )
        
        # Actualizar estado
        self.data.current_phase = new_phase
        self.data.phase_start_time = datetime.now()
        self.data.time_in_phase = 0.0
        self.data.cycles_completed += 1
        
        self.logger.info(f"Fase cambiada: {old_phase.value} -> {new_phase.value}")
        
        # Notificar cambio (para integración con otros sistemas)
        await self._notify_phase_change(old_phase, new_phase)
    
    async def _notify_phase_change(self, old_phase: TrafficLightPhase, 
                                 new_phase: TrafficLightPhase):
        """Notifica cambio de fase a otros componentes del sistema"""
        # Esta función se conectará con el sistema de comunicación
        notification = {
            "event_type": "phase_change",
            "light_id": self.data.light_id,
            "intersection_id": self.data.intersection_id,
            "old_phase": old_phase.value,
            "new_phase": new_phase.value,
            "timestamp": datetime.now().isoformat(),
            "position": {
                "lat": self.data.latitude,
                "lon": self.data.longitude
            }
        }
        
        # TODO: Integrar con sistema de comunicación del multi-agente
        self.logger.debug(f"Notificación de cambio de fase: {notification}")
    
    async def _activate_emergency_preemption(self):
        """Activa la preempción de emergencia"""
        if self.priority_override_active:
            return
        
        self.priority_override_active = True
        self.override_end_time = datetime.now() + timedelta(seconds=60)  # 1 minuto
        
        # Cambiar inmediatamente a verde para dirección de emergencia
        await self._change_phase(TrafficLightPhase.GREEN)
        
        self.logger.warning(f"Preempción de emergencia activada para {self.data.light_id}")
    
    async def _end_priority_override(self):
        """Termina el override de prioridad"""
        self.priority_override_active = False
        self.override_end_time = None
        
        # Volver a operación normal
        self.logger.info(f"Preempción de emergencia terminada para {self.data.light_id}")
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Maneja un evento específico"""
        event_type = event.get("type", "unknown")
        
        if event_type == "manual_override":
            await self._handle_manual_override(event)
        elif event_type == "maintenance_mode":
            await self._handle_maintenance_mode(event)
        elif event_type == "emergency_vehicle":
            await self._handle_emergency_vehicle(event)
        else:
            self.logger.warning(f"Tipo de evento no reconocido: {event_type}")
    
    async def _handle_manual_override(self, event: Dict[str, Any]):
        """Maneja override manual"""
        new_phase_str = event.get("phase")
        duration = event.get("duration", 60)
        
        if new_phase_str:
            try:
                new_phase = TrafficLightPhase(new_phase_str)
                self.data.manual_override = True
                await self._change_phase(new_phase)
                
                # Programar fin del override
                self.override_end_time = datetime.now() + timedelta(seconds=duration)
                
            except ValueError:
                self.logger.error(f"Fase inválida en override manual: {new_phase_str}")
    
    async def _handle_maintenance_mode(self, event: Dict[str, Any]):
        """Maneja modo de mantenimiento"""
        enable = event.get("enable", True)
        
        if enable:
            self.data.maintenance_mode = True
            self.data.is_operational = False
            await self._change_phase(TrafficLightPhase.FLASHING_YELLOW)
        else:
            self.data.maintenance_mode = False
            self.data.is_operational = True
            await self._change_phase(TrafficLightPhase.RED)  # Reiniciar con rojo
    
    async def _handle_emergency_vehicle(self, event: Dict[str, Any]):
        """Maneja presencia de vehículo de emergencia"""
        direction = event.get("direction")
        if direction:
            await self._activate_emergency_preemption()
    
    async def _update_metrics(self):
        """Actualiza métricas de rendimiento"""
        try:
            # Actualizar conteo de vehículos servidos
            current_traffic = self._get_current_direction_traffic()
            self.data.total_vehicles_served += current_traffic
            
            # Calcular tiempo de espera promedio
            waiting_vehicles = [v for v in self.nearby_vehicles if v["speed"] < 5.0]
            if waiting_vehicles:
                avg_wait = sum(v["waiting_time"] for v in waiting_vehicles) / len(waiting_vehicles)
                self.data.average_wait_time = (self.data.average_wait_time * 0.8 + avg_wait * 0.2)
            
            # Calcular score de eficiencia (simplificado)
            if self.data.average_wait_time > 0:
                self.data.efficiency_score = max(0.0, 1.0 - (self.data.average_wait_time / 120.0))
            else:
                self.data.efficiency_score = 1.0
                
        except Exception as e:
            self.logger.error(f"Error actualizando métricas: {e}")
    
    # Métodos públicos para interfaz externa
    
    def get_current_state(self) -> Dict[str, Any]:
        """Obtiene el estado actual del semáforo"""
        return {
            "light_id": self.data.light_id,
            "node_id": self.data.node_id,
            "current_phase": self.data.current_phase.value,
            "time_in_phase": self.data.time_in_phase,
            "is_operational": self.data.is_operational,
            "emergency_override": self.priority_override_active,
            "manual_override": self.data.manual_override,
            "maintenance_mode": self.data.maintenance_mode,
            "position": {
                "lat": self.data.latitude,
                "lon": self.data.longitude
            },
            "metrics": {
                "efficiency_score": self.data.efficiency_score,
                "average_wait_time": self.data.average_wait_time,
                "cycles_completed": self.data.cycles_completed,
                "vehicles_served": self.data.total_vehicles_served
            }
        }
    
    def get_traffic_data(self) -> Dict[str, Any]:
        """Obtiene datos de tráfico percibidos"""
        return {
            "nearby_vehicles": len(self.nearby_vehicles),
            "emergency_vehicles": sum(1 for v in self.nearby_vehicles if v["emergency"]),
            "traffic_flows": {
                direction.value: {
                    "vehicle_count": flow.vehicle_count,
                    "queue_length": flow.queue_length,
                    "average_speed": flow.average_speed
                }
                for direction, flow in self.perceived_traffic.items()
            }
        }
    
    async def force_phase_change(self, new_phase: str, duration: int = 60):
        """Fuerza un cambio de fase (para control externo)"""
        try:
            phase = TrafficLightPhase(new_phase)
            event = {
                "type": "manual_override",
                "phase": new_phase,
                "duration": duration
            }
            self.event_queue.append(event)
            
        except ValueError:
            raise ValueError(f"Fase inválida: {new_phase}")
    
    async def set_maintenance_mode(self, enable: bool):
        """Activa/desactiva modo de mantenimiento"""
        event = {
            "type": "maintenance_mode",
            "enable": enable
        }
        self.event_queue.append(event)
    
    def update_vehicle_data(self, vehicles: List[Dict[str, Any]]):
        """Actualiza datos de vehículos desde el entorno externo"""
        self.nearby_vehicles = vehicles
        
        # Reanalizar flujos de tráfico
        asyncio.create_task(self._analyze_traffic_flows())
