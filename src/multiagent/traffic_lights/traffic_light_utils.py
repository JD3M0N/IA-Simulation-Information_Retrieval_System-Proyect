"""
Utilidades para el Sistema de Semáforos
Funciones auxiliares para cálculos y análisis de patrones de tráfico
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict, deque

from .traffic_light_models import (
    TrafficLightPhase, TrafficDirection, TrafficFlow,
    TrafficLightData, PhaseConfiguration
)


def calculate_phase_timing(traffic_flows: Dict[TrafficDirection, TrafficFlow],
                          current_phase: TrafficLightPhase,
                          base_timings: Dict[TrafficLightPhase, float],
                          max_extension: float = 20.0) -> float:
    """
    Calcula el tiempo óptimo para una fase basado en flujos de tráfico
    
    Args:
        traffic_flows: Flujos de tráfico por dirección
        current_phase: Fase actual del semáforo
        base_timings: Tiempos base por fase
        max_extension: Extensión máxima permitida en segundos
        
    Returns:
        float: Tiempo optimizado para la fase en segundos
    """
    base_time = base_timings.get(current_phase, 30.0)
    
    if current_phase == TrafficLightPhase.YELLOW:
        # El amarillo no se extiende
        return base_time
    
    # Obtener direcciones relevantes para la fase actual
    if current_phase == TrafficLightPhase.GREEN:
        # Simplificación: Norte-Sur para verde
        relevant_directions = [TrafficDirection.NORTH, TrafficDirection.SOUTH]
    else:
        # Rojo permite Este-Oeste
        relevant_directions = [TrafficDirection.EAST, TrafficDirection.WEST]
    
    # Calcular demanda total en direcciones relevantes
    total_demand = 0
    total_queue = 0
    
    for direction in relevant_directions:
        if direction in traffic_flows:
            flow = traffic_flows[direction]
            total_demand += flow.vehicle_count
            total_queue += flow.queue_length
    
    # Calcular factor de extensión basado en demanda
    if total_demand == 0:
        return base_time
    
    # Factor de cola (más cola = más tiempo)
    queue_factor = min(2.0, total_queue / 10.0)
    
    # Factor de densidad
    density_factor = min(1.5, total_demand / 15.0)
    
    # Combinar factores
    extension_factor = (queue_factor * 0.6 + density_factor * 0.4)
    extension = min(max_extension, base_time * extension_factor * 0.5)
    
    return base_time + extension


def detect_traffic_patterns(historical_data: List[Dict[str, Any]], 
                          time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
    """
    Detecta patrones de tráfico en datos históricos
    
    Args:
        historical_data: Lista de registros históricos
        time_window: Ventana de tiempo para análisis
        
    Returns:
        Dict con patrones detectados
    """
    if not historical_data:
        return {"patterns": [], "peak_hours": [], "flow_trends": {}}
    
    # Agrupar datos por hora
    hourly_data = defaultdict(list)
    daily_patterns = defaultdict(list)
    
    for record in historical_data:
        timestamp = datetime.fromisoformat(record.get("timestamp", ""))
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        hourly_data[hour].append(record)
        daily_patterns[day_of_week].append(record)
    
    # Detectar horas pico
    peak_hours = []
    hourly_volumes = {}
    
    for hour, records in hourly_data.items():
        total_volume = sum(r.get("vehicle_count", 0) for r in records)
        hourly_volumes[hour] = total_volume
    
    # Identificar top 3 horas con más tráfico
    sorted_hours = sorted(hourly_volumes.items(), key=lambda x: x[1], reverse=True)
    peak_hours = [hour for hour, volume in sorted_hours[:3]]
    
    # Analizar tendencias de flujo
    flow_trends = {}
    for direction in TrafficDirection:
        direction_data = []
        for record in historical_data[-100:]:  # Últimos 100 registros
            flow_data = record.get("traffic_flows", {}).get(direction.value, {})
            if flow_data:
                direction_data.append(flow_data.get("vehicle_count", 0))
        
        if direction_data:
            trend = calculate_trend(direction_data)
            flow_trends[direction.value] = {
                "trend": trend,
                "average": statistics.mean(direction_data),
                "variance": statistics.variance(direction_data) if len(direction_data) > 1 else 0
            }
    
    # Detectar patrones específicos
    patterns = []
    
    # Patrón de rush hour
    if any(hourly_volumes.get(hour, 0) > statistics.mean(hourly_volumes.values()) * 1.5 
           for hour in [7, 8, 17, 18]):
        patterns.append({
            "type": "rush_hour",
            "description": "Patrón de hora pico detectado",
            "peak_hours": [h for h in [7, 8, 17, 18] if h in peak_hours]
        })
    
    # Patrón de tráfico nocturno bajo
    night_volume = sum(hourly_volumes.get(hour, 0) for hour in range(22, 24) + list(range(0, 6)))
    day_volume = sum(hourly_volumes.get(hour, 0) for hour in range(6, 22))
    
    if night_volume < day_volume * 0.2:
        patterns.append({
            "type": "low_night_traffic",
            "description": "Tráfico nocturno significativamente bajo",
            "ratio": night_volume / max(day_volume, 1)
        })
    
    return {
        "patterns": patterns,
        "peak_hours": peak_hours,
        "flow_trends": flow_trends,
        "hourly_volumes": hourly_volumes
    }


def calculate_trend(data: List[float]) -> str:
    """Calcula la tendencia de una serie de datos"""
    if len(data) < 2:
        return "stable"
    
    # Regresión lineal simple
    x = list(range(len(data)))
    n = len(data)
    
    sum_x = sum(x)
    sum_y = sum(data)
    sum_xy = sum(x[i] * data[i] for i in range(n))
    sum_x2 = sum(x[i] ** 2 for i in range(n))
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    
    if slope > 0.1:
        return "increasing"
    elif slope < -0.1:
        return "decreasing"
    else:
        return "stable"


def calculate_optimal_cycle_time(intersection_flows: Dict[TrafficDirection, TrafficFlow],
                               min_cycle: float = 60.0,
                               max_cycle: float = 180.0) -> float:
    """
    Calcula el tiempo de ciclo óptimo usando el método de Webster
    
    Args:
        intersection_flows: Flujos de tráfico en la intersección
        min_cycle: Tiempo mínimo de ciclo
        max_cycle: Tiempo máximo de ciclo
        
    Returns:
        float: Tiempo de ciclo óptimo en segundos
    """
    # Constantes de Webster
    L = 10.0  # Tiempo perdido total por ciclo (segundos)
    
    # Calcular flujos de saturación y tasas de flujo
    critical_ratios = []
    
    # Agrupar direcciones por fases
    phase_groups = [
        [TrafficDirection.NORTH, TrafficDirection.SOUTH],
        [TrafficDirection.EAST, TrafficDirection.WEST]
    ]
    
    for group in phase_groups:
        group_flow = 0
        group_saturation = 0
        
        for direction in group:
            if direction in intersection_flows:
                flow = intersection_flows[direction]
                # Flujo actual (vehículos por segundo)
                current_flow = flow.flow_rate / 60.0 if flow.flow_rate > 0 else 0
                # Flujo de saturación estimado (vehículos por segundo)
                saturation_flow = 0.5  # Aproximadamente 1800 veh/h = 0.5 veh/s
                
                group_flow += current_flow
                group_saturation += saturation_flow
        
        if group_saturation > 0:
            critical_ratios.append(group_flow / group_saturation)
    
    # Suma de ratios críticos
    Y = sum(critical_ratios)
    
    if Y >= 1.0:
        # Intersección sobresaturada, usar ciclo máximo
        return max_cycle
    
    # Fórmula de Webster
    optimal_cycle = (1.5 * L + 5) / (1 - Y)
    
    # Limitar entre mínimo y máximo
    return max(min_cycle, min(optimal_cycle, max_cycle))


def calculate_green_splits(intersection_flows: Dict[TrafficDirection, TrafficFlow],
                          cycle_time: float) -> Dict[str, float]:
    """
    Calcula la división de tiempo verde entre fases
    
    Args:
        intersection_flows: Flujos de tráfico
        cycle_time: Tiempo total del ciclo
        
    Returns:
        Dict con tiempos de verde por fase
    """
    # Tiempos fijos
    yellow_time = 4.0
    all_red_time = 2.0
    lost_time = (yellow_time + all_red_time) * 2  # 2 fases
    
    effective_green = cycle_time - lost_time
    
    # Calcular demandas por fase
    phase_demands = {}
    
    # Fase Norte-Sur
    ns_demand = 0
    if TrafficDirection.NORTH in intersection_flows:
        ns_demand += intersection_flows[TrafficDirection.NORTH].flow_rate
    if TrafficDirection.SOUTH in intersection_flows:
        ns_demand += intersection_flows[TrafficDirection.SOUTH].flow_rate
    
    # Fase Este-Oeste
    ew_demand = 0
    if TrafficDirection.EAST in intersection_flows:
        ew_demand += intersection_flows[TrafficDirection.EAST].flow_rate
    if TrafficDirection.WEST in intersection_flows:
        ew_demand += intersection_flows[TrafficDirection.WEST].flow_rate
    
    total_demand = ns_demand + ew_demand
    
    if total_demand == 0:
        # División igual si no hay demanda
        green_ns = effective_green / 2
        green_ew = effective_green / 2
    else:
        # División proporcional a la demanda
        green_ns = (ns_demand / total_demand) * effective_green
        green_ew = (ew_demand / total_demand) * effective_green
    
    return {
        "north_south_green": max(15.0, green_ns),  # Mínimo 15 segundos
        "east_west_green": max(15.0, green_ew),
        "yellow": yellow_time,
        "all_red": all_red_time
    }


def calculate_intersection_delay(intersection_flows: Dict[TrafficDirection, TrafficFlow],
                               phase_timings: Dict[str, float],
                               cycle_time: float) -> float:
    """
    Calcula el retraso promedio en una intersección usando HCM
    
    Args:
        intersection_flows: Flujos de tráfico
        phase_timings: Tiempos de cada fase
        cycle_time: Tiempo del ciclo
        
    Returns:
        float: Retraso promedio en segundos por vehículo
    """
    total_delay = 0.0
    total_volume = 0
    
    # Calcular retraso por dirección
    for direction, flow in intersection_flows.items():
        if flow.vehicle_count == 0:
            continue
        
        # Determinar tiempo de verde efectivo para esta dirección
        if direction in [TrafficDirection.NORTH, TrafficDirection.SOUTH]:
            effective_green = phase_timings.get("north_south_green", 30.0)
        else:
            effective_green = phase_timings.get("east_west_green", 30.0)
        
        # Parámetros para cálculo de retraso
        volume = flow.flow_rate / 60.0  # veh/s
        saturation_flow = 0.5  # veh/s
        capacity = saturation_flow * (effective_green / cycle_time)
        
        # Ratio volumen/capacidad
        X = volume / max(capacity, 0.01)
        
        # Retraso uniforme (HCM)
        d1 = (0.5 * cycle_time * (1 - effective_green / cycle_time) ** 2) / (1 - min(1.0, X) * effective_green / cycle_time)
        
        # Retraso por overflow (simplificado)
        d2 = 0
        if X > 1.0:
            d2 = 900 * ((X - 1) + math.sqrt((X - 1) ** 2 + (8 * X) / (capacity * cycle_time)))
        
        direction_delay = d1 + d2
        total_delay += direction_delay * flow.vehicle_count
        total_volume += flow.vehicle_count
    
    return total_delay / max(total_volume, 1)


def calculate_level_of_service(average_delay: float) -> str:
    """
    Calcula el nivel de servicio basado en el retraso promedio
    
    Args:
        average_delay: Retraso promedio en segundos por vehículo
        
    Returns:
        str: Nivel de servicio (A-F)
    """
    if average_delay <= 10:
        return "A"
    elif average_delay <= 20:
        return "B"
    elif average_delay <= 35:
        return "C"
    elif average_delay <= 55:
        return "D"
    elif average_delay <= 80:
        return "E"
    else:
        return "F"


def estimate_fuel_consumption(intersection_flows: Dict[TrafficDirection, TrafficFlow],
                            average_delay: float) -> float:
    """
    Estima el consumo de combustible adicional debido a paradas
    
    Args:
        intersection_flows: Flujos de tráfico
        average_delay: Retraso promedio por vehículo
        
    Returns:
        float: Consumo adicional estimado en litros por hora
    """
    # Parámetros de consumo
    idle_consumption = 0.5  # L/h en ralentí
    acceleration_penalty = 0.1  # L por parada/arranque
    
    total_vehicles = sum(flow.vehicle_count for flow in intersection_flows.values())
    
    if total_vehicles == 0:
        return 0.0
    
    # Estimar número de paradas basado en el retraso
    stops_per_vehicle = min(average_delay / 30.0, 3.0)  # Máximo 3 paradas
    
    # Consumo por ralentí
    idle_hours = (average_delay / 3600.0) * total_vehicles
    idle_fuel = idle_hours * idle_consumption
    
    # Consumo por paradas/arranques
    total_stops = stops_per_vehicle * total_vehicles
    stop_fuel = total_stops * acceleration_penalty
    
    return idle_fuel + stop_fuel


def calculate_coordination_benefit(before_delay: float, after_delay: float,
                                 vehicle_volume: int) -> Dict[str, float]:
    """
    Calcula los beneficios de la coordinación de semáforos
    
    Args:
        before_delay: Retraso antes de coordinación
        after_delay: Retraso después de coordinación
        vehicle_volume: Volumen de vehículos por hora
        
    Returns:
        Dict con métricas de beneficio
    """
    time_savings = before_delay - after_delay  # segundos por vehículo
    
    if vehicle_volume == 0:
        return {"time_savings": 0, "fuel_savings": 0, "emission_reduction": 0}
    
    # Beneficios totales por hora
    total_time_savings = (time_savings * vehicle_volume) / 3600.0  # horas
    
    # Estimación de ahorro de combustible
    fuel_savings = estimate_fuel_consumption({}, before_delay) - estimate_fuel_consumption({}, after_delay)
    
    # Estimación de reducción de emisiones (kg CO2/h)
    # Factor aproximado: 2.3 kg CO2 por litro de gasolina
    emission_reduction = fuel_savings * 2.3
    
    return {
        "time_savings_hours": total_time_savings,
        "fuel_savings_liters": fuel_savings,
        "emission_reduction_kg": emission_reduction,
        "delay_reduction_percent": ((before_delay - after_delay) / max(before_delay, 0.1)) * 100
    }


def validate_phase_configuration(phase_config: Dict[TrafficLightPhase, PhaseConfiguration]) -> List[str]:
    """
    Valida la configuración de fases de un semáforo
    
    Args:
        phase_config: Configuración de fases
        
    Returns:
        List[str]: Lista de errores encontrados
    """
    errors = []
    
    # Verificar fases mínimas requeridas
    required_phases = {TrafficLightPhase.GREEN, TrafficLightPhase.YELLOW, TrafficLightPhase.RED}
    existing_phases = set(phase_config.keys())
    
    missing_phases = required_phases - existing_phases
    if missing_phases:
        errors.append(f"Faltan fases requeridas: {[p.value for p in missing_phases]}")
    
    # Verificar duraciones válidas
    for phase, config in phase_config.items():
        if config.duration <= 0:
            errors.append(f"Duración inválida para fase {phase.value}: {config.duration}")
        
        if config.min_duration < 0:
            errors.append(f"Duración mínima inválida para fase {phase.value}: {config.min_duration}")
        
        if config.max_duration <= config.min_duration:
            errors.append(f"Duración máxima debe ser mayor que mínima para fase {phase.value}")
        
        if config.duration < config.min_duration or config.duration > config.max_duration:
            errors.append(f"Duración fuera de rango para fase {phase.value}")
    
    # Verificar que amarillo sea corto
    if TrafficLightPhase.YELLOW in phase_config:
        yellow_config = phase_config[TrafficLightPhase.YELLOW]
        if yellow_config.duration > 10.0:
            errors.append("Duración de amarillo excesivamente larga")
        if yellow_config.duration < 3.0:
            errors.append("Duración de amarillo muy corta")
    
    return errors


def generate_phase_schedule(flows: Dict[TrafficDirection, TrafficFlow],
                          schedule_duration: timedelta = timedelta(hours=24)) -> List[Dict[str, Any]]:
    """
    Genera un horario de fases optimizado para un período
    
    Args:
        flows: Flujos de tráfico históricos o predichos
        schedule_duration: Duración del horario a generar
        
    Returns:
        List con horario de configuraciones
    """
    schedule = []
    current_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = current_time + schedule_duration
    
    # Configuraciones por período del día
    periods = [
        {"hours": (0, 6), "description": "Madrugada", "cycle_time": 60, "green_ratio": 0.4},
        {"hours": (6, 9), "description": "Hora pico mañana", "cycle_time": 120, "green_ratio": 0.6},
        {"hours": (9, 16), "description": "Día normal", "cycle_time": 90, "green_ratio": 0.5},
        {"hours": (16, 19), "description": "Hora pico tarde", "cycle_time": 120, "green_ratio": 0.6},
        {"hours": (19, 22), "description": "Noche temprana", "cycle_time": 80, "green_ratio": 0.45},
        {"hours": (22, 24), "description": "Noche", "cycle_time": 60, "green_ratio": 0.4}
    ]
    
    while current_time < end_time:
        hour = current_time.hour
        
        # Encontrar período correspondiente
        period = None
        for p in periods:
            start_hour, end_hour = p["hours"]
            if start_hour <= hour < end_hour or (start_hour > end_hour and (hour >= start_hour or hour < end_hour)):
                period = p
                break
        
        if period:
            cycle_time = period["cycle_time"]
            green_ratio = period["green_ratio"]
            
            # Calcular tiempos de fase
            green_time = cycle_time * green_ratio
            yellow_time = 4.0
            red_time = cycle_time - green_time - yellow_time
            
            schedule_entry = {
                "start_time": current_time.isoformat(),
                "end_time": (current_time + timedelta(hours=1)).isoformat(),
                "period": period["description"],
                "configuration": {
                    "cycle_time": cycle_time,
                    "green_duration": green_time,
                    "yellow_duration": yellow_time,
                    "red_duration": red_time,
                    "adaptive": period.get("adaptive", True)
                }
            }
            schedule.append(schedule_entry)
        
        current_time += timedelta(hours=1)
    
    return schedule
