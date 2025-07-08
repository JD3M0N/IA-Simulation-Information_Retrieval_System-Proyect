"""
Configuración de parámetros para la simulación optimizada
"""

# Configuración de simulación finita
SIMULATION_CONFIG = {
    # Parámetros principales
    "max_epochs": 500,              # Número máximo de épocas
    "metrics_report_interval": 20,  # Imprimir métricas cada N épocas
    "step_delay": 0.001,            # Delay entre pasos (segundos) - ULTRA RÁPIDO
    
    # Parámetros de entorno
    "time_step": 5.0,               # Segundos por paso de simulación - MÁS AUMENTADO
    "simulation_speed": 5.0,        # Multiplicador de velocidad - MÁS AUMENTADO
    "num_vehicles": 10,             # Número de vehículos civiles - REDUCIDO PARA VELOCIDAD
    
    # Parámetros climáticos
    "weather_update_interval": 900,      # Intervalo de actualización climática (segundos) - AUMENTADO
    "extreme_weather_probability": 0.01, # Probabilidad de eventos climáticos extremos - REDUCIDO
    
    # Parámetros de eventos de tráfico
    "traffic_event_probability": 0.002,  # Probabilidad muy baja de eventos de tráfico - REDUCIDO
    
    # Parámetros de vehículos
    "vehicle_speed_multiplier": 2.0,     # Multiplicador de velocidad para vehículos - MÁS AUMENTADO
    "min_vehicle_speed": 75.0,           # Velocidad mínima para movimiento visible - MÁS AUMENTADO
    
    # Parámetros de métricas
    "metrics_calculation_interval": {
        "speed": 15,          # Calcular velocidad promedio cada 15 pasos - AUMENTADO
        "congestion": 20,     # Calcular congestión cada 20 pasos - AUMENTADO
        "distance": 10        # Actualizar distancia cada 10 pasos - AUMENTADO
    },
    
    # Configuración de BDI
    "bdi_trucks_count": 3,           # Número de camiones BDI de demostración
    "bdi_delivery_locations": 5,     # Número de ubicaciones de entrega por camión
    
    # Configuración de visualización
    "show_detailed_logs": False,     # Mostrar logs detallados
    "show_bdi_status_interval": 40,  # Mostrar estado BDI cada N épocas
}

# Configuración para modo debug (simulación más lenta pero con más detalles)
DEBUG_CONFIG = {
    **SIMULATION_CONFIG,
    "max_epochs": 100,
    "metrics_report_interval": 10,
    "step_delay": 0.05,
    "show_detailed_logs": True,
    "show_bdi_status_interval": 20,
}

# Configuración para modo rápido (simulación muy rápida con métricas mínimas)
FAST_CONFIG = {
    **SIMULATION_CONFIG,
    "max_epochs": 1000,
    "metrics_report_interval": 50,
    "step_delay": 0.001,           # MÁS RÁPIDO AÚN
    "time_step": 5.0,              # PASOS MÁS LARGOS
    "simulation_speed": 5.0,       # VELOCIDAD MÁS ALTA
    "num_vehicles": 10,            # MENOS VEHÍCULOS
    "weather_update_interval": 1800, # MENOS ACTUALIZACIONES CLIMÁTICAS
    "traffic_event_probability": 0.0005, # MÍNIMOS EVENTOS
    "vehicle_speed_multiplier": 2.0,     # VEHÍCULOS MÁS RÁPIDOS
    "min_vehicle_speed": 60.0,           # VELOCIDAD MÍNIMA MÁS ALTA
    "show_detailed_logs": False,
    "show_bdi_status_interval": 100,
}

def get_config(mode: str = "normal"):
    """
    Obtiene la configuración según el modo especificado
    
    Args:
        mode: "normal", "debug" o "fast"
    
    Returns:
        Diccionario con la configuración correspondiente
    """
    configs = {
        "normal": SIMULATION_CONFIG,
        "debug": DEBUG_CONFIG,
        "fast": FAST_CONFIG
    }
    
    return configs.get(mode, SIMULATION_CONFIG)
