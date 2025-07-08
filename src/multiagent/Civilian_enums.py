from enum import Enum


class CivilianBehavior(Enum):
    """Tipos de comportamiento de vehículos civiles"""
    CONSERVATIVE = "conservador"
    NORMAL = "normal" 
    AGGRESSIVE = "agresivo"
    CAUTIOUS = "cauteloso"
    RECKLESS = "temerario"


class MovementState(Enum):
    """Estados de movimiento del vehículo civil"""
    IDLE = "parado"
    MOVING = "movimiento"
    WAITING = "esperando"
    PARKING = "estacionando"
    AVOIDING = "evitando"
    EMERGENCY_STOP = "parada_emergencia"