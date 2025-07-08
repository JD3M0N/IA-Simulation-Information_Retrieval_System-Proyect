
from enum import Enum


class WeatherCondition(Enum):
    """Condiciones climáticas posibles"""
    CLEAR = "despejado"
    CLOUDY = "nublado"
    LIGHT_RAIN = "lluvia_ligera"
    HEAVY_RAIN = "lluvia_fuerte"
    STORM = "tormenta"
    FOG = "niebla"
    EXTREME_HEAT = "calor_extremo"


class RoadCondition(Enum):
    """Condiciones de las vías"""
    EXCELLENT = "excelente"
    GOOD = "buena"
    REGULAR = "regular"
    BAD = "mala"
    CLOSED = "cerrada"


class TrafficEventType(Enum):
    """Tipos de eventos de tráfico"""
    ACCIDENT = "accidente"
    CONSTRUCTION = "construccion"
    PROTEST = "protesta"
    SPECIAL_EVENT = "evento_especial"
    VEHICLE_BREAKDOWN = "averia_vehiculo"
    EMERGENCY = "emergencia"
    ROAD_CLOSURE = "cierre_vial"
    FLOODING = "inundacion"
