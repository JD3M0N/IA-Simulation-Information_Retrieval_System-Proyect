"""
Agente de Clima para Simulación de Transporte
Maneja la evolución y predicción del clima en el entorno urbano
"""

import sys
import os
import random
import asyncio
import numpy as np
import math
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from src.multiagent.Environment_enums import WeatherCondition

@dataclass
class WeatherForecast:
    """Pronóstico del clima"""
    timestamp: datetime
    condition: WeatherCondition
    temperature: float
    humidity: float
    wind_speed: float
    precipitation: float
    visibility: float
    pressure: float
    confidence: float = 0.85  # Confianza del pronóstico

@dataclass
class WeatherPattern:
    """Patrón climático estacional"""
    season: str  # spring, summer, fall, winter
    avg_temperature: float
    temp_variance: float
    humidity_range: Tuple[float, float]
    common_conditions: List[WeatherCondition]
    extreme_weather_probability: float
    precipitation_probability: float

class WeatherAgent:
    """
    Agente especializado en el manejo del clima
    Genera evolución realista del clima y sus impactos
    """
    
    def __init__(self, agent_id: str = "weather_agent", location: Tuple[float, float] = (0.0, 0.0)):
        # Agent properties
        self.agent_id = agent_id
        self.agent_type = "weather"
        self.state = "active"
        
        # Location (lat, lon)
        self.location = location
        
        # Logger
        self.logger = logging.getLogger(f"WeatherAgent_{agent_id}")
        
        # Weather patterns by season (initialize first)
        self.seasonal_patterns = self._initialize_seasonal_patterns()
        
        # Current weather state
        self.current_weather = self._initialize_weather()
        
        # Weather history for pattern analysis
        self.weather_history: List[Dict[str, Any]] = []
        self.max_history_size = 1440  # 24 horas con updates cada minuto
        
        # Forecast system
        self.forecasts: List[WeatherForecast] = []
        self.max_forecast_hours = 24
        
        # Configuration
        self.config = {
            "update_interval": 60,  # segundos
            "forecast_update_interval": 3600,  # 1 hora
            "extreme_weather_duration": {"min": 30, "max": 180},  # minutos
            "weather_change_probability": 0.15,  # por update
            "seasonal_influence": 0.7,
            "historical_influence": 0.3,
            "climate_change_factor": 1.02,  # Gradual warming trend
        }
        
        # Metrics
        self.metrics = {
            "total_updates": 0,
            "extreme_weather_events": 0,
            "forecast_accuracy": 0.0,
            "temperature_trend": 0.0,
            "precipitation_events": 0,
            "weather_changes": 0
        }
        
        # State management
        self.last_update = datetime.now()
        self.last_forecast_update = datetime.now()
        self.current_extreme_event = None
        self.extreme_event_end_time = None
        
        # Weather impact factors
        self.impact_factors = self._initialize_impact_factors()
        
        self.logger.info(f"WeatherAgent {agent_id} initialized at location {location}")
    
    def _initialize_weather(self) -> Dict[str, Any]:
        """Inicializa el estado del clima basado en la ubicación y estación"""
        current_month = datetime.now().month
        season = self._get_current_season(current_month)
        pattern = self.seasonal_patterns[season]
        
        # Generate realistic initial conditions
        temperature = np.random.normal(pattern.avg_temperature, pattern.temp_variance)
        humidity = np.random.uniform(*pattern.humidity_range)
        condition = np.random.choice(pattern.common_conditions)
        
        return {
            "condition": condition,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": np.random.uniform(5, 25),
            "precipitation": 0.0 if condition in [WeatherCondition.CLEAR, WeatherCondition.CLOUDY] 
                          else np.random.uniform(0.1, 15.0),
            "visibility": 10.0 if condition != WeatherCondition.FOG else np.random.uniform(0.5, 3.0),
            "pressure": np.random.uniform(1000, 1025),
            "timestamp": datetime.now()
        }
    
    def _initialize_seasonal_patterns(self) -> Dict[str, WeatherPattern]:
        """Inicializa patrones climáticos estacionales"""
        return {
            "spring": WeatherPattern(
                season="spring",
                avg_temperature=20.0,
                temp_variance=5.0,
                humidity_range=(50, 75),
                common_conditions=[WeatherCondition.CLEAR, WeatherCondition.CLOUDY, WeatherCondition.LIGHT_RAIN],
                extreme_weather_probability=0.1,
                precipitation_probability=0.3
            ),
            "summer": WeatherPattern(
                season="summer",
                avg_temperature=30.0,
                temp_variance=7.0,
                humidity_range=(60, 85),
                common_conditions=[WeatherCondition.CLEAR, WeatherCondition.EXTREME_HEAT, WeatherCondition.STORM],
                extreme_weather_probability=0.15,
                precipitation_probability=0.2
            ),
            "fall": WeatherPattern(
                season="fall",
                avg_temperature=18.0,
                temp_variance=6.0,
                humidity_range=(55, 80),
                common_conditions=[WeatherCondition.CLOUDY, WeatherCondition.LIGHT_RAIN, WeatherCondition.FOG],
                extreme_weather_probability=0.08,
                precipitation_probability=0.4
            ),
            "winter": WeatherPattern(
                season="winter",
                avg_temperature=12.0,
                temp_variance=4.0,
                humidity_range=(40, 70),
                common_conditions=[WeatherCondition.CLEAR, WeatherCondition.CLOUDY, WeatherCondition.FOG],
                extreme_weather_probability=0.05,
                precipitation_probability=0.25
            )
        }
    
    def _initialize_impact_factors(self) -> Dict[str, Dict[str, float]]:
        """Inicializa factores de impacto del clima en el tráfico"""
        return {
            WeatherCondition.CLEAR.value: {
                "speed_factor": 1.0,
                "visibility_factor": 1.0,
                "accident_risk": 1.0,
                "fuel_consumption": 1.0
            },
            WeatherCondition.CLOUDY.value: {
                "speed_factor": 0.98,
                "visibility_factor": 0.95,
                "accident_risk": 1.05,
                "fuel_consumption": 1.02
            },
            WeatherCondition.LIGHT_RAIN.value: {
                "speed_factor": 0.85,
                "visibility_factor": 0.8,
                "accident_risk": 1.4,
                "fuel_consumption": 1.1
            },
            WeatherCondition.HEAVY_RAIN.value: {
                "speed_factor": 0.65,
                "visibility_factor": 0.6,
                "accident_risk": 2.2,
                "fuel_consumption": 1.25
            },
            WeatherCondition.STORM.value: {
                "speed_factor": 0.45,
                "visibility_factor": 0.4,
                "accident_risk": 3.5,
                "fuel_consumption": 1.4
            },
            WeatherCondition.FOG.value: {
                "speed_factor": 0.5,
                "visibility_factor": 0.3,
                "accident_risk": 2.8,
                "fuel_consumption": 1.15
            },
            WeatherCondition.EXTREME_HEAT.value: {
                "speed_factor": 0.9,
                "visibility_factor": 0.95,
                "accident_risk": 1.3,
                "fuel_consumption": 1.2
            }
        }
    
    def _get_current_season(self, month: int) -> str:
        """Determina la estación actual basada en el mes"""
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall"
        else:
            return "winter"
    
    async def next_step(self, environment_state: Dict[str, Any]):
        """Actualiza el estado del clima"""
        current_time = datetime.now()
        
        # Check if it's time for a weather update
        if (current_time - self.last_update).total_seconds() >= self.config["update_interval"]:
            await self._update_weather(environment_state)
            self.last_update = current_time
        
        # Check if it's time for forecast update
        if (current_time - self.last_forecast_update).total_seconds() >= self.config["forecast_update_interval"]:
            await self._update_forecast()
            self.last_forecast_update = current_time
        
        # Handle extreme weather events
        await self._handle_extreme_weather()
        
        # Update metrics
        self._update_metrics()
    
    async def _update_weather(self, environment_state: Dict[str, Any]):
        """Actualiza las condiciones climáticas actuales"""
        current_time = datetime.now()
        season = self._get_current_season(current_time.month)
        pattern = self.seasonal_patterns[season]
        
        # Store previous state
        self.weather_history.append(self.current_weather.copy())
        if len(self.weather_history) > self.max_history_size:
            self.weather_history.pop(0)
        
        # Determine if weather should change
        should_change = random.random() < self.config["weather_change_probability"]
        
        if should_change:
            await self._evolve_weather_naturally(pattern)
        else:
            await self._maintain_weather_stability()
        
        # Apply external influences
        await self._apply_environmental_influences(environment_state)
        
        # Update timestamp
        self.current_weather["timestamp"] = current_time
        
        self.metrics["total_updates"] += 1
        
        self.logger.debug(f"Weather updated: {self.current_weather['condition'].value}, "
                         f"temp: {self.current_weather['temperature']:.1f}°C")
    
    async def _evolve_weather_naturally(self, pattern: WeatherPattern):
        """Evoluciona el clima de manera natural"""
        # Temperature evolution
        temp_change = np.random.normal(0, 2)
        seasonal_target = pattern.avg_temperature
        
        # Tendency towards seasonal average
        temp_drift = (seasonal_target - self.current_weather["temperature"]) * 0.1
        new_temp = self.current_weather["temperature"] + temp_change + temp_drift
        
        # Apply climate change trend
        new_temp *= self.config["climate_change_factor"]
        
        self.current_weather["temperature"] = np.clip(new_temp, -20, 50)
        
        # Humidity evolution
        humidity_change = np.random.normal(0, 5)
        new_humidity = self.current_weather["humidity"] + humidity_change
        self.current_weather["humidity"] = np.clip(new_humidity, 0, 100)
        
        # Condition evolution based on temperature and humidity
        await self._update_weather_condition(pattern)
        
        # Update other parameters based on condition
        await self._update_secondary_parameters()
        
        self.metrics["weather_changes"] += 1
    
    async def _maintain_weather_stability(self):
        """Mantiene estabilidad en las condiciones actuales"""
        # Small random fluctuations
        self.current_weather["temperature"] += np.random.normal(0, 0.5)
        self.current_weather["humidity"] += np.random.normal(0, 2)
        self.current_weather["wind_speed"] += np.random.normal(0, 1)
        
        # Keep within realistic bounds
        self.current_weather["temperature"] = np.clip(self.current_weather["temperature"], -20, 50)
        self.current_weather["humidity"] = np.clip(self.current_weather["humidity"], 0, 100)
        self.current_weather["wind_speed"] = np.clip(self.current_weather["wind_speed"], 0, 100)
    
    async def _update_weather_condition(self, pattern: WeatherPattern):
        """Actualiza la condición climática principal"""
        temp = self.current_weather["temperature"]
        humidity = self.current_weather["humidity"]
        
        # Probability weights for different conditions
        condition_probs = {}
        
        # Base probabilities from seasonal pattern
        for condition in pattern.common_conditions:
            condition_probs[condition] = 0.3
        
        # Temperature influences
        if temp > 35:
            condition_probs[WeatherCondition.EXTREME_HEAT] = condition_probs.get(WeatherCondition.EXTREME_HEAT, 0) + 0.4
        elif temp < 0:
            # Could add snow condition here
            condition_probs[WeatherCondition.FOG] = condition_probs.get(WeatherCondition.FOG, 0) + 0.2
        
        # Humidity influences
        if humidity > 80:
            condition_probs[WeatherCondition.FOG] = condition_probs.get(WeatherCondition.FOG, 0) + 0.3
            condition_probs[WeatherCondition.LIGHT_RAIN] = condition_probs.get(WeatherCondition.LIGHT_RAIN, 0) + 0.2
        
        if humidity > 90:
            condition_probs[WeatherCondition.HEAVY_RAIN] = condition_probs.get(WeatherCondition.HEAVY_RAIN, 0) + 0.25
            condition_probs[WeatherCondition.STORM] = condition_probs.get(WeatherCondition.STORM, 0) + 0.15
        
        # Pressure influences (if available in history)
        if len(self.weather_history) > 0:
            pressure_trend = self.current_weather["pressure"] - self.weather_history[-1]["pressure"]
            if pressure_trend < -5:  # Rapid pressure drop
                condition_probs[WeatherCondition.STORM] = condition_probs.get(WeatherCondition.STORM, 0) + 0.2
        
        # Normalize probabilities
        total_prob = sum(condition_probs.values())
        if total_prob > 0:
            for condition in condition_probs:
                condition_probs[condition] /= total_prob
            
            # Select new condition
            conditions = list(condition_probs.keys())
            probabilities = list(condition_probs.values())
            
            if conditions:
                self.current_weather["condition"] = np.random.choice(conditions, p=probabilities)
        
        # Check for extreme weather events
        if random.random() < pattern.extreme_weather_probability:
            await self._trigger_extreme_weather_event()
    
    async def _update_secondary_parameters(self):
        """Actualiza parámetros secundarios basados en la condición principal"""
        condition = self.current_weather["condition"]
        
        if condition == WeatherCondition.LIGHT_RAIN:
            self.current_weather["precipitation"] = np.random.uniform(0.5, 5.0)
            self.current_weather["visibility"] = np.random.uniform(5.0, 8.0)
            self.current_weather["wind_speed"] = np.random.uniform(10, 25)
            
        elif condition == WeatherCondition.HEAVY_RAIN:
            self.current_weather["precipitation"] = np.random.uniform(5.0, 20.0)
            self.current_weather["visibility"] = np.random.uniform(2.0, 5.0)
            self.current_weather["wind_speed"] = np.random.uniform(15, 35)
            
        elif condition == WeatherCondition.STORM:
            self.current_weather["precipitation"] = np.random.uniform(15.0, 50.0)
            self.current_weather["visibility"] = np.random.uniform(1.0, 3.0)
            self.current_weather["wind_speed"] = np.random.uniform(40, 80)
            self.current_weather["pressure"] = np.random.uniform(980, 1000)
            
        elif condition == WeatherCondition.FOG:
            self.current_weather["precipitation"] = 0.0
            self.current_weather["visibility"] = np.random.uniform(0.1, 2.0)
            self.current_weather["wind_speed"] = np.random.uniform(0, 10)
            self.current_weather["humidity"] = max(85, self.current_weather["humidity"])
            
        elif condition == WeatherCondition.CLEAR:
            self.current_weather["precipitation"] = 0.0
            self.current_weather["visibility"] = 10.0
            self.current_weather["wind_speed"] = np.random.uniform(5, 20)
            
        elif condition == WeatherCondition.EXTREME_HEAT:
            self.current_weather["precipitation"] = 0.0
            self.current_weather["visibility"] = np.random.uniform(8.0, 10.0)
            self.current_weather["humidity"] = min(40, self.current_weather["humidity"])
    
    async def _apply_environmental_influences(self, environment_state: Dict[str, Any]):
        """Aplica influencias del entorno en el clima"""
        # Urban heat island effect
        if environment_state.get("system_metrics", {}).get("total_vehicles", 0) > 50:
            self.current_weather["temperature"] += 0.5
        
        # Time of day effects
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 18:  # Daytime
            # Solar heating
            self.current_weather["temperature"] += np.random.uniform(0, 2)
        else:  # Nighttime
            # Radiative cooling
            self.current_weather["temperature"] -= np.random.uniform(0, 1.5)
    
    async def _trigger_extreme_weather_event(self):
        """Dispara un evento climático extremo"""
        extreme_conditions = [WeatherCondition.STORM, WeatherCondition.HEAVY_RAIN, 
                            WeatherCondition.EXTREME_HEAT, WeatherCondition.FOG]
        
        self.current_extreme_event = np.random.choice(extreme_conditions)
        duration = np.random.randint(
            self.config["extreme_weather_duration"]["min"],
            self.config["extreme_weather_duration"]["max"]
        )
        
        self.extreme_event_end_time = datetime.now() + timedelta(minutes=duration)
        self.current_weather["condition"] = self.current_extreme_event
        
        self.metrics["extreme_weather_events"] += 1
        
        self.logger.warning(f"Extreme weather event triggered: {self.current_extreme_event.value} "
                           f"for {duration} minutes")
    
    async def _handle_extreme_weather(self):
        """Maneja eventos climáticos extremos activos"""
        if (self.current_extreme_event and self.extreme_event_end_time and 
            datetime.now() >= self.extreme_event_end_time):
            
            self.logger.info(f"Extreme weather event ended: {self.current_extreme_event.value}")
            self.current_extreme_event = None
            self.extreme_event_end_time = None
            
            # Transition to milder condition
            mild_conditions = [WeatherCondition.CLOUDY, WeatherCondition.CLEAR]
            self.current_weather["condition"] = np.random.choice(mild_conditions)
    
    async def _update_forecast(self):
        """Actualiza el pronóstico del clima"""
        self.forecasts.clear()
        
        # Generate forecasts for next 24 hours
        base_time = datetime.now()
        current_condition = self.current_weather["condition"]
        current_temp = self.current_weather["temperature"]
        
        for hours_ahead in range(1, self.max_forecast_hours + 1):
            forecast_time = base_time + timedelta(hours=hours_ahead)
            
            # Predict based on current trends and seasonal patterns
            forecast = await self._generate_forecast(forecast_time, current_condition, current_temp)
            self.forecasts.append(forecast)
    
    async def _generate_forecast(self, forecast_time: datetime, 
                               current_condition: WeatherCondition, 
                               current_temp: float) -> WeatherForecast:
        """Genera un pronóstico específico"""
        season = self._get_current_season(forecast_time.month)
        pattern = self.seasonal_patterns[season]
        
        # Temperature trend
        temp_change = np.random.normal(0, 1) * (forecast_time.hour / 24)
        forecast_temp = current_temp + temp_change
        
        # Condition evolution
        stability = 0.7  # Probability of maintaining condition
        if random.random() < stability:
            forecast_condition = current_condition
        else:
            forecast_condition = np.random.choice(pattern.common_conditions)
        
        # Other parameters
        forecast_humidity = np.random.uniform(*pattern.humidity_range)
        forecast_wind = np.random.uniform(5, 30)
        forecast_precipitation = 0.0
        
        if forecast_condition in [WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN, WeatherCondition.STORM]:
            forecast_precipitation = np.random.uniform(0.5, 20.0)
        
        # Confidence decreases with time
        confidence = max(0.5, 0.95 - (forecast_time.hour / 24) * 0.3)
        
        return WeatherForecast(
            timestamp=forecast_time,
            condition=forecast_condition,
            temperature=forecast_temp,
            humidity=forecast_humidity,
            wind_speed=forecast_wind,
            precipitation=forecast_precipitation,
            visibility=10.0 if forecast_condition != WeatherCondition.FOG else 2.0,
            pressure=np.random.uniform(1005, 1020),
            confidence=confidence
        )
    
    def _update_metrics(self):
        """Actualiza métricas del agente"""
        if len(self.weather_history) >= 2:
            # Calculate temperature trend
            recent_temps = [w["temperature"] for w in self.weather_history[-10:]]
            if len(recent_temps) >= 2:
                self.metrics["temperature_trend"] = np.mean(np.diff(recent_temps))
        
        # Count precipitation events
        if self.current_weather["precipitation"] > 0:
            self.metrics["precipitation_events"] += 1
    
    def get_current_weather(self) -> Dict[str, Any]:
        """Retorna el estado actual del clima"""
        return self.current_weather.copy()
    
    def get_weather_impact_factors(self) -> Dict[str, float]:
        """Retorna factores de impacto del clima actual en el tráfico"""
        condition = self.current_weather["condition"].value
        return self.impact_factors.get(condition, {
            "speed_factor": 1.0,
            "visibility_factor": 1.0,
            "accident_risk": 1.0,
            "fuel_consumption": 1.0
        })
    
    def get_forecast(self, hours_ahead: int = 1) -> Optional[WeatherForecast]:
        """Retorna pronóstico para X horas adelante"""
        if 1 <= hours_ahead <= len(self.forecasts):
            return self.forecasts[hours_ahead - 1]
        return None
    
    def get_weather_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Retorna historial del clima"""
        max_records = min(hours * 60 // self.config["update_interval"], len(self.weather_history))
        return self.weather_history[-max_records:] if max_records > 0 else []
    
    def export_weather_data(self) -> Dict[str, Any]:
        """Exporta datos completos del clima"""
        return {
            "agent_id": self.agent_id,
            "current_weather": self.current_weather,
            "forecasts": [
                {
                    "timestamp": f.timestamp.isoformat(),
                    "condition": f.condition.value,
                    "temperature": f.temperature,
                    "humidity": f.humidity,
                    "wind_speed": f.wind_speed,
                    "precipitation": f.precipitation,
                    "visibility": f.visibility,
                    "pressure": f.pressure,
                    "confidence": f.confidence
                } for f in self.forecasts
            ],
            "impact_factors": self.get_weather_impact_factors(),
            "metrics": self.metrics,
            "history_size": len(self.weather_history)
        }
    
    def __str__(self) -> str:
        condition = self.current_weather["condition"].value
        temp = self.current_weather["temperature"]
        humidity = self.current_weather["humidity"]
        return f"WeatherAgent({condition}, {temp:.1f}°C, {humidity:.1f}% humidity)"
