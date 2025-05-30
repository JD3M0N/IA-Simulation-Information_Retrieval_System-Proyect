import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

from base_crawler import BaseCrawler


class WeatherBCrawler(BaseCrawler):
    """
    Crawler para OpenWeatherMap que obtiene clima actual y pronósticos
    (horario y diario) para una latitud/longitud dada.
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        api_key: str | None = None,
        units: str = "metric",
        lang: str = "es",
        retries: int = 5,
        delay: float = 3.0,
    ):
        """
        :param lat: Latitud del punto de interés
        :param lon: Longitud del punto de interés
        :param api_key: API key de OpenWeatherMap (opcional, se busca en env)
        :param units: 'metric', 'imperial' o 'standard'
        :param lang: Idioma ('es' para español)
        :param retries: Reintentos ante fallo de red
        :param delay: Espera entre reintentos en segundos
        """
        # 1. API key segura
        self.api_key = api_key or os.getenv("OWM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Se requiere una API key de OpenWeatherMap. "
                "Defínela en OWM_API_KEY o pásala por parámetro."
            )

        # 2. Parámetros de ubicación y formato
        self.lat = lat
        self.lon = lon
        self.units = units
        self.lang = lang

        # 3. Apuntamos al endpoint OneCall
        base_url = "https://api.openweathermap.org/data/2.5/onecall"

        # 4. Llamamos al constructor de BaseCrawler
        super().__init__(base_url=base_url, retries=retries, delay=delay)

    # Paso orquestador: construir params → fetch → parse
    def crawl(self) -> Dict[str, Any]:
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key,
            "units": self.units,
            "lang": self.lang,
            # Excluimos lo que no necesitamos para aligerar la respuesta
            "exclude": "minutely,alerts",
        }

        response = self.fetch(params=params)
        if not response:
            return {}

        return self.parse(response)

    # Limpieza y normalización del JSON
    def parse(self, response: requests.Response) -> Dict[str, Any]:
        data = response.json()

        def ts_to_iso(ts: int) -> str:
            """Convierte timestamp Unix → string ISO-8601 en UTC."""
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        parsed: Dict[str, Any] = {
            "coordinates": {"lat": data.get("lat"), "lon": data.get("lon")},
            "timezone": data.get("timezone"),
            # -------- Clima actual --------
            "current": {
                "timestamp": ts_to_iso(data["current"]["dt"]),
                "temp": data["current"]["temp"],
                "feels_like": data["current"]["feels_like"],
                "humidity": data["current"]["humidity"],
                "wind_speed": data["current"]["wind_speed"],
                "description": data["current"]["weather"][0]["description"],
                "icon": data["current"]["weather"][0]["icon"],
            },
            # -------- Próximas 24 horas --------
            "hourly": [
                {
                    "timestamp": ts_to_iso(h["dt"]),
                    "temp": h["temp"],
                    "pop": h.get("pop", 0),           # prob. precipitación
                    "description": h["weather"][0]["description"],
                }
                for h in data.get("hourly", [])[:24]
            ],
            # -------- Próximos 7 días --------
            "daily": [
                {
                    "date": ts_to_iso(d["dt"])[:10],  # YYYY-MM-DD
                    "temp_min": d["temp"]["min"],
                    "temp_max": d["temp"]["max"],
                    "description": d["weather"][0]["description"],
                    "pop": d.get("pop", 0),
                }
                for d in data.get("daily", [])[:7]
            ],
        }

        return parsed
    


