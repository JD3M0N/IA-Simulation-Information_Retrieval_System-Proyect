from datetime import datetime
from typing import Any, Dict, List
import requests

from crawler.base_crawler import BaseCrawler


class OpenMeteoCrawler(BaseCrawler):
    """
    Crawler que utiliza la API pública de Open-Meteo para obtener
    probabilidades de lluvia y velocidad del viento, tanto horarias como diarias.
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        retries: int = 3,
        delay: float = 1.0,
    ):
        """
        :param lat: Latitud del punto geográfico
        :param lon: Longitud del punto geográfico
        :param retries: Número de reintentos en caso de error de red
        :param delay: Pausa entre reintentos
        """
        self.lat = lat
        self.lon = lon

        # URL base de Open-Meteo
        base_url = "https://api.open-meteo.com/v1/forecast"

        super().__init__(base_url=base_url, retries=retries, delay=delay)

    def crawl(self) -> Dict[str, Any]:
        """
        Construye los parámetros de la consulta y obtiene la respuesta de la API.
        """
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": "precipitation_probability,wind_speed_10m",
            "daily": "precipitation_probability_max,wind_speed_10m_max",
            "timezone": "auto"
        }

        response = self.fetch(params=params)
        if not response:
            return {}

        return self.parse(response)

    def parse(self, response: requests.Response) -> Dict[str, Any]:
        """
        Extrae los datos útiles del JSON de respuesta.
        """
        data = response.json()

        # Procesamos datos horarios para hoy
        hourly = []
        if "hourly" in data:
            times = data["hourly"]["time"]
            rain_probs = data["hourly"]["precipitation_probability"]
            wind_speeds = data["hourly"]["wind_speed_10m"]

            for t, rain, wind in zip(times, rain_probs, wind_speeds):
                if self._is_today(t):
                    hourly.append({
                        "time": t,
                        "rain_probability": rain,
                        "wind_speed": wind
                    })

        # Procesamos los datos diarios (próximos 5 días)
        daily = []
        if "daily" in data:
            dates = data["daily"]["time"]
            rain_probs = data["daily"]["precipitation_probability_max"]
            wind_speeds = data["daily"]["wind_speed_10m_max"]

            for date, rain, wind in zip(dates[:5], rain_probs[:5], wind_speeds[:5]):
                daily.append({
                    "date": date,
                    "rain_probability_max": rain,
                    "wind_speed_max": wind
                })

        return {
            "coordinates": {"lat": self.lat, "lon": self.lon},
            "hourly_today": hourly,
            "daily_forecast": daily,
        }

    def _is_today(self, iso_datetime: str) -> bool:
        """
        Determina si una fecha en formato ISO (YYYY-MM-DDTHH:MM) corresponde al día de hoy.
        """
        try:
            date_part = iso_datetime.split("T")[0]
            return date_part == datetime.now().date().isoformat()
        except Exception:
            return False
