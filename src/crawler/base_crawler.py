import requests
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseCrawler(ABC):
    """
    Clase base abstracta para todos los crawlers del sistema.
    Define la interfaz y funcionalidad común (HTTP requests, logs, retries).
    """

    def __init__(self, base_url: str, retries: int = 3, delay: float = 1.0):
        """
        Constructor común para inicializar el crawler.

        :param base_url: URL principal del sitio que se quiere crawlear
        :param retries: Número de reintentos si falla la petición
        :param delay: Tiempo (en segundos) entre reintentos
        """
        self.base_url = base_url
        self.retries = retries
        self.delay = delay
        self.session = requests.Session()  # Mantiene cookies, headers, etc.
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ProyectoIA/1.0)"
        }

        # Configuración básica de logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch(self, url: Optional[str] = None, params: dict = None) -> Optional[requests.Response]:
        """
        Realiza una petición HTTP GET con manejo de errores y reintentos.

        :param url: URL a la que se quiere hacer la petición (usa base_url si es None)
        :param params: Parámetros para la petición GET
        :return: Objeto Response si fue exitosa, None si falló
        """
        target_url = url or self.base_url

        for attempt in range(1, self.retries + 1):
            try:
                self.logger.info(f"Intentando acceso a: {target_url} (Intento {attempt})")
                response = self.session.get(target_url, headers=self.headers, params=params, timeout=10)

                if response.status_code == 200:
                    return response
                else:
                    self.logger.warning(f"Respuesta no exitosa: {response.status_code}")
            except requests.RequestException as e:
                self.logger.error(f"Error de conexión: {e}")

            # Espera entre reintentos
            time.sleep(self.delay)

        self.logger.error(f"Fallo al obtener datos después de {self.retries} intentos.")
        return None

    @abstractmethod
    def crawl(self) -> Any:
        """
        Método abstracto que debe ser implementado por cada subclase.
        Define el flujo general de crawling para una fuente específica.
        """
        pass

    @abstractmethod
    def parse(self, response: requests.Response) -> Any:
        """
        Método abstracto para extraer y estructurar la información desde la respuesta.

        :param response: Objeto Response de la petición HTTP
        :return: Objeto con los datos extraídos (lista, dict, etc.)
        """
        pass
