from typing import Any, Dict, List, Optional
import requests
from bs4 import BeautifulSoup

from crawler.base_crawler import BaseCrawler


class TrafficCrawler(BaseCrawler):
    """
    Crawler específico para extraer información de la etiqueta
    'Ministerio de Transporte (MITRANS)' en Cubadebate.

    Para cada artículo listado en:
      http://www.cubadebate.cu/etiqueta/ministerio-de-transporte-mitrans/
    extrae:
      - título
      - URL del artículo
      - fecha de publicación (datetime ISO)
      - (opcional) un breve párrafo de resumen
    """

    def __init__(
        self,
        retries: int = 3,
        delay: float = 1.0,
    ):
        """
        :param retries: Número de reintentos en caso de fallo de red
        :param delay: Pausa (en segundos) entre cada reintento
        """
        # 1) Definimos la URL de la etiqueta “MITRANS” en Cubadebate
        base_url = "http://www.cubadebate.cu/etiqueta/ministerio-de-transporte-mitrans/"
        super().__init__(base_url=base_url, retries=retries, delay=delay)

    def crawl(self) -> List[Dict[str, Any]]:
        """
        1. Llama a self.fetch() para descargar el HTML de la página de MITRANS.
        2. Llama a self.parse() para extraer los datos.
        3. Devuelve una lista de diccionarios con la info de cada artículo.
        """
        response = self.fetch()
        if not response:
            # Si fetch() devolvió None, retornamos una lista vacía
            print("HTML no recuperado")
            return []

        return self.parse(response)

    def parse(self, response: requests.Response) -> List[Dict[str, Any]]:
        """
        1. Convierte la respuesta text en un objeto BeautifulSoup.
        2. Busca todos los <article> que representen noticias.
        3. Para cada artículo extrae título, link y fecha.
        4. (Opcional) Extrae el resumen (primer párrafo dentro de <div class="td-excerpt">).
        """
        html = response.text
        print("HTML recuperado: " + html)
        soup = BeautifulSoup(html, "html.parser")

        resultados: List[Dict[str, Any]] = []

        # 1) Encontrar todos los <article> en la página
        #    (en Cubadebate cada noticia suele ir dentro de un <article> con múltiples clases)
        articulos = soup.find_all("article")

        for art in articulos:
            # 2) Extraer el título y la URL del artículo
            h3 = art.find("h3", class_="entry-title")  # <h3 class="entry-title td-module-title">
            if not h3:
                continue  # Si no hay <h3 class="entry-title">, pasamos al siguiente <article>

            enlace = h3.find("a")
            if not enlace or not enlace.get("href"):
                continue

            titulo = enlace.get_text(strip=True)
            url = enlace["href"].strip()

            # 3) Extraer fecha de publicación
            #    La fecha va en <time class="entry-date updated td-module-date" datetime="...">
            time_tag = art.find("time", class_="entry-date")
            fecha_iso: Optional[str] = None
            if time_tag and time_tag.has_attr("datetime"):
                fecha_iso = time_tag["datetime"].strip()
            else:
                # Si no encontró el atributo “datetime”, podemos intentar leer el texto interno
                fecha_iso = time_tag.get_text(strip=True) if time_tag else None

            # 4) (Opcional) Extraer un snippet / resumen breve
            snippet = None
            excerpt_div = art.find("div", class_="td-excerpt")
            if excerpt_div:
                # Tomamos el primer <p> interno, si existe
                p = excerpt_div.find("p")
                if p:
                    snippet = p.get_text(strip=True)

            resultados.append({
                "title": titulo,
                "url": url,
                "date": fecha_iso,
                "snippet": snippet,
            })

        return resultados

