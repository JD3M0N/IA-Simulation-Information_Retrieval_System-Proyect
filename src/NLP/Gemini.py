import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("API_KEY_GEMINI")

class GeminiJSON:
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        genai.configure(api_key=api_key)

    def ask(self, prompt: str) -> str:
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        return response.text

    def parse_order(self, text: str) -> dict:
        """
        Envía al modelo la instrucción de formatear la orden como JSON.
        Limpia la respuesta para extraer solo el objeto JSON y lo parsea.
        """
        prompt = f"""
Analiza este mensaje de cliente y devuelve **solo** un JSON con las claves:
  - "direccion": la calle o dirección completa,
  - "pedido": artículo y cantidad,
  - "notas": demás información (puntos de referencia, restricciones de ruta, etc.)

Ejemplo de entrada y salida:

Entrada: "3 camisetas. cuarteles #120 entre ave de las misiones y habana. por el museo de la revolucion"
Salida:
{{
  "direccion": "Cuarteles #120 entre ave de las misiones y habana",
  "pedido": "3 camisetas",
  "notas": "El cliente añade como punto de referencia el museo de la revolución."
}}

Ahora procesa este texto:
"{text}"
"""
        raw = self.ask(prompt).strip()

        # 1) Si viene en bloque Markdown, quita los fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            # descarta la primera línea (```json) y la última (```)
            raw = "\n".join(lines[1:-1]).strip()

        # 2) Extrae sólo el contenido entre la primera { y la última }
        start = raw.find("{")
        end   = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No encontré un JSON en la respuesta:\n{raw!r}")
        clean_json = raw[start:end+1]

        # 3) Parséalo
        try:
            return json.loads(clean_json)
        except json.JSONDecodeError as e:
            # En caso de fallo, lanza un error más descriptivo
            raise ValueError(f"Error al parsear JSON:\n{clean_json!r}\nOriginal raw:\n{raw!r}") from e


if __name__ == "__main__":
    gemini = GeminiJSON()

    ejemplos = [
        "3 camisetas. cuarteles #120 entre ave de las misiones y habana. por el museo de la revolucion",
        "calle e 404 entre 14 y 16. No quiero que pase por playa. Vedado. 2 gafas de sol"
    ]

    for ejemplo in ejemplos:
        resultado = gemini.parse_order(ejemplo)
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
