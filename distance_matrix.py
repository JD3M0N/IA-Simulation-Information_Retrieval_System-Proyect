import os
import json

def get_distance_from_cache(source, target, cache_dict):
    """Obtiene la distancia entre dos nodos desde la caché"""
    key = f"{source}_{target}"
    return cache_dict.get(key)

def save_distance_to_cache(source, target, distance, cache_dict):
    """Guarda la distancia entre dos nodos en la caché"""
    key = f"{source}_{target}"
    cache_dict[key] = distance
    return cache_dict

def load_distance_cache(cache_file='cache/distance_cache.json'):
    """
    Carga la caché de distancias desde un archivo.
    Maneja errores cuando el archivo no existe o está mal formateado.
    """
    try:
        # Verificar si el archivo existe
        if not os.path.exists(cache_file):
            print(f"Archivo de caché no encontrado. Se creará uno nuevo en {cache_file}")
            return {}
        
        # Verificar si el archivo está vacío
        if os.path.getsize(cache_file) == 0:
            print(f"Archivo de caché vacío. Se iniciará con una caché nueva.")
            return {}
        
        # Intentar cargar el archivo
        with open(cache_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error al decodificar el archivo de caché. Se iniciará con una caché nueva.")
        # Hacer backup del archivo corrupto
        if os.path.exists(cache_file):
            backup_file = f"{cache_file}.bak"
            try:
                os.rename(cache_file, backup_file)
                print(f"Se ha creado una copia de seguridad del archivo corrupto en {backup_file}")
            except Exception as e:
                print(f"No se pudo crear copia de seguridad: {e}")
        return {}
    except Exception as e:
        print(f"Error al cargar la caché de distancias: {e}")
        return {}

def save_distance_cache(cache_dict, cache_file='cache/distance_cache.json'):
    """
    Guarda la caché de distancias en un archivo.
    Crea el directorio si no existe.
    """
    try:
        # Asegurar que el directorio cache existe
        cache_dir = os.path.dirname(cache_file)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Directorio de caché creado: {cache_dir}")
        
        # Guardar el archivo
        with open(cache_file, 'w') as f:
            json.dump(cache_dict, f)
        print(f"Caché guardada con éxito. Entradas: {len(cache_dict)}")
    except Exception as e:
        print(f"Error al guardar la caché de distancias: {e}")