import asyncio
import websockets
import json
import random
import os
import networkx as nx
from datetime import datetime
import math

# Número de vehículos a simular (reducido para depuración)
NUM_VEHICLES = 10

# Centro de La Habana
lat_base, lon_base = 23.1136, -82.3666

# Grafo de calles y rutas
street_graph = nx.Graph()
all_nodes = []
vehicle_speeds = {}  # Velocidades diferentes para cada vehículo
vehicles = {}

def haversine(lat1, lon1, lat2, lon2):
    """Calcula la distancia entre dos puntos geográficos en km"""
    R = 6371.0  # Radio de la Tierra en km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def load_streets():
    """Carga los datos del mapa desde los archivos de caché y construye el grafo de calles"""
    global street_graph, all_nodes
    
    # Cargar datos de OSM desde el archivo de caché
    cache_file = os.path.join("cache", "479c34c9f9679cb8467293e0403a0250c7ef8556.json")
    
    try:
        print(f"Intentando abrir archivo de caché: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            osm_data = json.load(f)
        
        # Extraer nodos y crear grafo
        nodes = {}
        for element in osm_data.get('elements', []):
            if element.get('type') == 'node':
                node_id = element.get('id')
                lat = element.get('lat')
                lon = element.get('lon')
                if node_id and lat and lon:
                    nodes[node_id] = (float(lat), float(lon))  # Asegurarse de que sean float
                    street_graph.add_node(node_id, lat=float(lat), lon=float(lon))
        
        print(f"Nodos extraídos: {len(nodes)}")
        
        # Extraer vías (ways) y crear aristas
        edge_count = 0
        for element in osm_data.get('elements', []):
            if element.get('type') == 'way' and element.get('tags', {}).get('highway'):
                way_nodes = element.get('nodes', [])
                for i in range(len(way_nodes) - 1):
                    if way_nodes[i] in nodes and way_nodes[i+1] in nodes:
                        node1 = way_nodes[i]
                        node2 = way_nodes[i+1]
                        lat1, lon1 = nodes[node1]
                        lat2, lon2 = nodes[node2]
                        # Calcular distancia entre nodos
                        distance = haversine(lat1, lon1, lat2, lon2)
                        street_graph.add_edge(node1, node2, weight=distance)
                        edge_count += 1
        
        # Lista de todos los nodos del grafo
        all_nodes = list(street_graph.nodes())
        print(f"Grafo cargado con {len(all_nodes)} nodos y {edge_count} aristas")
        
    except Exception as e:
        print(f"Error cargando datos de calles: {e}")
        print("Creando grafo de desarrollo...")
        # Crear un grafo mínimo para desarrollo
        for i in range(20):
            lat = lat_base + random.uniform(-0.01, 0.01)
            lon = lon_base + random.uniform(-0.01, 0.01)
            street_graph.add_node(i, lat=lat, lon=lon)
            if i > 0:
                street_graph.add_edge(i-1, i)
        all_nodes = list(street_graph.nodes())
        print("Usando grafo de desarrollo con 20 nodos")

def initialize_vehicles():
    """Inicializa la posición de los vehículos y les asigna rutas aleatorias"""
    global vehicle_speeds, vehicles, all_nodes
    
    vehicles = {}
    
    if not all_nodes:
        print("No hay nodos en el grafo. Inicialización fallida.")
        return
    
    print(f"Inicializando {NUM_VEHICLES} vehículos...")
    
    for i in range(NUM_VEHICLES):
        vid = f"veh_{i}"
        
        # Seleccionar un nodo aleatorio para comenzar
        start_node = random.choice(all_nodes)
        node_data = street_graph.nodes[start_node]
        
        # Ubicar el vehículo en ese nodo
        vehicles[vid] = {
            "lat": float(node_data['lat']),  # Asegurarse de que sean float
            "lon": float(node_data['lon']),
            "current_node": start_node,
            "next_node": None,
            "progress": 0.0,  # Progreso en el tramo actual (0.0 a 1.0)
        }
        
        # Asignar velocidad aleatoria (km por actualización)
        # Incrementada para que se mueva más rápido y sea visible
        vehicle_speeds[vid] = 0.0005 + random.uniform(0, 0.0005)
        
        # Asignar una ruta aleatoría
        assign_random_route(vid)
    
    print(f"Vehículos inicializados: {len(vehicles)}")

def assign_random_route(vehicle_id):
    """Asigna una ruta aleatoria a un vehículo"""
    global all_nodes, vehicles
    
    if not all_nodes or len(all_nodes) < 2:
        print("No hay suficientes nodos para asignar rutas")
        return
    
    try:
        current = vehicles[vehicle_id]["current_node"]
        
        # Encontrar nodos conectados
        neighbors = list(street_graph.neighbors(current))
        
        if neighbors:
            # Elegir un vecino aleatorio como destino
            destination = random.choice(neighbors)
            
            # Asignar ese nodo como el siguiente en la ruta
            vehicles[vehicle_id]["next_node"] = destination
            vehicles[vehicle_id]["progress"] = 0.0
        else:
            # Si no hay vecinos, asignar un nodo aleatorio y teletransportarse
            new_node = random.choice(all_nodes)
            node_data = street_graph.nodes[new_node]
            vehicles[vehicle_id]["current_node"] = new_node
            vehicles[vehicle_id]["lat"] = float(node_data['lat'])
            vehicles[vehicle_id]["lon"] = float(node_data['lon'])
            assign_random_route(vehicle_id)
    
    except Exception as e:
        print(f"Error asignando ruta para {vehicle_id}: {e}")

def update_vehicle_positions():
    """Actualiza las posiciones de los vehículos en sus rutas"""
    for vid, v in vehicles.items():
        current_node = v["current_node"]
        next_node = v["next_node"]
        
        # Si no tiene un nodo siguiente, asignar uno
        if next_node is None:
            assign_random_route(vid)
            continue
            
        # Obtener las coordenadas de los nodos
        try:
            current_lat = float(street_graph.nodes[current_node]['lat'])
            current_lon = float(street_graph.nodes[current_node]['lon'])
            next_lat = float(street_graph.nodes[next_node]['lat'])
            next_lon = float(street_graph.nodes[next_node]['lon'])
            
            # Actualizar el progreso basado en la velocidad
            v["progress"] += vehicle_speeds[vid]
            
            if v["progress"] >= 1.0:
                # Llegó al nodo siguiente
                v["current_node"] = next_node
                v["next_node"] = None
                v["lat"] = next_lat
                v["lon"] = next_lon
                
                # Asignar una nueva ruta
                assign_random_route(vid)
            else:
                # Interpolación lineal entre los nodos
                v["lat"] = current_lat + (next_lat - current_lat) * v["progress"]
                v["lon"] = current_lon + (next_lon - current_lon) * v["progress"]
                
        except Exception as e:
            # En caso de error, asignar una nueva ruta
            print(f"Error en actualización de {vid}: {e}")
            assign_random_route(vid)

async def send_positions(websocket):
    """Envía las posiciones actualizadas de los vehículos al cliente"""
    while True:
        try:
            # Actualizar las posiciones
            update_vehicle_positions()
            
            # Empaquetar y enviar los datos
            payload = {
                "timestamp": datetime.now().isoformat(),
                "vehicles": [
                    {"id": vid, "lat": v["lat"], "lon": v["lon"]}
                    for vid, v in vehicles.items()
                ]
            }
            
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(0.2)  # Actualización más frecuente
        except websockets.exceptions.ConnectionClosed:
            print("Cliente desconectado")
            break
        except Exception as e:
            print(f"Error enviando datos: {e}")
            await asyncio.sleep(1)

async def handler(websocket):
    print("Cliente conectado")
    await send_positions(websocket)

async def main():
    # Cargar calles y inicializar vehículos
    load_streets()
    initialize_vehicles()
    
    print("Servidor WebSocket iniciando en puerto 8765...")
    async with websockets.serve(
        handler, 
        "localhost", 
        8765,
        ping_interval=30,
        ping_timeout=10
    ):
        # Mantener el servidor ejecutándose indefinidamente
        await asyncio.Future()

# Ejecuta el punto de entrada principal
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Servidor detenido por el usuario")

