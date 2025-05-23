import asyncio
import websockets
import json
import random
import os
import networkx as nx
from datetime import datetime
import math
from src.vehicle import initialize_vehicles, update_vehicle_positions
from src.traffic_lights import initialize_traffic_lights, update_traffic_lights


traffic_lights = {}  # node_id: {"state": "red"/"green", "timer": X}

# Centro de La Habana
lat_base, lon_base = 23.1136, -82.3666

# Grafo de calles y rutas
street_graph = nx.MultiDiGraph()
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
                    nodes[node_id] = (float(lat), float(lon)) 
                    street_graph.add_node(node_id, lat=float(lat), lon=float(lon))
        
        print(f"Nodos extraídos: {len(nodes)}")
        
        # Extraer vías (ways) y crear aristas
        edge_count = 0
        for element in osm_data.get('elements', []):
            if element.get('type') == 'way' and element.get('tags', {}).get('highway'):
                way_nodes = element.get('nodes', [])
                
                # Verificar si es de un solo sentido
                oneway = element.get('tags', {}).get('oneway', 'no')
                
                for i in range(len(way_nodes) - 1):
                    if way_nodes[i] in nodes and way_nodes[i+1] in nodes:
                        node1 = way_nodes[i]
                        node2 = way_nodes[i+1]
                        lat1, lon1 = nodes[node1]
                        lat2, lon2 = nodes[node2]
                        # Calcular distancia entre nodos
                        distance = haversine(lat1, lon1, lat2, lon2)
                        
                        # Añadir arista(s) según dirección
                        if oneway == 'yes':
                            # Solo añadir en la dirección especificada
                            street_graph.add_edge(node1, node2, weight=distance)
                            edge_count += 1
                        else:
                            # Añadir en ambas direcciones si es bidireccional
                            street_graph.add_edge(node1, node2, weight=distance)
                            street_graph.add_edge(node2, node1, weight=distance)
                            edge_count += 2
        
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



async def send_positions(websocket):
    """Envía las posiciones actualizadas de los vehículos al cliente"""
    while True:
        try:
            # Actualizar las posiciones
            update_vehicle_positions(street_graph, traffic_lights, vehicles, vehicle_speeds, all_nodes)
            update_traffic_lights(traffic_lights)
            
            # Empaquetar y enviar los datos
            payload = {
                "timestamp": datetime.now().isoformat(),
                "vehicles": [
                    {"id": vid, "lat": v["lat"], "lon": v["lon"]}
                    for vid, v in vehicles.items()
                ],
                "traffic_lights": [
                    {
                        "node_id": nid,
                        "lat": data["lat"],
                        "lon": data["lon"],
                        "state": data["state"],
                        "zone": data.get("zone", 0),
                        "direction": data.get("direction", "east")
                    } for nid, data in traffic_lights.items()
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
    # Cargar calles, inicializar vehículos y semaforos
    load_streets()
    initialize_vehicles(street_graph, all_nodes, vehicle_speeds, vehicles)
    initialize_traffic_lights(street_graph, traffic_lights)


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

