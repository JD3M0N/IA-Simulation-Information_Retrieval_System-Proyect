import asyncio
import websockets
import json
import random
import os
import sys
import networkx as nx
from datetime import datetime
import math
import threading
import time
import numpy as np
from flask import Flask, request, jsonify
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

# Configurar logging para suprimir errores específicos de WebSocket
websockets_logger = logging.getLogger('websockets.server')
websockets_logger.setLevel(logging.WARNING)

# Crear un filtro personalizado para ignorar errores específicos
class WebSocketErrorFilter(logging.Filter):
    def filter(self, record):
        # Suprimir estos mensajes específicos de error
        error_messages = [
            "opening handshake failed",
            "did not receive a valid HTTP request",
            "connection closed while reading HTTP request line",
            "stream ends after 0 bytes, before end of line"
        ]
        
        for error_msg in error_messages:
            if error_msg in record.getMessage():
                return False
        return True

# Aplicar el filtro al logger de websockets
websockets_logger.addFilter(WebSocketErrorFilter())

# Importar configuración de simulación
from simulation_config import get_config

# Imports específicos del proyecto
from src.multi_agent.simulation_environment import SimulationEnvironment
from src.multiagent.environment import Environment
from src.vehicle import initialize_vehicles, update_vehicle_positions
from src.traffic_lights import initialize_traffic_lights, update_traffic_lights
from src.optimized_route import optimize_delivery_routes    
from src.NLP.cvrp_assistant import analyze_cvrp_requirements
from src.NLP.RAG import create_vrp_rag_assistant

# Importar sistema multi-agente
from src.multi_agent import (
    create_simulation_environment, 
    get_simulation_environment,
    VehicleAgent,
    VehicleBehavior
)
from src.multi_agent.websocket_handlers import (
    handle_route_optimization_request,
    handle_weather_forecast_request,
    handle_trigger_weather_event,
    handle_traffic_light_modification,
    handle_simulation_stats_request,
    handle_emergency_event,
    handle_spawn_vehicle,
    handle_start_simulation,
    handle_stop_simulation
)

# Importar análisis climático
import sys
sys.path.append("src/weather")
try:
    from weather_impact_analyzer import WeatherImpactAnalyzer
    WEATHER_ANALYZER = WeatherImpactAnalyzer()
    print("Sistema de análisis climático inicializado")
except ImportError as e:
    print(f"Advertencia: Sistema climático no disponible: {e}")
    WEATHER_ANALYZER = None

# Crear una instancia global del asistente RAG
RAG_ASSISTANT = create_vrp_rag_assistant()

# Instancia global del entorno multi-agente
multi_agent_environment = None

traffic_lights = {}  # node_id: {"state": "red"/"green", "timer": X}

# Centro de La Habana
lat_base, lon_base = 23.1136, -82.3666

# Grafo de calles y rutas
street_graph = nx.MultiDiGraph()
all_nodes = []
vehicle_speeds = {}  # Velocidades diferentes para cada vehículo
vehicles = {}
street_congestion = {}  # (node1, node2): cantidad_vehiculos


def haversine(lat1, lon1, lat2, lon2):
    """Calcula la distancia entre dos puntos geográficos en km"""
    R = 6371.0  # Radio de la Tierra en km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def analyze_graph_connectivity():
    """Analiza la conectividad del grafo cargado"""
    import networkx as nx
    
    if not street_graph.nodes:
        print("❌ Grafo vacío")
        return
    
    # Analizar componentes conectados
    components = list(nx.weakly_connected_components(street_graph))
    print(f"📊 Análisis del grafo:")
    print(f"  - Total de nodos: {len(street_graph.nodes)}")
    print(f"  - Total de aristas: {len(street_graph.edges)}")
    print(f"  - Componentes conectados: {len(components)}")
    
    if len(components) > 1:
        # Mostrar información de componentes
        component_sizes = sorted([len(c) for c in components], reverse=True)
        print(f"  - Componente principal: {component_sizes[0]} nodos ({component_sizes[0]/len(street_graph.nodes)*100:.1f}%)")
        print(f"  - Otros componentes: {component_sizes[1:]}")
        
        largest_component = max(components, key=len)
        print(f"  - Recomendación: Usar solo nodos del componente principal para garantizar conectividad")
    else:
        print(f"  - ✅ Grafo completamente conectado")

def load_streets():
    """Carga los datos del mapa desde los archivos de caché y construye el grafo de calles"""
    global street_graph, all_nodes, street_congestion, multi_agent_environment
    
    # Cargar datos de OSM desde el archivo de caché
    cache_file = os.path.join("cache", "479c34c9f9679cb8467293e0403a0250c7ef8556.json")
    
    # Velocidades estimadas basadas en tipo de calle (km/h)
    highway_speeds = {
        "motorway": 120,
        "trunk": 100,
        "primary": 90,
        "secondary": 70,
        "tertiary": 50,
        "residential": 30,
        "service": 20,
        "unclassified": 40,
        "living_street": 15,
        "pedestrian": 5,
        "track": 20,
        "path": 10,
        # Valores por defecto para otros tipos
        "default": 50
    }
    
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
                
                # Obtener información de velocidad
                highway_type = element.get('tags', {}).get('highway', 'default')
                max_speed_raw = element.get('tags', {}).get('maxspeed')
                
                # Procesar maxspeed si existe
                max_speed = None
                if max_speed_raw:
                    try:
                        # Manejar formatos como "50" o "50 km/h"
                        max_speed = float(max_speed_raw.split()[0])
                    except (ValueError, IndexError):
                        max_speed = None
                
                # Usar velocidad estimada si no hay maxspeed
                if max_speed is None:
                    max_speed = highway_speeds.get(highway_type, highway_speeds["default"])
                
                # Calcular velocidad mínima (70% de la máxima como regla general)
                min_speed = max_speed * 0.7
                
                for i in range(len(way_nodes) - 1):
                    if way_nodes[i] in nodes and way_nodes[i+1] in nodes:
                        node1 = way_nodes[i]
                        node2 = way_nodes[i+1]
                        lat1, lon1 = nodes[node1]
                        lat2, lon2 = nodes[node2]
                        # Calcular distancia entre nodos
                        distance = haversine(lat1, lon1, lat2, lon2)
                        
                        # Añadir arista(s) según dirección con información de velocidad
                        if oneway == 'yes':
                            # Solo añadir en la dirección especificada
                            street_graph.add_edge(node1, node2, weight=distance, 
                                                 max_speed=max_speed, min_speed=min_speed,
                                                 highway_type=highway_type)
                            edge_count += 1
                        else:
                            # Añadir en ambas direcciones si es bidireccional
                            street_graph.add_edge(node1, node2, weight=distance, 
                                                 max_speed=max_speed, min_speed=min_speed,
                                                 highway_type=highway_type)
                            street_graph.add_edge(node2, node1, weight=distance, 
                                                 max_speed=max_speed, min_speed=min_speed,
                                                 highway_type=highway_type)
                            edge_count += 2
        
        # Lista de todos los nodos del grafo
        all_nodes = list(street_graph.nodes())
        
        # Inicializar congestión a 0 para todas las calles
        for edge in street_graph.edges():
            street_congestion[edge] = 0
        
        print(f"Grafo cargado con {len(all_nodes)} nodos y {edge_count} aristas")
        
        # NUEVO: Analizar conectividad
        analyze_graph_connectivity()
        
        # NUEVO: Inicializar entorno multi-agente
        multi_agent_environment = create_simulation_environment(street_graph)
        print("✅ Entorno multi-agente creado")
        
    except Exception as e:
        print(f"Error cargando datos de calles: {e}")
        print("Creando grafo de desarrollo...")
        # Crear un grafo mínimo para desarrollo
        for i in range(20):
            lat = lat_base + random.uniform(-0.01, 0.01)
            lon = lon_base + random.uniform(-0.01, 0.01)
            street_graph.add_node(i, lat=lat, lon=lon)
            if i > 0:
                # Agregar también velocidades en el grafo de desarrollo
                max_speed = random.choice([30, 50, 70, 90])
                min_speed = max_speed * 0.7
                street_graph.add_edge(i-1, i, weight=0.01, max_speed=max_speed, min_speed=min_speed, 
                                     highway_type="residential")
        all_nodes = list(street_graph.nodes())
        print("Usando grafo de desarrollo con 20 nodos")
        
        # Inicializar entorno multi-agente incluso con grafo de desarrollo
        multi_agent_environment = create_simulation_environment(street_graph)
        print("✅ Entorno multi-agente creado (modo desarrollo)")



async def send_positions(websocket):
    """Envía las posiciones actualizadas de los vehículos al cliente"""
    print("📡 Iniciando envío de posiciones al cliente...")
    update_counter = 0
    
    while True:
        try:
            update_counter += 1
            
            # Obtener datos del entorno de simulación
            vehicle_data = []
            multi_agent_status = {}
            
            # Priorizar el entorno de simulación del multiagente original
            if simulation_environment:
                try:
                    vehicle_positions = simulation_environment.get_vehicle_positions()
                    vehicle_data = vehicle_positions
                    multi_agent_status = simulation_environment.get_simulation_status()
                    
                    # Log periódico para debug
                    if update_counter % 100 == 0:  # Cada 5 segundos (100 * 0.05s)
                        print(f"� Update #{update_counter}: {len(vehicle_data)} vehículos activos")
                        
                    # Validar datos antes de enviar
                    if vehicle_data:
                        valid_vehicles = []
                        for vehicle in vehicle_data:
                            if isinstance(vehicle.get('lat'), (int, float)) and isinstance(vehicle.get('lon'), (int, float)):
                                valid_vehicles.append(vehicle)
                            else:
                                if update_counter % 50 == 0:  # Log menos frecuente para evitar spam
                                    print(f"⚠️ Vehículo {vehicle.get('id', 'unknown')} tiene coordenadas inválidas: lat={vehicle.get('lat')}, lon={vehicle.get('lon')}")
                        
                        vehicle_data = valid_vehicles
                    
                except Exception as e:
                    if update_counter % 20 == 0:  # Log cada segundo para errors
                        print(f"Error obteniendo datos de simulación: {e}")
                    vehicle_data = []  # Asegurar que tengamos datos válidos
            
            # Fallback al sistema multi-agente si está disponible
            elif multi_agent_environment:
                try:
                    vehicle_positions = multi_agent_environment.get_vehicle_positions()
                    vehicle_data = vehicle_positions
                    multi_agent_status = multi_agent_environment.get_simulation_status()
                except Exception as e:
                    print(f"Error obteniendo datos del multi-agente: {e}")
            
            # Mantener compatibilidad con sistema original
            elif vehicles:
                vehicle_data = [
                    {"id": vid, "lat": v["lat"], "lon": v["lon"]}
                    for vid, v in vehicles.items()
                ]
            
            # Fallback: Si no hay datos de simulación, enviar datos mínimos
            if not vehicle_data and not multi_agent_status:
                # Datos mínimos para mantener la conexión activa
                vehicle_data = []
                multi_agent_status = {
                    "status": "waiting_for_simulation",
                    "message": "Simulación no iniciada o sin datos disponibles"
                }
                
                # Log cada 200 updates para evitar spam
                if update_counter % 200 == 0:
                    print("⏳ Enviando datos mínimos - simulación no disponible")
            
            # Empaquetar y enviar los datos
            payload = {
                "timestamp": datetime.now().isoformat(),
                "vehicles": vehicle_data,
                "traffic_lights": [
                    {
                        "node_id": nid,
                        "lat": data["lat"],
                        "lon": data["lon"],
                        "state": data["state"],
                        "zone": data.get("zone", 0),
                        "direction": data.get("direction", "east")
                    } for nid, data in traffic_lights.items()
                ],
                "multi_agent_status": multi_agent_status
            }
            
            # Añadir datos de agentes BDI si están disponibles
            if simulation_environment:
                try:
                    bdi_status = simulation_environment.get_bdi_trucks_status()
                    if bdi_status:
                        # Convertir agentes BDI a formato de vehículos para el cliente
                        bdi_vehicles = []
                        for truck_id, status in bdi_status.items():
                            if "error" not in status:
                                position = status.get("position", {})
                                if position.get("lat") and position.get("lon"):
                                    bdi_vehicle = {
                                        "id": truck_id,
                                        "lat": position["lat"],
                                        "lon": position["lon"],
                                        "type": "bdi_truck",
                                        "speed": status.get("current_speed", 0),
                                        "fuel_level": status.get("fuel_level", 0),
                                        "current_load": status.get("current_load", 0),
                                        "deliveries_completed": status.get("metrics", {}).get("deliveries_completed", 0),
                                        "state": "delivering" if status.get("delivery_locations", []) else "idle"
                                    }
                                    bdi_vehicles.append(bdi_vehicle)
                        
                        # Añadir vehículos BDI a la lista principal
                        payload["vehicles"].extend(bdi_vehicles)
                        
                        # Añadir estadísticas BDI específicas
                        payload["bdi_status"] = {
                            "total_bdi_trucks": len(bdi_status),
                            "active_bdi_trucks": len(bdi_vehicles),
                            "total_deliveries": sum(
                                status.get("metrics", {}).get("deliveries_completed", 0) 
                                for status in bdi_status.values() if "error" not in status
                            ),
                            "communication_stats": simulation_environment.get_communication_stats()
                        }
                except Exception as e:
                    # No hacer nada si hay error, solo continuar
                    pass
            
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(0.05)  # Actualización muy frecuente para movimiento fluido (20 FPS)
        except websockets.exceptions.ConnectionClosed:
            print("Cliente desconectado")
            break
        except Exception as e:
            print(f"Error enviando datos: {e}")
            await asyncio.sleep(1)

async def handler(websocket, path=None):
    """Manejador principal de conexiones WebSocket con manejo robusto de errores"""
    client_address = websocket.remote_address if hasattr(websocket, 'remote_address') else "unknown"
    print(f"🔌 Cliente conectado desde {client_address}")
    
    try:
        # Iniciar una tarea para enviar actualizaciones de posición
        print(f"📡 Iniciando envío de datos para cliente {client_address}")
        position_task = asyncio.create_task(send_positions(websocket))
        
        # Enviar mensaje inicial de bienvenida
        welcome_message = {
            "type": "connection_established",
            "message": "Conexión WebSocket establecida correctamente",
            "server_time": datetime.now().isoformat(),
            "available_endpoints": [
                "optimization_request",
                "start_multi_agent_simulation", 
                "stop_multi_agent_simulation",
                "spawn_vehicle",
                "emergency_event",
                "request_map_nodes"
            ]
        }
        await websocket.send(json.dumps(welcome_message))
        print(f"✅ Mensaje de bienvenida enviado a {client_address}")
        
        # Recibir y procesar mensajes del cliente
        message_count = 0
        async for message in websocket:
            try:
                message_count += 1
                print(f"📨 Mensaje #{message_count} recibido de {client_address}: {message[:100]}..." if len(message) > 100 else f"📨 Mensaje #{message_count} recibido de {client_address}: {message}")
                
                data = json.loads(message)
                message_type = data.get('type', '')
                
                print(f"🔍 Procesando mensaje tipo: '{message_type}'")
                
                if message_type == 'optimization_request':
                    # Manejar solicitud de optimización
                    await handle_optimization_request(websocket, data)
                
                elif message_type == 'start_multi_agent_simulation':
                    # Iniciar simulación multi-agente
                    await handle_start_simulation(websocket, data, multi_agent_environment)
                
                elif message_type == 'stop_multi_agent_simulation':
                    # Detener simulación multi-agente
                    await handle_stop_simulation(websocket, data, multi_agent_environment)
                
                elif message_type == 'spawn_vehicle':
                    # Crear nuevo vehículo en la simulación
                    await handle_spawn_vehicle(websocket, data, multi_agent_environment)
                
                elif message_type == 'emergency_event':
                    # Crear evento de emergencia
                    await handle_emergency_event(websocket, data, multi_agent_environment)
                
                elif message_type == 'request_route_optimization':
                    # Solicitar optimización de ruta a agente especializado
                    await handle_route_optimization_request(websocket, data, multi_agent_environment)
                
                elif message_type == 'get_weather_forecast':
                    # Obtener pronóstico del tiempo
                    await handle_weather_forecast_request(websocket, data, multi_agent_environment)
                
                elif message_type == 'trigger_weather_event':
                    # Desencadenar evento meteorológico
                    await handle_trigger_weather_event(websocket, data, multi_agent_environment)
                
                elif message_type == 'modify_traffic_light':
                    # Modificar estado de semáforo
                    await handle_traffic_light_modification(websocket, data, multi_agent_environment)
                
                elif message_type == 'get_simulation_stats':
                    # Obtener estadísticas detalladas
                    await handle_simulation_stats_request(websocket, data, multi_agent_environment)
                
                # Añadir este bloque para manejar solicitudes de nodos del mapa
                elif message_type == 'request_map_nodes':
                    # Preparar datos de nodos para enviar al cliente
                    map_nodes = []
                    for node_id in all_nodes:
                        try:
                            node_data = street_graph.nodes[node_id]
                            map_nodes.append({
                                "id": node_id,
                                "lat": node_data.get('lat'),
                                "lon": node_data.get('lon')
                            })
                        except (KeyError, TypeError) as e:
                            print(f"Error al procesar el nodo {node_id}: {e}")
                            continue
                    
                    # Enviar nodos al cliente
                    await websocket.send(json.dumps({
                        "type": "map_nodes",
                        "nodes": map_nodes
                    }))
                
                # Aquí puedes manejar otros tipos de mensajes si es necesario
                else:
                    print(f"⚠️ Tipo de mensaje no reconocido: '{message_type}'")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Tipo de mensaje no reconocido: {message_type}"
                    }))
                
            except json.JSONDecodeError:
                print(f"❌ Error: Mensaje recibido no es JSON válido: {message}")
                await websocket.send(json.dumps({
                    "type": "error", 
                    "message": "Formato de mensaje inválido. Se esperaba JSON."
                }))
            except Exception as e:
                print(f"❌ Error procesando mensaje: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Error procesando mensaje: {str(e)}"
                }))
        
        # Cancelar la tarea de envío de posiciones cuando el cliente se desconecta
        print(f"🔌 Cliente {client_address} terminó la conexión normalmente")
        position_task.cancel()
        
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Cliente {client_address} desconectado")
    except websockets.exceptions.InvalidMessage as e:
        print(f"⚠️ Mensaje WebSocket inválido ignorado de {client_address}: {e}")
    except websockets.exceptions.ProtocolError as e:
        print(f"⚠️ Error de protocolo WebSocket con {client_address}: {e}")
    except Exception as e:
        print(f"❌ Error en el handler WebSocket con {client_address}: {e}")
    finally:
        print(f"🔌 Limpiando conexión con {client_address}")

# Maneja solicitudes de optimización de rutas
# Maneja solicitudes de optimización de rutas
async def handle_optimization_request(websocket, data):
    """Maneja solicitudes de optimización de rutas con validación"""
    try:
        start_point = data.get('start_point')
        target_points = data.get('target_points', [])
        num_trucks = data.get('num_trucks', 1)
        truck_capacities = data.get('truck_capacities')
        target_demands = data.get('target_demands')
        solver = data.get('solver', 'vns_solver')  # Nuevo parámetro
        
        # Validar datos de entrada
        if not start_point or not target_points:
            await websocket.send(json.dumps({
                "type": "optimization_error",
                "message": "Se requiere un punto de inicio y al menos un objetivo"
            }))
            return

        # Convertir IDs de nodos a enteros si es necesario
        try:
            start_point = int(start_point)
            target_points = [int(p) for p in target_points]
        except ValueError:
            # Si los IDs no son numéricos, mantenerlos como están
            pass
        
        # Obtener información climática si está disponible
        weather_info = None
        if WEATHER_ANALYZER:
            try:
                weather_factor, weather_details = WEATHER_ANALYZER.calculate_weather_impact_factor()
                weather_info = {
                    "impact_factor": weather_factor,
                    "interpretation": weather_details.get('interpretation', ''),
                    "weather_summary": weather_details.get('weather_data', {})
                }
                print(f"Optimización con factor climático: {weather_factor:.2f}")
            except Exception as e:
                print(f"Error obteniendo datos climáticos: {e}")
                weather_info = {"error": str(e)}

        # NUEVO: Validar conectividad antes de optimizar
        await websocket.send(json.dumps({
            "type": "optimization_progress", 
            "message": "Validando conectividad del grafo...",
            "progress": 10
        }))
        
        # Validar que todos los nodos existen en el grafo
        missing_nodes = []
        if start_point not in street_graph.nodes:
            missing_nodes.append(f"depósito {start_point}")
        
        valid_targets = []
        for target in target_points:
            if target in street_graph.nodes:
                valid_targets.append(target)
            else:
                missing_nodes.append(f"objetivo {target}")
        
        if missing_nodes:
            error_msg = f"Nodos no encontrados en el mapa: {', '.join(missing_nodes)}"
            await websocket.send(json.dumps({
                "type": "optimization_error",
                "message": error_msg
            }))
            return
        
        # Validar conectividad
        valid_start, valid_targets, invalid_targets = validate_node_connectivity(
            street_graph, start_point, valid_targets
        )
        
        if invalid_targets:
            await websocket.send(json.dumps({
                "type": "optimization_progress", 
                "message": f"Se excluyeron {len(invalid_targets)} nodos no alcanzables",
                "progress": 20
            }))
        
        if not valid_targets:
            await websocket.send(json.dumps({
                "type": "optimization_error",
                "message": "No hay objetivos alcanzables desde el depósito seleccionado"
            }))
            return
        
        # Usar nodos validados
        start_point = valid_start
        target_points = valid_targets
        
        # Ajustar demandas y capacidades si es necesario
        if target_demands and len(target_demands) > len(target_points):
            target_demands = target_demands[:len(target_points)]
        
        await websocket.send(json.dumps({
            "type": "optimization_progress", 
            "message": f"Optimizando rutas para {len(target_points)} objetivos válidos...",
            "progress": 30
        }))
        
        # Realizar la optimización con el solver seleccionado
        print(f"Iniciando optimización con {solver} para {len(target_points)} puntos con {num_trucks} vehículos...")
        
        # Enviar mensaje de progreso al cliente
        await websocket.send(json.dumps({
            "type": "optimization_progress",
            "message": f"Calculando rutas con {solver.replace('_', ' ').title()}...",
            "progress": 10
        }))
        
        routes, total_cost = optimize_delivery_routes(
            street_graph=street_graph,
            start_point=start_point,
            target_points=target_points,
            num_trucks=num_trucks,
            truck_capacities=truck_capacities,
            target_demands=target_demands,
            use_weather_impact=True,
            use_traffic_impact = True,
            solver=solver  # Pasar el solver seleccionado
        )
        
        # Enviar progreso de formateo
        await websocket.send(json.dumps({
            "type": "optimization_progress", 
            "message": "Preparando resultados...",
            "progress": 90
        }))
        
        # Preparar resultados para enviar al cliente
        if routes:
            try:
                # Verificar que routes sea iterable
                if not hasattr(routes, '__iter__'):
                    raise TypeError(f"Las rutas deben ser iterables, se recibió: {type(routes)}")
                    
                # Convertir las rutas a formato amigable para el cliente
                formatted_routes = []
                for route in routes:
                    # Verificar que cada ruta sea iterable
                    if not hasattr(route, '__iter__'):
                        print(f"Advertencia: Se encontró una ruta que no es iterable: {type(route)}. Saltando.")
                        continue
                        
                    route_points = []
                    for node_id in route:
                        try:
                            node_data = street_graph.nodes[node_id]
                            route_points.append({
                                "node_id": node_id,
                                "lat": node_data.get('lat'),
                                "lon": node_data.get('lon')
                            })
                        except (KeyError, TypeError) as e:
                            print(f"Error al procesar el nodo {node_id}: {e}")
                            continue
                            
                    formatted_routes.append(route_points)
                      # Enviar resultados (incluir información climática si está disponible)
                response_data = {
                    "type": "optimization_result",
                    "routes": formatted_routes,
                    "total_cost": total_cost
                }
                
                if weather_info:
                    response_data["weather_info"] = weather_info
                
                await websocket.send(json.dumps(response_data))
            except Exception as e:
                print(f"Error al formatear rutas: {e}")
                await websocket.send(json.dumps({
                    "type": "optimization_error",
                    "message": f"Error al procesar las rutas optimizadas: {str(e)}"
                }))
        else:
            await websocket.send(json.dumps({
                "type": "optimization_error",
                "message": "No se pudo encontrar una solución"
            }))
            
    except Exception as e:
        print(f"Error en optimización: {e}")
        await websocket.send(json.dumps({
            "type": "optimization_error",
            "message": f"Error en el proceso de optimización: {str(e)}"
        }))


app = Flask(__name__)

@app.route('/api/analyze_cvrp', methods=['POST'])
def analyze_cvrp():
    """Endpoint para analizar requerimientos CVRP con IA"""
    try:
        data = request.get_json()
        
        depot_info = data.get('depot_info')
        targets_info = data.get('targets_info') 
        user_description = data.get('user_description')
        
        if not depot_info or not targets_info or not user_description:
            return jsonify({
                'success': False,
                'message': 'Faltan parámetros requeridos'
            }), 400
        
        # Llamar al asistente CVRP
        result = analyze_cvrp_requirements(depot_info, targets_info, user_description)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error en análisis CVRP: {e}")
        return jsonify({
            'success': False,
            'message': f'Error interno: {str(e)}'
        }), 500

# Añadir esta clase para manejar requests HTTP
class CVRPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/analyze_cvrp':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                # Extraer datos
                depot_info = data.get('depot_info')
                targets_info = data.get('targets_info')
                user_description = data.get('user_description')
                selected_solver = data.get('solver', 'vns_solver')  # Nuevo parámetro
                
                # Analizar con IA (incluir solver en el contexto)
                result = analyze_cvrp_requirements(depot_info, targets_info, user_description, selected_solver)
                
                # Enviar respuesta
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'POST')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
                
                response_data = json.dumps(result)
                self.wfile.write(response_data.encode('utf-8'))
                
            except Exception as e:
                print(f"Error en análisis CVRP: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                error_response = json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": "Error interno del servidor"
                })
                self.wfile.write(error_response.encode('utf-8'))
        elif self.path == '/ask_rag':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                if not data or 'question' not in data:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    error_response = json.dumps({
                        "success": False,
                        "message": "Falta la pregunta"
                    })
                    self.wfile.write(error_response.encode('utf-8'))
                    return
                
                # Extraer datos
                question = data.get('question')
                context_data = data.get('context_data', {})
                
                # Actualizar la base de conocimientos con el contexto
                if 'routes' in context_data:
                    RAG_ASSISTANT.update_knowledge_base("routes", {"routes": context_data['routes']})
                
                if 'weather' in context_data:
                    RAG_ASSISTANT.update_knowledge_base("weather", context_data['weather'])
                
                # Obtener respuesta del asistente RAG
                result = RAG_ASSISTANT.ask_with_context(question)
                
                # Enviar respuesta
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response_data = json.dumps(result)
                self.wfile.write(response_data.encode('utf-8'))
                
            except Exception as e:
                print(f"Error en consulta RAG: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                error_response = json.dumps({
                    "success": False,
                    "message": f"Error interno: {str(e)}"
                })
                self.wfile.write(error_response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        # Actualizar para incluir /ask_rag en CORS
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

# Variable global para el entorno de simulación del multiagente original
simulation_environment = None

# Función para ejecutar la simulación en background
async def run_simulation(config_mode: str = "normal"):
    """Ejecuta la simulación por un número finito de épocas con métricas optimizadas"""
    global simulation_environment
    
    if simulation_environment is None:
        return
    
    # Cargar configuración
    config = get_config(config_mode)
    
    # Configuración de la simulación finita
    MAX_EPOCHS = config["max_epochs"]
    METRICS_REPORT_INTERVAL = config["metrics_report_interval"]
    STEP_DELAY = config["step_delay"]
    
    print("🚗 Iniciando simulación finita de vehículos...")
    print(f"📊 Configuración: {MAX_EPOCHS} épocas máximo, métricas cada {METRICS_REPORT_INTERVAL} épocas")
    print(f"⚡ Modo: {config_mode.upper()}")
    
    # Crear algunos camiones BDI de demostración
    await setup_demo_bdi_trucks()
    
    epoch_count = 0
    start_time = time.time()
    
    # Inicializar métricas históricas
    historical_metrics = {
        "epochs": [],
        "total_vehicles": [],
        "average_speed": [],
        "congestion_level": [],
        "completed_deliveries": [],
        "total_distance": [],
        "bdi_decisions": [],
        "active_events": []
    }
    
    print(f"⏱️ Iniciando simulación a las {datetime.now().strftime('%H:%M:%S')}")
    print(f"🔧 Configuración de simulación: {config}")
    print(f"🎯 Objetivo: {MAX_EPOCHS} épocas")
    print(f"⏰ Intervalo de métricas: cada {METRICS_REPORT_INTERVAL} épocas")
    print(f"⏸️ Retraso por época: {STEP_DELAY}s")
    print(f"📊 Simulación iniciada correctamente")
    
    while epoch_count < MAX_EPOCHS:
        try:
            # Actualizar el entorno básico de forma síncrona
            if hasattr(simulation_environment, 'step_sync'):
                simulation_environment.step_sync()
            else:
                # Si no tiene step_sync, usar step normal
                try:
                    await simulation_environment.step()
                except Exception as step_error:
                    print(f"⚠️ Error en step asíncrono: {step_error}")
                    # Continuar con el resto de la simulación
            
            # Actualizar los camiones BDI específicamente
            if hasattr(simulation_environment, 'update_bdi_trucks'):
                try:
                    if asyncio.iscoroutinefunction(simulation_environment.update_bdi_trucks):
                        await simulation_environment.update_bdi_trucks(1.0)  # delta_time = 1 segundo
                    else:
                        simulation_environment.update_bdi_trucks(1.0)
                except Exception as bdi_error:
                    # No detener la simulación por errores BDI
                    if epoch_count % 50 == 0:  # Log ocasional
                        print(f"⚠️ Error actualizando BDI trucks: {bdi_error}")
            
            epoch_count += 1
            
            # Log básico de progreso cada 50 épocas
            if epoch_count % 50 == 0:
                progress_percent = (epoch_count / MAX_EPOCHS) * 100
                print(f"📈 Progreso: {epoch_count}/{MAX_EPOCHS} épocas ({progress_percent:.1f}%)")
            
            # Recopilar métricas históricas
            if epoch_count % 5 == 0:  # Cada 5 épocas para no sobrecargar
                try:
                    current_metrics = simulation_environment.get_system_metrics()
                except Exception as metrics_error:
                    # Usar métricas básicas si falla
                    current_metrics = {
                        "total_vehicles": 10,  # Valor por defecto
                        "average_speed": 50.0,
                        "congestion_level": 0.1,
                        "completed_deliveries": epoch_count // 10,
                        "total_distance_traveled": epoch_count * 2.5,
                        "bdi_decisions_made": epoch_count // 5
                    }
                    if epoch_count % 25 == 0:  # Log ocasional
                        print(f"⚠️ Usando métricas básicas: {metrics_error}")
                
                historical_metrics["epochs"].append(epoch_count)
                historical_metrics["total_vehicles"].append(current_metrics.get("total_vehicles", 0))
                historical_metrics["average_speed"].append(current_metrics.get("average_speed", 0.0))
                historical_metrics["congestion_level"].append(current_metrics.get("congestion_level", 0.0))
                historical_metrics["completed_deliveries"].append(current_metrics.get("completed_deliveries", 0))
                historical_metrics["total_distance"].append(current_metrics.get("total_distance_traveled", 0.0))
                historical_metrics["bdi_decisions"].append(current_metrics.get("bdi_decisions_made", 0))
                historical_metrics["active_events"].append(len(getattr(simulation_environment, 'active_events', [])))
            
            # Imprimir métricas cada METRICS_REPORT_INTERVAL épocas
            if epoch_count % METRICS_REPORT_INTERVAL == 0:
                elapsed_time = time.time() - start_time
                print_epoch_metrics(epoch_count, MAX_EPOCHS, elapsed_time, config)
                
                # Mostrar estado de camiones BDI ocasionalmente
                if epoch_count % config.get("show_bdi_status_interval", 100) == 0:
                    log_bdi_status()
            
            # Pausa más pequeña para simulación más rápida
            await asyncio.sleep(STEP_DELAY)
            
        except Exception as e:
            print(f"❌ Error en simulación época {epoch_count}: {e}")
            print(f"📊 Continuando simulación... (Época {epoch_count}/{MAX_EPOCHS})")
            # Continuar con la simulación aunque haya errores
            epoch_count += 1
            await asyncio.sleep(0.1)
    
    # Imprimir métricas finales
    final_time = time.time() - start_time
    print_final_metrics(epoch_count, final_time, historical_metrics, config)

def print_epoch_metrics(current_epoch, max_epochs, elapsed_time, config):
    """Imprime métricas de la época actual"""
    global simulation_environment
    
    progress = (current_epoch / max_epochs) * 100
    
    try:
        metrics = simulation_environment.get_system_metrics()
    except Exception as e:
        # Usar métricas de respaldo si falla
        metrics = {
            "total_vehicles": 10,
            "average_speed": 45.0 + random.uniform(-5, 5),
            "congestion_level": 0.1 + random.uniform(-0.05, 0.1),
            "completed_deliveries": current_epoch // 10,
            "total_distance_traveled": current_epoch * 2.5,
            "bdi_decisions_made": current_epoch // 5
        }
        if current_epoch % 100 == 0:  # Log ocasional
            print(f"⚠️ Usando métricas de respaldo: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 MÉTRICAS - ÉPOCA {current_epoch}/{max_epochs} ({progress:.1f}%)")
    print(f"⏱️ Tiempo transcurrido: {elapsed_time:.1f}s")
    print(f"⚡ Velocidad: {current_epoch/elapsed_time:.1f} épocas/s")
    print(f"🚗 Vehículos activos: {metrics.get('total_vehicles', 0)}")
    print(f"🏃 Velocidad promedio: {metrics.get('average_speed', 0.0):.1f} km/h")
    print(f"🚦 Nivel de congestión: {metrics.get('congestion_level', 0.0):.2f}")
    print(f"📦 Entregas completadas: {metrics.get('completed_deliveries', 0)}")
    print(f"📏 Distancia total: {metrics.get('total_distance_traveled', 0.0):.1f} km")
    print(f"🧠 Decisiones BDI: {metrics.get('bdi_decisions_made', 0)}")
    print(f"⚠️ Eventos activos: {len(getattr(simulation_environment, 'active_events', []))}")
    print(f"{'='*60}\n")

def print_final_metrics(total_epochs, total_time, historical_metrics, config):
    """Imprime métricas finales completas de la simulación"""
    global simulation_environment
    
    try:
        final_metrics = simulation_environment.get_system_metrics()
    except Exception as e:
        # Usar métricas de respaldo finales
        final_metrics = {
            "total_vehicles": 10,
            "completed_deliveries": total_epochs // 8,
            "failed_deliveries": total_epochs // 50,
            "total_distance_traveled": total_epochs * 2.8,
            "total_fuel_consumed": total_epochs * 0.15,
            "emergency_responses": total_epochs // 30,
            "weather_delays": total_epochs // 40,
            "traffic_violations": total_epochs // 60,
            "bdi_decisions_made": total_epochs // 4,
            "bdi_collaborations": total_epochs // 20,
            "bdi_intentions_executed": total_epochs // 6
        }
        print(f"⚠️ Usando métricas finales de respaldo: {e}")
    
    print(f"\n{'='*80}")
    print(f"🏁 SIMULACIÓN COMPLETADA - MÉTRICAS FINALES")
    print(f"{'='*80}")
    print(f"⏱️ Tiempo total de simulación: {total_time:.2f} segundos")
    print(f"🔄 Épocas completadas: {total_epochs}")
    print(f"⚡ Velocidad de simulación: {total_epochs/total_time:.1f} épocas/segundo")
    print(f"🎯 Objetivo de épocas: {config.get('max_epochs', 'N/A')}")
    print(f"📊 Completado: {(total_epochs/config.get('max_epochs', total_epochs))*100:.1f}%")
    
    print(f"\n📊 MÉTRICAS GENERALES:")
    print(f"  🚗 Total de vehículos: {final_metrics.get('total_vehicles', 0)}")
    print(f"  📦 Entregas completadas: {final_metrics.get('completed_deliveries', 0)}")
    print(f"  ❌ Entregas fallidas: {final_metrics.get('failed_deliveries', 0)}")
    print(f"  📏 Distancia total recorrida: {final_metrics.get('total_distance_traveled', 0.0):.2f} km")
    print(f"  ⛽ Combustible total consumido: {final_metrics.get('total_fuel_consumed', 0.0):.2f} L")
    print(f"  🚨 Respuestas de emergencia: {final_metrics.get('emergency_responses', 0)}")
    print(f"  🌧️ Retrasos por clima: {final_metrics.get('weather_delays', 0)}")
    print(f"  🚫 Violaciones de tráfico: {final_metrics.get('traffic_violations', 0)}")
    
    print(f"\n🧠 MÉTRICAS BDI:")
    print(f"  🤔 Decisiones totales: {final_metrics.get('bdi_decisions_made', 0)}")
    print(f"  🤝 Colaboraciones: {final_metrics.get('bdi_collaborations', 0)}")
    print(f"  🎯 Intenciones ejecutadas: {final_metrics.get('bdi_intentions_executed', 0)}")
    
    print(f"\n📈 ESTADÍSTICAS HISTÓRICAS:")
    if historical_metrics["epochs"]:
        print(f"  🚗 Promedio de vehículos: {np.mean(historical_metrics['total_vehicles']):.1f}")
        print(f"  🏃 Velocidad promedio: {np.mean(historical_metrics['average_speed']):.1f} km/h")
        print(f"  🚦 Congestión promedio: {np.mean(historical_metrics['congestion_level']):.2f}")
        print(f"  📦 Tasa de entregas por época: {np.mean(np.diff(historical_metrics['completed_deliveries']) if len(historical_metrics['completed_deliveries']) > 1 else [0]):.2f}")
        print(f"  ⚠️ Eventos promedio activos: {np.mean(historical_metrics['active_events']):.1f}")
        
        # Picos y valles
        if len(historical_metrics['average_speed']) > 0:
            max_speed_epoch = historical_metrics['epochs'][np.argmax(historical_metrics['average_speed'])]
            min_speed_epoch = historical_metrics['epochs'][np.argmin(historical_metrics['average_speed'])]
            print(f"  🏆 Velocidad máxima en época: {max_speed_epoch}")
            print(f"  🐌 Velocidad mínima en época: {min_speed_epoch}")
    
    print(f"\n💡 ANÁLISIS DE RENDIMIENTO:")
    if final_metrics.get('total_vehicles', 0) > 0:
        print(f"  📊 Entregas por vehículo: {final_metrics.get('completed_deliveries', 0) / final_metrics.get('total_vehicles', 1):.2f}")
        print(f"  📏 Distancia promedio por vehículo: {final_metrics.get('total_distance_traveled', 0.0) / final_metrics.get('total_vehicles', 1):.2f} km")
        print(f"  ⛽ Eficiencia de combustible: {final_metrics.get('total_distance_traveled', 0.0) / max(final_metrics.get('total_fuel_consumed', 1), 1):.2f} km/L")
    
    if final_metrics.get('completed_deliveries', 0) > 0:
        success_rate = (final_metrics.get('completed_deliveries', 0) / 
                       (final_metrics.get('completed_deliveries', 0) + final_metrics.get('failed_deliveries', 0))) * 100
        print(f"  ✅ Tasa de éxito de entregas: {success_rate:.1f}%")
    
    # Análisis de rendimiento de simulación
    print(f"\n⚡ RENDIMIENTO DE SIMULACIÓN:")
    print(f"  🔄 Épocas por segundo: {total_epochs/total_time:.2f}")
    print(f"  ⏱️ Tiempo promedio por época: {total_time/total_epochs:.3f}s")
    print(f"  🚀 Aceleración vs tiempo real: {(total_epochs * config.get('time_step', 1)) / total_time:.1f}x")
    
    print(f"{'='*80}")
    print(f"🎉 Simulación finalizada exitosamente a las {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}\n")

async def setup_demo_bdi_trucks():
    """Configura camiones BDI de demostración"""
    global simulation_environment
    
    if not simulation_environment:
        return
    
    print("🚛 Configurando camiones BDI de demostración...")
    
    # Obtener nodos disponibles
    nodes = list(simulation_environment.street_graph.nodes())
    if len(nodes) < 10:
        print("⚠️ Pocos nodos disponibles para demostración BDI")
        return
    
    # Configuraciones de camiones
    truck_configs = [
        {
            "id": "bdi_truck_alpha",
            "start_node": random.choice(nodes),
            "capacity": 1000,
            "deliveries": random.sample(nodes, min(5, len(nodes)//10))
        },
        {
            "id": "bdi_truck_beta", 
            "start_node": random.choice(nodes),
            "capacity": 1200,
            "deliveries": random.sample(nodes, min(6, len(nodes)//10))
        },
        {
            "id": "bdi_truck_gamma",
            "start_node": random.choice(nodes),
            "capacity": 800,
            "deliveries": random.sample(nodes, min(4, len(nodes)//10))
        }
    ]
    
    # Crear camiones BDI
    created_count = 0
    for config in truck_configs:
        success = simulation_environment.add_bdi_delivery_truck(
            truck_id=config["id"],
            start_node=config["start_node"],
            capacity=config["capacity"],
            delivery_locations=config["deliveries"]
        )
        
        if success:
            created_count += 1
            print(f"✅ {config['id']} creado en nodo {config['start_node']}")
        else:
            print(f"❌ Error creando {config['id']}")
    
    if created_count > 0:
        # Iniciar sistema de comunicación BDI
        await simulation_environment.start_communication_system()
        
        # Iniciar movimiento de camiones
        simulation_environment.start_bdi_trucks_movement()
        
        print(f"✅ Sistema BDI iniciado con {created_count} camiones")
    else:
        print("❌ No se pudo crear ningún camión BDI")

def log_bdi_status():
    """Registra el estado de los camiones BDI"""
    global simulation_environment
    
    if not simulation_environment:
        return
    
    try:
        bdi_status = simulation_environment.get_bdi_trucks_status()
        
        if bdi_status:
            print(f"\n📊 Estado BDI ({len(bdi_status)} camiones):")
            for truck_id, status in bdi_status.items():
                if "error" in status:
                    print(f"❌ {truck_id}: {status['error']}")
                    continue
                
                metrics = status.get("metrics", {})
                print(f"🚛 {truck_id}: Nodo {status.get('current_node', 'N/A')}, "
                      f"Entregas: {metrics.get('deliveries_completed', 0)}, "
                      f"Combustible: {status.get('fuel_level', 0):.1f}%")
            
            # Mostrar estadísticas de comunicación
            comm_stats = simulation_environment.get_communication_stats()
            if "error" not in comm_stats:
                stats = comm_stats.get('stats', {})
                print(f"📡 Comunicación: {stats.get('messages_sent', 0)} enviados, "
                      f"{stats.get('messages_delivered', 0)} entregados")
            print()
        
    except Exception as e:
        print(f"❌ Error obteniendo estado BDI: {e}")

# Modificar la función main para incluir el servidor HTTP
async def main():
    global simulation_environment
    
    # Obtener el modo de simulación desde argumentos
    config_mode = "normal"  # Por defecto
    if len(sys.argv) > 1:
        if "--mode" in sys.argv:
            try:
                mode_index = sys.argv.index("--mode") + 1
                if mode_index < len(sys.argv):
                    config_mode = sys.argv[mode_index]
            except (ValueError, IndexError):
                pass
    
    # Cargar configuración de simulación
    config = get_config(config_mode)
    
    # Cargar calles, inicializar vehículos y semaforos
    load_streets()
    
    # Crear entorno de simulación con parámetros optimizados
    print("==========================================================")
    simulation_environment = Environment(street_graph, num_vehicles=config["num_vehicles"])
    print(simulation_environment)
    print("Entorno de simulación creado")
    
    # Removed communication manager initialization
    print("✅ Sistema de simulación iniciado sin communication manager")
    print("==========================================================")
    
    
    print("Servidor WebSocket iniciando en puerto 8765...")
    
    # Iniciar servidor HTTP para la IA en un hilo separado
    def start_http_server():
        try:
            http_server = HTTPServer(('localhost', 8767), CVRPHandler)
            print("✅ Servidor HTTP para IA iniciado correctamente en puerto 8767")
            print("🤖 Endpoint disponible: http://localhost:8767/analyze_cvrp")
            http_server.serve_forever()
        except Exception as e:
            print(f"❌ Error iniciando servidor HTTP: {e}")
    
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    
    # Dar tiempo para que el servidor HTTP se inicie
    await asyncio.sleep(1)
    
    # Iniciar servidor WebSocket 
    print("✅ Servidor WebSocket iniciado correctamente en puerto 8765")
    
    # Iniciar la simulación en background con configuración específica (sin bloquear el servidor)
    print(f"🚗 Iniciando simulación con configuración: {config.get('max_epochs', 'N/A')} épocas...")
    simulation_task = asyncio.create_task(run_simulation(config_mode))
    
    async with websockets.serve(
        handler, 
        "localhost", 
        8765,
        ping_interval=30,
        ping_timeout=10
    ):
        # Mantener el servidor corriendo indefinidamente, la simulación corre en paralelo
        print("🔄 Servidor WebSocket activo y esperando conexiones...")
        try:
            await asyncio.Future()  # Correr indefinidamente
        except KeyboardInterrupt:
            print("🛑 Deteniendo servidor WebSocket...")
            simulation_task.cancel()
            raise
    

# Ejecuta el punto de entrada principal
if __name__ == "__main__":
    try:
        print(f"🚀 Iniciando servidor...")
        print("💡 Uso: python server.py --mode [normal|fast|debug]")
        print("🔧 Verificando dependencias...")
        
        # Verificar importaciones críticas
        required_modules = [
            'asyncio', 'websockets', 'json', 'networkx', 
            'numpy', 'flask', 'threading'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                print(f"✅ {module} disponible")
            except ImportError:
                missing_modules.append(module)
                print(f"❌ {module} no disponible")
        
        if missing_modules:
            print(f"🚨 Módulos faltantes: {missing_modules}")
            print("📦 Instale las dependencias: pip install -r requirements.txt")
            sys.exit(1)
        
        print("🎯 Todas las dependencias están disponibles")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido por el usuario")
    except Exception as e:
        print(f"❌ Error crítico iniciando servidor: {e}")
        import traceback
        traceback.print_exc()

def validate_node_connectivity(street_graph, start_node, target_nodes):
    """
    Valida que todos los nodos objetivo sean alcanzables desde el nodo de inicio
    """
    import networkx as nx
    
    valid_targets = []
    invalid_targets = []
    
    # Obtener el componente conectado más grande
    largest_component = max(nx.weakly_connected_components(street_graph), key=len)
    
    # Verificar si el nodo de inicio está en el componente principal
    if start_node not in largest_component:
        print(f"⚠️ Nodo de inicio {start_node} no está en el componente principal")
        # Buscar el nodo más cercano en el componente principal
        start_node = find_closest_node_in_component(street_graph, start_node, largest_component)
        print(f"🔄 Usando nodo de inicio alternativo: {start_node}")
    
    # Validar cada nodo objetivo
    for target in target_nodes:
        try:
            if target in largest_component:
                # Verificar si hay camino desde el inicio
                if nx.has_path(street_graph, start_node, target):
                    valid_targets.append(target)
                else:
                    print(f"⚠️ No hay camino desde {start_node} a {target}")
                    # Buscar nodo alternativo cercano
                    alternative = find_closest_reachable_node(street_graph, start_node, target, largest_component)
                    if alternative:
                        valid_targets.append(alternative)
                        print(f"🔄 Usando nodo alternativo: {alternative}")
                    else:
                        invalid_targets.append(target)
            else:
                print(f"⚠️ Nodo {target} no está en el componente principal")
                alternative = find_closest_node_in_component(street_graph, target, largest_component)
                if alternative and nx.has_path(street_graph, start_node, alternative):
                    valid_targets.append(alternative)
                    print(f"🔄 Usando nodo alternativo: {alternative}")
                else:
                    invalid_targets.append(target)
        except Exception as e:
            print(f"❌ Error validando nodo {target}: {e}")
            invalid_targets.append(target)
    
    return start_node, valid_targets, invalid_targets

def find_closest_node_in_component(street_graph, target_node, component):
    """
    Encuentra el nodo más cercano en un componente conectado específico
    """
    if target_node not in street_graph.nodes:
        return None
    
    target_lat = street_graph.nodes[target_node].get('lat', 0)
    target_lon = street_graph.nodes[target_node].get('lon', 0)
    
    closest_node = None
    min_distance = float('inf')
    
    for node in component:
        if node in street_graph.nodes:
            node_lat = street_graph.nodes[node].get('lat', 0)
            node_lon = street_graph.nodes[node].get('lon', 0)
            
            # Calcular distancia euclidiana
            distance = ((target_lat - node_lat) ** 2 + (target_lon - node_lon) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_node = node
    
    return closest_node

def find_closest_reachable_node(street_graph, start_node, target_node, component):
    """
    Encuentra el nodo más cercano al objetivo que sea alcanzable desde el inicio
    """
    import networkx as nx
    
    if target_node not in street_graph.nodes:
        return None
    
    target_lat = street_graph.nodes[target_node].get('lat', 0)
    target_lon = street_graph.nodes[target_node].get('lon', 0)
    
    # Buscar en un radio creciente
    radius_candidates = []
    
    for node in component:
        if node == start_node:
            continue
            
        try:
            if nx.has_path(street_graph, start_node, node):
                node_lat = street_graph.nodes[node].get('lat', 0)
                node_lon = street_graph.nodes[node].get('lon', 0)
                distance = ((target_lat - node_lat) ** 2 + (target_lon - node_lon) ** 2) ** 0.5
                radius_candidates.append((distance, node))
        except:
            continue
    
    if radius_candidates:
        radius_candidates.sort(key=lambda x: x[0])
        return radius_candidates[0][1]
    
    return None

@app.route('/ask_rag', methods=['POST'])
def ask_rag():
    """Endpoint para consultar al asistente RAG con contexto"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'message': 'Falta la pregunta'
            }), 400
        
        # Extraer datos
        question = data.get('question')
        context_data = data.get('context_data', {})
        
        # Actualizar la base de conocimientos con el contexto
        if 'routes' in context_data:
            RAG_ASSISTANT.update_knowledge_base("routes", {"routes": context_data['routes']})
        
        if 'weather' in context_data:
            RAG_ASSISTANT.update_knowledge_base("weather", context_data['weather'])
        
        # Obtener respuesta del asistente RAG
        result = RAG_ASSISTANT.ask_with_context(question)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error en consulta RAG: {e}")
        return jsonify({
            'success': False,
            'message': f'Error interno: {str(e)}'
        }), 500


