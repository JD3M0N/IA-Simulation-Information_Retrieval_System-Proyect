import random

# Número de vehículos a simular (reducido para depuración)
NUM_VEHICLES = 10

def initialize_vehicles(street_graph, all_nodes, vehicle_speeds, vehicles):
    """Inicializa la posición de los vehículos y les asigna rutas aleatorias"""
    
    
    
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
        assign_random_route(vid, street_graph, all_nodes, vehicles)
    
    print(f"Vehículos inicializados: {len(vehicles)}")

def assign_random_route(vehicle_id, street_graph, all_nodes, vehicles):
    """Asigna una ruta aleatoria a un vehículo"""
    
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
            assign_random_route(vehicle_id, street_graph, all_nodes, vehicles)
    
    except Exception as e:
        print(f"Error asignando ruta para {vehicle_id}: {e}")

def update_vehicle_positions(street_graph, traffic_lights, vehicles, vehicle_speeds, all_nodes):
    """Actualiza las posiciones de los vehículos en sus rutas"""
    for vid, v in vehicles.items():
        current_node = v["current_node"]
        next_node = v["next_node"]
        
        # Verifica si hay semáforo en el siguiente nodo y si está en rojo
        if next_node in traffic_lights and traffic_lights[next_node]["state"] == "red":
            continue  # el vehículo espera

        # Si no tiene un nodo siguiente, asignar uno
        if next_node is None:
            assign_random_route(vid, street_graph, all_nodes, vehicles)
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
                assign_random_route(vid, street_graph, all_nodes, vehicles)
            else:
                # Interpolación lineal entre los nodos
                v["lat"] = current_lat + (next_lat - current_lat) * v["progress"]
                v["lon"] = current_lon + (next_lon - current_lon) * v["progress"]
                
        except Exception as e:
            # En caso de error, asignar una nueva ruta
            print(f"Error en actualización de {vid}: {e}")
            assign_random_route(vid, street_graph, all_nodes, vehicles)