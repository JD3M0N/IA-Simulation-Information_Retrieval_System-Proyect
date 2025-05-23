import random

# Número de vehículos a simular (reducido para depuración)
NUM_VEHICLES = 10000

def initialize_vehicles(street_graph, all_nodes, vehicle_speeds, vehicles):
    """Inicializa la posición de los vehículos y les asigna rutas aleatorias"""
    
    
    if not all_nodes:
        print("No hay nodos en el grafo. Inicialización fallida.")
        return
    
    print(f"Inicializando {NUM_VEHICLES} vehículos...")
    
    # Categorías de vehículos con diferentes comportamientos
    vehicle_types = ["normal", "agresivo", "cauteloso", "lento", "rápido"]
    
    for i in range(NUM_VEHICLES):
        vid = f"veh_{i}"
        
        # Seleccionar un nodo aleatorio para comenzar
        start_node = random.choice(all_nodes)
        node_data = street_graph.nodes[start_node]
        
        # Tipo de vehículo para comportamiento
        vehicle_type = random.choice(vehicle_types)
        
        # Ubicar el vehículo en ese nodo
        vehicles[vid] = {
            "lat": float(node_data['lat']),
            "lon": float(node_data['lon']),
            "current_node": start_node,
            "next_node": None,
            "previous_node": None,  # Para mantener dirección
            "progress": 0.0,
            "type": vehicle_type,   # Tipo de comportamiento
        }
        
        # Asignar velocidad según tipo de vehículo
        base_speed = 0.005
        if vehicle_type == "rápido":
            speed_factor = random.uniform(1.3, 1.8)
        elif vehicle_type == "agresivo":
            speed_factor = random.uniform(1.2, 1.5)
        elif vehicle_type == "lento":
            speed_factor = random.uniform(0.5, 0.8)
        elif vehicle_type == "cauteloso":
            speed_factor = random.uniform(0.7, 0.9)
        else:  # normal
            speed_factor = random.uniform(0.9, 1.1)
        
        # Asignar velocidad aleatoria (km por actualización)
        # Incrementada para que se mueva más rápido y sea visible
        vehicle_speeds[vid] = base_speed * speed_factor
        
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

def plan_continuous_route(vehicle_id, street_graph, all_nodes, vehicles):
    """Asigna una ruta que mantiene la dirección de movimiento actual"""
    try:
        current = vehicles[vehicle_id]["current_node"]
        previous = vehicles[vehicle_id].get("previous_node")
        
        # Encontrar nodos conectados
        neighbors = list(street_graph.neighbors(current))
        
        if not neighbors:
            # Si no hay vecinos, asignar un nodo aleatorio
            assign_random_route(vehicle_id, street_graph, all_nodes, vehicles)
            return
            
        if previous and len(neighbors) > 1:
            # Intentar mantener la dirección (no volver al nodo anterior)
            filtered_neighbors = [n for n in neighbors if n != previous]
            if filtered_neighbors:
                destination = random.choice(filtered_neighbors)
            else:
                destination = random.choice(neighbors)
        else:
            # Sin nodo anterior, simplemente elegir uno aleatorio
            destination = random.choice(neighbors)
        
        # Guardar el nodo actual como previo para la próxima vez
        vehicles[vehicle_id]["previous_node"] = current
        
        # Asignar el próximo nodo y resetear progreso
        vehicles[vehicle_id]["next_node"] = destination
        vehicles[vehicle_id]["progress"] = 0.0
        
    except Exception as e:
        print(f"Error planificando ruta continua para {vehicle_id}: {e}")
        # Fallback a ruta aleatoria
        assign_random_route(vehicle_id, street_graph, all_nodes, vehicles)
        


def update_vehicle_positions(street_graph, traffic_lights, vehicles, vehicle_speeds, all_nodes):
    """Actualiza las posiciones de los vehículos en sus rutas"""
    for vid, v in vehicles.items():
        current_node = v["current_node"]
        next_node = v["next_node"]

        # Si no tiene un nodo siguiente, asignar uno
        if next_node is None:
            assign_random_route(vid, street_graph, all_nodes, vehicles)
            continue
            
        # Obtener las coordenadas de los nodos
        try:
            # Comportamiento ante semáforos
            if next_node in traffic_lights:
                light_state = traffic_lights[next_node]["state"]
                
                # Si el semáforo está en rojo y estamos cerca, desacelerar
                if light_state == "red" and v["progress"] > 0.7 and v["progress"] < 0.9:
                    # Reducir velocidad drásticamente cuando está cerca
                    v["progress"] += vehicle_speeds[vid] * 0.1
                    continue
                elif light_state == "red" and v["progress"] > 0.95:
                    # Reducir velocidad moderadamente con semáforo amarillo
                    v["progress"] += vehicle_speeds[vid] * 0
                    continue
            
            
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
                
                # Decidir si continuar en la misma dirección o cambiar
                if random.random() < 0.7:  # 70% de probabilidad de mantener dirección
                    plan_continuous_route(vid, street_graph, all_nodes, vehicles)
                else:
                    # Asignar una nueva ruta aleatoria
                    assign_random_route(vid, street_graph, all_nodes, vehicles)
                    
            else:
                # Interpolación lineal entre los nodos con pequeña variación
                variation = random.uniform(-0.000000005, 0.000000005)  # Pequeña variación para evitar solapamiento
                v["lat"] = current_lat + (next_lat - current_lat) * v["progress"] + variation
                v["lon"] = current_lon + (next_lon - current_lon) * v["progress"] + variation
                
        except Exception as e:
            # En caso de error, asignar una nueva ruta
            print(f"Error en actualización de {vid}: {e}")
            assign_random_route(vid, street_graph, all_nodes, vehicles)