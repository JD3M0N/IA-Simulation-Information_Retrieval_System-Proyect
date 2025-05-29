import sys
sys.path.append("src/vrp_solver")
import improved_genetic_vrp_final as vrp_solver
import networkx as nx

def expand_route_with_path_nodes(street_graph, route):
    """
    Expande una ruta para incluir todos los nodos intermedios del camino
    """
    expanded_route = []
    for i in range(len(route) - 1):
        src = route[i]
        dst = route[i + 1]
        try:
            # Obtener el camino completo entre los dos puntos
            path = nx.shortest_path(street_graph, src, dst, weight='weight')
            # Añadir todos los nodos excepto el último (para evitar duplicados)
            expanded_route.extend(path[:-1])
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            print(f"Error al expandir ruta entre {src} y {dst}: {e}")
            # Si no hay camino, al menos incluir el origen
            expanded_route.append(src)
    
    # Añadir el último nodo de la ruta original
    if route:
        expanded_route.append(route[-1])
    
    return expanded_route


def optimize_delivery_routes(street_graph, start_point, target_points, num_trucks=1, 
                            truck_capacities=None, target_demands=None):
    """
    Optimiza rutas de entrega utilizando el solver VRP avanzado
    
    Args:
        street_graph: Grafo de NetworkX con la red de calles
        start_point: Nodo de inicio (depósito)
        target_points: Lista de nodos objetivo
        num_trucks: Número de vehículos disponibles
        truck_capacities: Lista de capacidades de los vehículos
        target_demands: Lista de demandas para cada punto objetivo
    
    Returns:
        (rutas_optimizadas, costo_total)
    """
    try:
        # Validar y preparar capacidades
        if not truck_capacities:
            truck_capacities = [100] * num_trucks  # Capacidad por defecto
        elif len(truck_capacities) < num_trucks:
            # Extender con la última capacidad si faltan valores
            truck_capacities.extend([truck_capacities[-1]] * (num_trucks - len(truck_capacities)))
        
        # Validar y preparar demandas
        if not target_demands:
            target_demands = [1] * len(target_points)  # Demanda por defecto
        elif len(target_demands) < len(target_points):
            # Extender con la última demanda si faltan valores
            target_demands.extend([target_demands[-1]] * (len(target_points) - len(target_demands)))
        
        # Crear matriz de distancias
        all_points = [start_point] + target_points
        n = len(all_points)
        dist_matrix = []
        
        # Inicializar matriz con distancias infinitas
        for i in range(n):
            row = []
            for j in range(n):
                row.append(float('inf'))
            dist_matrix.append(row)
        
        # Calcular distancias reales usando el algoritmo de camino más corto
        for i in range(n):
            # La distancia de un nodo a sí mismo es 0
            dist_matrix[i][i] = 0
            source = all_points[i]
            
            # Calcular las distancias desde este nodo a todos los demás
            try:
                # Usar Dijkstra para calcular las distancias más cortas
                shortest_paths = nx.single_source_dijkstra_path_length(
                    street_graph, source, weight='weight')
                
                for j in range(n):
                    target = all_points[j]
                    if target in shortest_paths:
                        dist_matrix[i][j] = shortest_paths[target]
            except nx.NetworkXNoPath:
                print(f"Advertencia: No hay camino desde {source} a algunos destinos")
                continue
        
        # Preparar vector de demandas (incluyendo el depósito como 0)
        demands = [0] + target_demands
        
        # Ejecutar el solver VRP
        routes = vrp_solver.solve_vrp(dist_matrix, demands, truck_capacities)
        
        # Calcular el costo total de las rutas
        total_cost = 0
        final_routes = []
        
        for route in routes:
            # Convertir índices de ruta a IDs de nodos reales
            real_route = [all_points[i] for i in route]
            # Añadir el depósito al principio y final de cada ruta
            complete_route = [start_point] + real_route + [start_point]
            expanded_route = expand_route_with_path_nodes(street_graph, complete_route)
            final_routes.append(expanded_route)
            
            # Calcular costo de esta ruta
            route_cost = 0
            for i in range(len(complete_route) - 1):
                src = complete_route[i]
                dst = complete_route[i + 1]
                try:
                    # Usar el camino más corto entre cada par de nodos
                    route_cost += nx.shortest_path_length(
                        street_graph, src, dst, weight='weight')
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    print(f"Advertencia: No hay camino entre {src} y {dst}")
                    continue
            
            total_cost += route_cost
        
        return final_routes, total_cost
        
    except Exception as e:
        print(f"Error en la optimización de rutas: {e}")
        import traceback
        traceback.print_exc()
        return [], 0