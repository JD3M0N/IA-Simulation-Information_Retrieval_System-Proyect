import networkx as nx
import sys
sys.path.append("src/metaheuristic/ag_solver")
import ag_solver as ag_solver
sys.path.append("src/metaheuristic/vns_solver") 
import vns_solver 
sys.path.append("src/metaheuristic/sa_solver") 
import sa_solver 
sys.path.append("src/metaheuristic/ts_solver") 
import ts_solver  
from distance_matrix import load_distance_cache, save_distance_cache, save_distance_to_cache, get_distance_from_cache


# def expand_route_with_path_nodes(street_graph, route):
#     """
#     Expande una ruta para incluir todos los nodos intermedios del camino
#     """
#     expanded_route = []
#     for i in range(len(route) - 1):
#         src = route[i]
#         dst = route[i + 1]
#         try:
#             # Obtener el camino completo entre los dos puntos
#             path = nx.shortest_path(street_graph, src, dst, weight='weight')
#             # Añadir todos los nodos excepto el último (para evitar duplicados)
#             expanded_route.extend(path[:-1])
#         except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
#             print(f"Error al expandir ruta entre {src} y {dst}: {e}")
#             # Si no hay camino, al menos incluir el origen
#             expanded_route.append(src)
    
#     # Añadir el último nodo de la ruta original
#     if route:
#         expanded_route.append(route[-1])
    
#     return expanded_route

def is_source_fully_cached(source, all_points, distance_cache):
    """
    Verifica si todas las distancias desde source a todos los puntos en all_points
    ya están almacenadas en la caché.
    """
    # Ignoramos el propio nodo source (distancia a sí mismo es 0)
    targets_needed = len(all_points) - 1
    targets_cached = 0
    
    for target in all_points:
        if source == target:
            continue
        if get_distance_from_cache(source, target, distance_cache) is not None:
            targets_cached += 1
    
    return targets_cached == targets_needed

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
        # Cargar caché de distancias
        distance_cache = load_distance_cache()
        cache_modified = False
        
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
        dist_matrix = [[float('inf') for _ in range(n)] for _ in range(n)]
        
        # Calcular distancias reales usando el algoritmo de camino más corto
        for i in range(n):
            # La distancia de un nodo a sí mismo es 0
            dist_matrix[i][i] = 0
            source = all_points[i]
            
            # Verificar si todas las distancias desde source ya están en caché
            if not is_source_fully_cached(source, all_points, distance_cache):
                try:
                    # Calcular todas las distancias desde source en una sola llamada
                    shortest_paths = nx.single_source_dijkstra_path_length(
                        street_graph, source, weight='weight')
                    
                    # Llenar toda la fila i (distancias desde source a todos los demás)
                    for j in range(n):
                        if i == j:
                            continue  # Ya asignado como 0
                        
                        target = all_points[j]
                        distance = shortest_paths.get(target, float('inf'))
                        
                        # Guardar en la matriz
                        dist_matrix[i][j] = distance
                        
                        # Guardar en caché (solo una dirección, ya que es un grafo dirigido)
                        distance_cache = save_distance_to_cache(source, target, distance, distance_cache)
                    
                    cache_modified = True
                except nx.NetworkXError as e:
                    print(f"Error al calcular distancias desde {source}: {e}")
            else:
                # Si todas las distancias ya están en caché, solo recuperarlas
                for j in range(n):
                    if i == j:
                        continue  # Ya asignado como 0
                    
                    target = all_points[j]
                    cached_distance = get_distance_from_cache(source, target, distance_cache)
                    if cached_distance is not None:
                        dist_matrix[i][j] = cached_distance
        
        # Guardar caché si se modificó
        if cache_modified:
            save_distance_cache(distance_cache)
        
        # Preparar vector de demandas (incluyendo el depósito como 0)
        demands = [0] + target_demands
        
        # ////////// GENETIC ALGORITHM SOLVER //////////
        # routes = ag_solver.solve_vrp(
        #     dist_matrix, 
        #     demands, 
        #     truck_capacities, 
        #     pop_size=400,
        #     sel_size=40,
        #     max_gen=1000,
        #     no_improve_limit=20,
        #     mut_rate=0.3)
        
        # ////////// SIMULATED ANNEALING SOLVER //////////

        # routes = sa_solver.solve(
        #     N=n,  # Número total de nodos (incluyendo el depósito)
        #     T=num_trucks,  # Número de vehículos
        #     capacity=truck_capacities,  # Capacidades de cada vehículo
        #     demand=demands,  # Demandas de cada nodo
        #     distMat=dist_matrix,  # Matriz de distancias
        #     T0=100.0,  # Temperatura inicial
        #     Tf=0.1,  # Temperatura final
        #     alpha=0.98,  # Factor de enfriamiento
        #     iterPerTemp=100,  # Iteraciones por nivel de temperatura
        #     lambdaPen=1000.0,  # Penalización por usar más vehículos
        #     maxSeconds=30.0,  # Tiempo máximo de ejecución en segundos
        #     seed=42  # Semilla aleatoria para reproducibilidad
        # )
        
        # Preparar los objetivos (índices de 1 a n-1, excluyendo el depósito)
        objectives = list(range(1, n))
        
        
        # ////////// TABU SEARCH SOLVER //////////
        
        routes = ts_solver.solve_vrp(
            dist_matrix,
            objectives,
            demands,
            truck_capacities,
            num_trucks,
            max_iter=1000,
            base_tabu_tenure=100,
            no_improve_limit=200,
            diversification_interval=500
        )
        
        # Calcular el costo total de las rutas
        total_cost = 0
        final_routes = []
        
        # Crear un mapa de índices para búsqueda rápida
        node_index_map = {node: idx for idx, node in enumerate(all_points)}
        
        for route in routes:
            # Convertir índices de ruta a IDs de nodos reales
            real_route = [all_points[i] for i in route]
            # Añadir el depósito al principio y final de cada ruta
            complete_route = [start_point] + real_route + [start_point]
            
            # Expandir la ruta para incluir nodos intermedios
            paths_and_costs = []
            route_cost = 0
            
            # Primero, calculamos los caminos más cortos entre cada par de nodos consecutivos
            for i in range(len(complete_route) - 1):
                src = complete_route[i]
                dst = complete_route[i + 1]
                
                # Intentamos obtener el camino y costo de la matriz o caché
                path = None
                cost = None
                
                # Primero intentamos usar la matriz de distancias precalculada
                if src in node_index_map and dst in node_index_map:
                    src_idx = node_index_map[src]
                    dst_idx = node_index_map[dst]
                    if dist_matrix[src_idx][dst_idx] != float('inf'):
                        cost = dist_matrix[src_idx][dst_idx]
                        try:
                            path = nx.shortest_path(street_graph, src, dst, weight='weight')
                        except nx.NetworkXError:
                            path = [src, dst]  # Fallback
                
                # Si no está en la matriz, intentamos caché
                if cost is None:
                    cached_distance = get_distance_from_cache(src, dst, distance_cache)
                    if cached_distance is not None:
                        cost = cached_distance
                        try:
                            path = nx.shortest_path(street_graph, src, dst, weight='weight')
                        except nx.NetworkXError:
                            path = [src, dst]  # Fallback
                
                # Como último recurso, calculamos el camino
                if cost is None or path is None:
                    try:
                        path = nx.shortest_path(street_graph, src, dst, weight='weight')
                        # Calcular costo sumando los pesos de las aristas en el camino
                        segment_cost = 0
                        for k in range(len(path) - 1):
                            u = path[k]
                            v = path[k + 1]
                            segment_cost += street_graph[u][v]['weight']
                        
                        cost = segment_cost
                        
                        # Guardar en caché
                        distance_cache = save_distance_to_cache(src, dst, cost, distance_cache)
                        cache_modified = True
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        print(f"Advertencia: No hay camino entre {src} y {dst}")
                        path = [src, dst]  # Fallback
                        cost = 0  # No sumamos costo para este segmento
                
                paths_and_costs.append((path, cost))
                route_cost += cost
            
            # Construir la ruta expandida y eliminar duplicados
            expanded_route = []
            for i, (path, _) in enumerate(paths_and_costs):
                if i == len(paths_and_costs) - 1:
                    # Incluir todo el último camino
                    expanded_route.extend(path)
                else:
                    # Excluir el último nodo para evitar duplicados
                    expanded_route.extend(path[:-1])
            
            final_routes.append(expanded_route)
            total_cost += route_cost
        
        # Guardar caché actualizada si se modificó
        if cache_modified:
            save_distance_cache(distance_cache)
        
        return final_routes, total_cost
        
    except Exception as e:
        print(f"Error en la optimización de rutas: {e}")
        import traceback
        traceback.print_exc()
        return [], 0
    

# def optimize_delivery_routes(street_graph, start_point, target_points, num_trucks=1, 
#                             truck_capacities=None, target_demands=None,
#                             max_iterations=2000, time_limit=10.0):
#     """
#     Optimiza rutas de entrega utilizando el solver VNS avanzado
    
#     Args:
#         street_graph: Grafo de NetworkX con la red de calles
#         start_point: Nodo de inicio (depósito)
#         target_points: Lista de nodos objetivo
#         num_trucks: Número de vehículos disponibles
#         truck_capacities: Lista de capacidades de los vehículos
#         target_demands: Lista de demandas para cada punto objetivo
#         max_iterations: Número máximo de iteraciones para VNS
#         time_limit: Límite de tiempo en segundos
    
#     Returns:
#         (rutas_optimizadas, costo_total)
#     """
#     try:
#         # Validar y preparar capacidades
#         if not truck_capacities:
#             truck_capacities = [100] * num_trucks  # Capacidad por defecto
#         elif len(truck_capacities) < num_trucks:
#             # Extender con la última capacidad si faltan valores
#             truck_capacities.extend([truck_capacities[-1]] * (num_trucks - len(truck_capacities)))
        
#         # Validar y preparar demandas
#         if not target_demands:
#             target_demands = [1] * len(target_points)  # Demanda por defecto
#         elif len(target_demands) < len(target_points):
#             # Extender con la última demanda si faltan valores
#             target_demands.extend([target_demands[-1]] * (len(target_points) - len(target_demands)))
        
#         # Crear matriz de distancias
#         all_points = [start_point] + target_points
#         n = len(all_points)
#         dist_matrix = []
        
#         # Inicializar matriz con distancias infinitas
#         for i in range(n):
#             row = []
#             for j in range(n):
#                 row.append(float('inf'))
#             dist_matrix.append(row)
        
#         # Calcular distancias reales usando el algoritmo de camino más corto
#         for i in range(n):
#             # La distancia de un nodo a sí mismo es 0
#             dist_matrix[i][i] = 0
#             source = all_points[i]
            
#             # Calcular las distancias desde este nodo a todos los demás
#             try:
#                 # Usar Dijkstra para calcular las distancias más cortas
#                 shortest_paths = nx.single_source_dijkstra_path_length(
#                     street_graph, source, weight='weight')
                
#                 for j in range(n):
#                     target = all_points[j]
#                     if target in shortest_paths:
#                         dist_matrix[i][j] = shortest_paths[target]
#             except nx.NetworkXNoPath:
#                 print(f"Advertencia: No hay camino desde {source} a algunos destinos")
#                 continue
        
#         # Preparar vector de demandas (incluyendo el depósito como 0)
#         demands = [0] + target_demands
        
#         # Ejecutar el solver VNS
#         solution = vns_solver.vns_hetero_improved(
#             dist_matrix, 
#             demands, 
#             truck_capacities, 
#             max_iterations, 
#             time_limit
#         )
        
#         # Calcular el costo total de las rutas
#         total_cost = solution.total_cost
#         final_routes = []
        
#         for route in solution.routes:
#             # Solo procesamos rutas que no estén vacías
#             if not route.nodes:
#                 continue
                
#             # Convertir índices de ruta a IDs de nodos reales
#             real_route = [all_points[i] for i in route.nodes]
#             # Añadir el depósito al principio y final de cada ruta
#             complete_route = [start_point] + real_route + [start_point]
#             expanded_route = expand_route_with_path_nodes(street_graph, complete_route)
#             final_routes.append(expanded_route)
        
#         return final_routes, total_cost
        
#     except Exception as e:
#         print(f"Error en la optimización de rutas: {e}")
#         import traceback
#         traceback.print_exc()
#         return [], 0