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

        # Ejecutar el solver Simulated Annealing
        routes = sa_solver.solve(
            N=n,  # Número total de nodos (incluyendo el depósito)
            T=num_trucks,  # Número de vehículos
            capacity=truck_capacities,  # Capacidades de cada vehículo
            demand=demands,  # Demandas de cada nodo
            distMat=dist_matrix,  # Matriz de distancias
            T0=100.0,  # Temperatura inicial
            Tf=0.1,  # Temperatura final
            alpha=0.98,  # Factor de enfriamiento
            iterPerTemp=100,  # Iteraciones por nivel de temperatura
            lambdaPen=1000.0,  # Penalización por usar más vehículos
            maxSeconds=30.0,  # Tiempo máximo de ejecución en segundos
            seed=42  # Semilla aleatoria para reproducibilidad
        )
        
        # Preparar los objetivos (índices de 1 a n-1, excluyendo el depósito)
        objectives = list(range(1, n))
        
        
        # ////////// TABU SEARCH SOLVER //////////
        
        # Ejecutar el solver Tabu Search
        # routes = ts_solver.solve_vrp(
        #     dist_matrix,
        #     objectives,
        #     demands,
        #     truck_capacities,
        #     num_trucks,
        #     max_iter=1000,
        #     base_tabu_tenure=20,
        #     no_improve_limit=200,
        #     diversification_interval=500
        # )
        
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