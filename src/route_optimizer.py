import networkx as nx
import random
from collections import defaultdict
import math

def calculate_distance(G, node1, node2):
    try:
        return nx.shortest_path_length(G, source=node1, target=node2, weight="weight")
    except nx.NetworkXNoPath:
        return float('inf')

def total_route_cost(G, route):
    cost = 0
    for i in range(len(route) - 1):
        cost += calculate_distance(G, route[i], route[i+1])
    return cost

def aco_vrp(G, depot, objectives, num_vehicles, capacities, demands, iterations=50, alpha=1, beta=2, evaporation=0.5):
    pheromones = defaultdict(lambda: 1.0)
    best_solution = None
    best_cost = float("inf")

    for it in range(iterations):
        solutions = []
        for _ in range(num_vehicles):
            current_node = depot
            load = 0
            visited = set()
            route = [depot]

            while len(visited) < len(objectives):
                feasible = [obj for obj in objectives if obj not in visited and load + demands[obj] <= capacities[_]]
                if not feasible:
                    break

                probabilities = []
                for target in feasible:
                    dist = calculate_distance(G, current_node, target)
                    tau = pheromones[(current_node, target)]
                    eta = 1.0 / (dist + 1e-6)
                    probabilities.append((target, (tau ** alpha) * (eta ** beta)))

                total = sum(p[1] for p in probabilities)
                chosen = random.choices(
                    [p[0] for p in probabilities],
                    weights=[p[1]/total for p in probabilities],
                    k=1
                )[0]

                route.append(chosen)
                visited.add(chosen)
                load += demands[chosen]
                current_node = chosen

            route.append(depot)  # Return to depot
            cost = total_route_cost(G, route)
            solutions.append((route, cost))

            # Update pheromones locally
            for i in range(len(route)-1):
                pheromones[(route[i], route[i+1])] += 1.0 / cost

        # Global update
        for k in list(pheromones.keys()):
            pheromones[k] *= (1 - evaporation)

        best_in_iteration = min(solutions, key=lambda x: x[1])
        if best_in_iteration[1] < best_cost:
            best_solution = best_in_iteration
            best_cost = best_in_iteration[1]

    return best_solution

def optimize_delivery_routes(street_graph, start_point, target_points, num_trucks, 
                             truck_capacities=None, target_demands=None, 
                             optimization_iterations=100):
    """
    Optimiza las rutas de entrega para una flota de camiones.
    
    Args:
        street_graph: Grafo de calles de NetworkX
        start_point: ID del nodo de salida
        target_points: Lista de IDs de nodos objetivo
        num_trucks: Número de camiones disponibles
        truck_capacities: Lista de capacidades de cada camión (si es None, se asume capacidad infinita)
        target_demands: Diccionario con la demanda de cada punto objetivo (si es None, se asume demanda de 1)
        optimization_iterations: Número de iteraciones para el algoritmo ACO
        
    Returns:
        Una tupla con (rutas_optimizadas, costo_total) donde rutas_optimizadas es una lista de listas,
        cada una representando la ruta de un camión.
    """
    # Configurar valores por defecto si no se proporcionan
    if truck_capacities is None:
        truck_capacities = [float('inf')] * num_trucks
    
    if target_demands is None:
        target_demands = {point: 1 for point in target_points}
    
    # Ejecutar el algoritmo ACO
    solution = aco_vrp(
        G=street_graph,
        depot=start_point,
        objectives=target_points,
        num_vehicles=num_trucks,
        capacities=truck_capacities,
        demands=target_demands,
        iterations=optimization_iterations,
        alpha=1.0,  # Importancia de las feromonas
        beta=2.0,   # Importancia de la distancia
        evaporation=0.5  # Tasa de evaporación de feromonas
    )
    
    if solution:
        routes, total_cost = solution
        return routes, total_cost
    else:
        return None, float('inf')
