"""
Optimizador de Semáforos con Algoritmos Avanzados
Implementa algoritmos de optimización para mejorar el rendimiento de la red de semáforos
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import json

from .traffic_light_models import (
    TrafficLightData, TrafficLightPhase, IntersectionData,
    TrafficDirection, TrafficFlow, PriorityLevel
)
from .traffic_light_utils import (
    calculate_optimal_cycle_time, calculate_green_splits,
    calculate_intersection_delay, calculate_level_of_service
)


@dataclass
class OptimizationResult:
    """Resultado de una optimización"""
    algorithm: str
    intersection_id: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percent: float
    execution_time: float
    parameters_used: Dict[str, Any]
    timestamp: datetime


class TrafficLightOptimizer:
    """
    Optimizador avanzado para redes de semáforos
    Implementa múltiples algoritmos de optimización
    """
    
    def __init__(self):
        """Inicializa el optimizador"""
        self.logger = logging.getLogger("TrafficLightOptimizer")
        
        # Algoritmos disponibles
        self.available_algorithms = {
            "genetic": self._genetic_algorithm,
            "simulated_annealing": self._simulated_annealing,
            "particle_swarm": self._particle_swarm_optimization,
            "hill_climbing": self._hill_climbing,
            "webster": self._webster_optimization,
            "adaptive": self._adaptive_optimization
        }
        
        # Parámetros de optimización
        self.optimization_params = {
            "genetic": {
                "population_size": 50,
                "generations": 100,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            },
            "simulated_annealing": {
                "initial_temperature": 1000.0,
                "cooling_rate": 0.95,
                "min_temperature": 1.0,
                "max_iterations": 1000
            },
            "particle_swarm": {
                "particles": 30,
                "iterations": 100,
                "inertia": 0.7,
                "cognitive": 1.5,
                "social": 1.5
            },
            "hill_climbing": {
                "max_iterations": 500,
                "step_size": 1.0,
                "tolerance": 0.001
            }
        }
        
        # Historial de optimizaciones
        self.optimization_history: List[OptimizationResult] = []
        
        # Cache de resultados
        self.result_cache: Dict[str, Any] = {}
        
        self.logger.info("Optimizador de semáforos inicializado")
    
    async def optimize_intersection(self, intersection_data: IntersectionData,
                                  algorithm: str = "adaptive",
                                  constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Optimiza una intersección usando el algoritmo especificado
        
        Args:
            intersection_data: Datos de la intersección
            algorithm: Algoritmo a utilizar
            constraints: Restricciones de optimización
            
        Returns:
            OptimizationResult: Resultado de la optimización
        """
        start_time = datetime.now()
        
        if algorithm not in self.available_algorithms:
            raise ValueError(f"Algoritmo no disponible: {algorithm}")
        
        self.logger.info(f"Iniciando optimización de {intersection_data.intersection_id} con {algorithm}")
        
        # Obtener métricas antes de la optimización
        before_metrics = await self._calculate_intersection_metrics(intersection_data)
        
        # Ejecutar algoritmo de optimización
        optimization_func = self.available_algorithms[algorithm]
        optimized_config = await optimization_func(intersection_data, constraints or {})
        
        # Aplicar configuración optimizada
        await self._apply_optimization_config(intersection_data, optimized_config)
        
        # Calcular métricas después de la optimización
        after_metrics = await self._calculate_intersection_metrics(intersection_data)
        
        # Calcular mejora
        improvement = self._calculate_improvement(before_metrics, after_metrics)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Crear resultado
        result = OptimizationResult(
            algorithm=algorithm,
            intersection_id=intersection_data.intersection_id,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percent=improvement,
            execution_time=execution_time,
            parameters_used=optimized_config,
            timestamp=datetime.now()
        )
        
        self.optimization_history.append(result)
        
        self.logger.info(f"Optimización completada: {improvement:.1f}% mejora en {execution_time:.2f}s")
        
        return result
    
    async def optimize_network(self, intersections: Dict[str, IntersectionData],
                             algorithm: str = "adaptive") -> List[OptimizationResult]:
        """
        Optimiza una red completa de intersecciones
        
        Args:
            intersections: Diccionario de intersecciones
            algorithm: Algoritmo a utilizar
            
        Returns:
            List[OptimizationResult]: Resultados por intersección
        """
        results = []
        
        # Priorizar intersecciones por nivel de congestión
        sorted_intersections = sorted(
            intersections.items(),
            key=lambda x: x[1].get_total_vehicle_count(),
            reverse=True
        )
        
        for intersection_id, intersection_data in sorted_intersections:
            try:
                result = await self.optimize_intersection(intersection_data, algorithm)
                results.append(result)
                
                # Pequeña pausa entre optimizaciones
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error optimizando intersección {intersection_id}: {e}")
        
        # Optimización global adicional
        if len(results) > 1:
            await self._global_network_optimization(intersections, results)
        
        return results
    
    async def _genetic_algorithm(self, intersection_data: IntersectionData,
                               constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Algoritmo genético para optimización de timings"""
        params = self.optimization_params["genetic"]
        
        # Definir rangos de variables
        variable_ranges = {
            "cycle_time": (60, 180),
            "green_ns_ratio": (0.3, 0.7),
            "green_ew_ratio": (0.3, 0.7),
            "yellow_time": (3, 8)
        }
        
        # Generar población inicial
        population = []
        for _ in range(params["population_size"]):
            individual = {}
            for var, (min_val, max_val) in variable_ranges.items():
                individual[var] = random.uniform(min_val, max_val)
            population.append(individual)
        
        # Evolucionar población
        for generation in range(params["generations"]):
            # Evaluar fitness
            fitness_scores = []
            for individual in population:
                fitness = await self._evaluate_fitness(intersection_data, individual)
                fitness_scores.append(fitness)
            
            # Selección
            new_population = []
            
            # Elitismo: mantener mejores individuos
            elite_count = max(1, params["population_size"] // 10)
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
            
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generar resto de población
            while len(new_population) < params["population_size"]:
                # Selección por torneo
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Cruzamiento
                if random.random() < params["crossover_rate"]:
                    child1, child2 = self._crossover(parent1, parent2, variable_ranges)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutación
                if random.random() < params["mutation_rate"]:
                    child1 = self._mutate(child1, variable_ranges)
                if random.random() < params["mutation_rate"]:
                    child2 = self._mutate(child2, variable_ranges)
                
                new_population.extend([child1, child2])
            
            population = new_population[:params["population_size"]]
        
        # Retornar mejor solución
        final_fitness = []
        for individual in population:
            fitness = await self._evaluate_fitness(intersection_data, individual)
            final_fitness.append(fitness)
        
        best_idx = max(range(len(final_fitness)), key=lambda i: final_fitness[i])
        return population[best_idx]
    
    async def _simulated_annealing(self, intersection_data: IntersectionData,
                                 constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Algoritmo de recocido simulado"""
        params = self.optimization_params["simulated_annealing"]
        
        # Solución inicial aleatoria
        current_solution = {
            "cycle_time": random.uniform(60, 180),
            "green_ns_ratio": random.uniform(0.3, 0.7),
            "green_ew_ratio": random.uniform(0.3, 0.7),
            "yellow_time": random.uniform(3, 8)
        }
        
        current_fitness = await self._evaluate_fitness(intersection_data, current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        temperature = params["initial_temperature"]
        
        for iteration in range(params["max_iterations"]):
            if temperature < params["min_temperature"]:
                break
            
            # Generar vecino
            neighbor = self._generate_neighbor(current_solution)
            neighbor_fitness = await self._evaluate_fitness(intersection_data, neighbor)
            
            # Decidir aceptación
            delta = neighbor_fitness - current_fitness
            
            if delta > 0 or random.random() < np.exp(delta / temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                if neighbor_fitness > best_fitness:
                    best_solution = neighbor.copy()
                    best_fitness = neighbor_fitness
            
            # Enfriar
            temperature *= params["cooling_rate"]
        
        return best_solution
    
    async def _particle_swarm_optimization(self, intersection_data: IntersectionData,
                                         constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimización por enjambre de partículas"""
        params = self.optimization_params["particle_swarm"]
        
        # Inicializar partículas
        particles = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        
        for _ in range(params["particles"]):
            particle = {
                "cycle_time": random.uniform(60, 180),
                "green_ns_ratio": random.uniform(0.3, 0.7),
                "green_ew_ratio": random.uniform(0.3, 0.7),
                "yellow_time": random.uniform(3, 8)
            }
            
            velocity = {key: random.uniform(-1, 1) for key in particle.keys()}
            
            particles.append(particle)
            velocities.append(velocity)
            personal_best.append(particle.copy())
            
            fitness = await self._evaluate_fitness(intersection_data, particle)
            personal_best_fitness.append(fitness)
        
        # Encontrar mejor global inicial
        global_best_idx = max(range(len(personal_best_fitness)), 
                            key=lambda i: personal_best_fitness[i])
        global_best = personal_best[global_best_idx].copy()
        
        # Iterar
        for iteration in range(params["iterations"]):
            for i in range(params["particles"]):
                # Actualizar velocidad
                for key in particles[i].keys():
                    r1, r2 = random.random(), random.random()
                    
                    cognitive = params["cognitive"] * r1 * (personal_best[i][key] - particles[i][key])
                    social = params["social"] * r2 * (global_best[key] - particles[i][key])
                    
                    velocities[i][key] = (params["inertia"] * velocities[i][key] + 
                                        cognitive + social)
                
                # Actualizar posición
                for key in particles[i].keys():
                    particles[i][key] += velocities[i][key]
                    
                    # Aplicar límites
                    if key == "cycle_time":
                        particles[i][key] = max(60, min(180, particles[i][key]))
                    elif "ratio" in key:
                        particles[i][key] = max(0.3, min(0.7, particles[i][key]))
                    elif key == "yellow_time":
                        particles[i][key] = max(3, min(8, particles[i][key]))
                
                # Evaluar fitness
                fitness = await self._evaluate_fitness(intersection_data, particles[i])
                
                # Actualizar personal best
                if fitness > personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    # Actualizar global best
                    if fitness > personal_best_fitness[global_best_idx]:
                        global_best = particles[i].copy()
                        global_best_idx = i
        
        return global_best
    
    async def _hill_climbing(self, intersection_data: IntersectionData,
                           constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Algoritmo de hill climbing"""
        params = self.optimization_params["hill_climbing"]
        
        # Solución inicial
        current_solution = {
            "cycle_time": 90,
            "green_ns_ratio": 0.5,
            "green_ew_ratio": 0.5,
            "yellow_time": 4
        }
        
        current_fitness = await self._evaluate_fitness(intersection_data, current_solution)
        
        for iteration in range(params["max_iterations"]):
            improved = False
            
            # Probar mejoras en cada variable
            for key in current_solution.keys():
                for direction in [-1, 1]:
                    new_solution = current_solution.copy()
                    
                    step = params["step_size"] * direction
                    if key == "cycle_time":
                        step *= 5  # Pasos más grandes para cycle time
                    elif key == "yellow_time":
                        step *= 0.5  # Pasos más pequeños para yellow
                    
                    new_solution[key] += step
                    
                    # Aplicar límites
                    if key == "cycle_time":
                        new_solution[key] = max(60, min(180, new_solution[key]))
                    elif "ratio" in key:
                        new_solution[key] = max(0.3, min(0.7, new_solution[key]))
                    elif key == "yellow_time":
                        new_solution[key] = max(3, min(8, new_solution[key]))
                    
                    # Evaluar
                    fitness = await self._evaluate_fitness(intersection_data, new_solution)
                    
                    if fitness > current_fitness + params["tolerance"]:
                        current_solution = new_solution
                        current_fitness = fitness
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return current_solution
    
    async def _webster_optimization(self, intersection_data: IntersectionData,
                                  constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimización usando método de Webster"""
        # Usar utilidades existentes
        optimal_cycle = calculate_optimal_cycle_time(intersection_data.traffic_flows)
        green_splits = calculate_green_splits(intersection_data.traffic_flows, optimal_cycle)
        
        return {
            "cycle_time": optimal_cycle,
            "green_ns_ratio": green_splits["north_south_green"] / optimal_cycle,
            "green_ew_ratio": green_splits["east_west_green"] / optimal_cycle,
            "yellow_time": green_splits["yellow"]
        }
    
    async def _adaptive_optimization(self, intersection_data: IntersectionData,
                                   constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimización adaptativa basada en condiciones actuales"""
        # Analizar condiciones actuales
        total_vehicles = intersection_data.get_total_vehicle_count()
        max_queue = intersection_data.get_max_queue_length()
        
        # Estrategia basada en nivel de congestión
        if total_vehicles < 5:
            # Tráfico ligero - ciclos cortos
            return {
                "cycle_time": 60,
                "green_ns_ratio": 0.45,
                "green_ew_ratio": 0.45,
                "yellow_time": 4
            }
        elif total_vehicles > 20 or max_queue > 10:
            # Tráfico pesado - ciclos largos con optimización por demanda
            return await self._webster_optimization(intersection_data, constraints)
        else:
            # Tráfico moderado - usar algoritmo genético rápido
            params_backup = self.optimization_params["genetic"].copy()
            self.optimization_params["genetic"]["generations"] = 20
            self.optimization_params["genetic"]["population_size"] = 20
            
            result = await self._genetic_algorithm(intersection_data, constraints)
            
            self.optimization_params["genetic"] = params_backup
            return result
    
    async def _evaluate_fitness(self, intersection_data: IntersectionData,
                              solution: Dict[str, Any]) -> float:
        """Evalúa la aptitud de una solución"""
        try:
            # Calcular tiempos de fase a partir de la solución
            cycle_time = solution["cycle_time"]
            yellow_time = solution["yellow_time"]
            all_red_time = 2.0
            
            green_ns = cycle_time * solution["green_ns_ratio"]
            green_ew = cycle_time * solution["green_ew_ratio"]
            
            phase_timings = {
                "north_south_green": green_ns,
                "east_west_green": green_ew,
                "yellow": yellow_time,
                "all_red": all_red_time
            }
            
            # Calcular métricas de rendimiento
            delay = calculate_intersection_delay(
                intersection_data.traffic_flows,
                phase_timings,
                cycle_time
            )
            
            # Penalizar retrasos altos
            if delay > 120:  # Más de 2 minutos es inaceptable
                return -delay
            
            # Fitness basado en múltiples factores
            delay_score = max(0, 100 - delay)  # Menor retraso = mejor score
            
            # Penalizar ciclos muy largos o muy cortos
            cycle_penalty = 0
            if cycle_time < 60 or cycle_time > 150:
                cycle_penalty = 20
            
            # Penalizar splits muy desbalanceados
            balance_penalty = 0
            ratio_diff = abs(solution["green_ns_ratio"] - solution["green_ew_ratio"])
            if ratio_diff > 0.3:
                balance_penalty = 10
            
            fitness = delay_score - cycle_penalty - balance_penalty
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error evaluando fitness: {e}")
            return -1000  # Penalización por error
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float],
                            tournament_size: int = 3) -> Dict[str, Any]:
        """Selección por torneo"""
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        
        winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any],
                  variable_ranges: Dict[str, Tuple[float, float]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Cruzamiento uniforme"""
        child1, child2 = {}, {}
        
        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any],
               variable_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Mutación gaussiana"""
        mutated = individual.copy()
        
        for key, (min_val, max_val) in variable_ranges.items():
            if random.random() < 0.3:  # 30% probabilidad de mutación por variable
                std_dev = (max_val - min_val) * 0.1  # 10% del rango
                mutation = random.gauss(0, std_dev)
                mutated[key] = max(min_val, min(max_val, individual[key] + mutation))
        
        return mutated
    
    def _generate_neighbor(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Genera solución vecina para simulated annealing"""
        neighbor = solution.copy()
        
        # Modificar una variable aleatoria
        key = random.choice(list(solution.keys()))
        
        if key == "cycle_time":
            neighbor[key] += random.gauss(0, 10)
            neighbor[key] = max(60, min(180, neighbor[key]))
        elif "ratio" in key:
            neighbor[key] += random.gauss(0, 0.05)
            neighbor[key] = max(0.3, min(0.7, neighbor[key]))
        elif key == "yellow_time":
            neighbor[key] += random.gauss(0, 0.5)
            neighbor[key] = max(3, min(8, neighbor[key]))
        
        return neighbor
    
    async def _calculate_intersection_metrics(self, intersection_data: IntersectionData) -> Dict[str, float]:
        """Calcula métricas de rendimiento de una intersección"""
        # Configuración actual simplificada
        current_config = {
            "north_south_green": 30,
            "east_west_green": 25,
            "yellow": 4,
            "all_red": 2
        }
        
        cycle_time = sum(current_config.values())
        
        delay = calculate_intersection_delay(
            intersection_data.traffic_flows,
            current_config,
            cycle_time
        )
        
        los = calculate_level_of_service(delay)
        
        return {
            "average_delay": delay,
            "level_of_service": ord(los) - ord('A'),  # A=0, B=1, etc.
            "cycle_time": cycle_time,
            "total_vehicles": intersection_data.get_total_vehicle_count(),
            "max_queue": intersection_data.get_max_queue_length()
        }
    
    async def _apply_optimization_config(self, intersection_data: IntersectionData,
                                       config: Dict[str, Any]):
        """Aplica configuración optimizada a la intersección"""
        # En una implementación real, esto actualizaría los semáforos reales
        # Por ahora solo registramos la configuración
        
        self.logger.debug(f"Aplicando configuración optimizada: {config}")
        
        # Simular aplicación de configuración
        cycle_time = config["cycle_time"]
        green_ns = cycle_time * config["green_ns_ratio"]
        green_ew = cycle_time * config["green_ew_ratio"]
        
        intersection_data.last_optimization = datetime.now()
    
    def _calculate_improvement(self, before: Dict[str, float], 
                             after: Dict[str, float]) -> float:
        """Calcula porcentaje de mejora"""
        if "average_delay" in before and "average_delay" in after:
            before_delay = before["average_delay"]
            after_delay = after["average_delay"]
            
            if before_delay > 0:
                improvement = ((before_delay - after_delay) / before_delay) * 100
                return max(-100, min(100, improvement))  # Limitar entre -100% y 100%
        
        return 0.0
    
    async def _global_network_optimization(self, intersections: Dict[str, IntersectionData],
                                         individual_results: List[OptimizationResult]):
        """Optimización global de la red completa"""
        # Identificar corredores principales
        corridors = self._identify_corridors(intersections)
        
        # Optimizar coordinación entre intersecciones
        for corridor in corridors:
            await self._optimize_corridor_coordination(corridor, intersections)
    
    def _identify_corridors(self, intersections: Dict[str, IntersectionData]) -> List[List[str]]:
        """Identifica corredores principales en la red"""
        # Implementación simplificada
        # En la práctica, usaríamos análisis geoespacial
        
        corridors = []
        processed = set()
        
        for intersection_id in intersections:
            if intersection_id not in processed:
                corridor = [intersection_id]
                processed.add(intersection_id)
                
                # Simular búsqueda de intersecciones conectadas
                # En implementación real, usaríamos el grafo de calles
                
                if len(corridor) >= 2:
                    corridors.append(corridor)
        
        return corridors
    
    async def _optimize_corridor_coordination(self, corridor: List[str],
                                            intersections: Dict[str, IntersectionData]):
        """Optimiza la coordinación de un corredor"""
        if len(corridor) < 2:
            return
        
        self.logger.info(f"Optimizando coordinación de corredor: {corridor}")
        
        # Calcular offsets óptimos para onda verde
        # Implementación simplificada
        
        base_offset = 0
        for i, intersection_id in enumerate(corridor):
            if intersection_id in intersections:
                # Aplicar offset calculado
                # En implementación real, esto modificaría los timings
                pass
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de optimizaciones realizadas"""
        if not self.optimization_history:
            return {"total_optimizations": 0}
        
        total_optimizations = len(self.optimization_history)
        avg_improvement = sum(r.improvement_percent for r in self.optimization_history) / total_optimizations
        
        algorithm_usage = {}
        for result in self.optimization_history:
            alg = result.algorithm
            algorithm_usage[alg] = algorithm_usage.get(alg, 0) + 1
        
        return {
            "total_optimizations": total_optimizations,
            "average_improvement": avg_improvement,
            "algorithm_usage": algorithm_usage,
            "last_optimization": self.optimization_history[-1].timestamp.isoformat(),
            "best_improvement": max(r.improvement_percent for r in self.optimization_history)
        }
