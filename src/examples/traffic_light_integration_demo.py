"""
Ejemplo de Integración del Sistema de Semáforos
Demuestra cómo usar el sistema modular de semáforos con el entorno existente
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Añadir path del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from multiagent.environment import Environment
from multiagent.traffic_lights import (
    TrafficLightIntegration, 
    initialize_traffic_light_system,
    get_traffic_light_data_for_vehicle
)


async def demo_traffic_light_integration():
    """
    Demostración completa de la integración del sistema de semáforos
    """
    print("=== Demo: Sistema de Semáforos Modular ===\n")
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TrafficLightDemo")
    
    try:
        # 1. Crear entorno básico (usando el sistema existente)
        print("1. Creando entorno multi-agente...")
        
        # Simular grafo de calles simple para la demo
        import networkx as nx
        import random
        
        street_graph = nx.Graph()
        
        # Crear nodos con coordenadas
        nodes = [
            (1, {"lat": 23.1136, "lon": -82.3666}),
            (2, {"lat": 23.1146, "lon": -82.3676}),
            (3, {"lat": 23.1156, "lon": -82.3686}),
            (4, {"lat": 23.1166, "lon": -82.3696}),
            (5, {"lat": 23.1136, "lon": -82.3676}),
            (6, {"lat": 23.1146, "lon": -82.3686}),
            (7, {"lat": 23.1156, "lon": -82.3696}),
            (8, {"lat": 23.1166, "lon": -82.3706})
        ]
        
        for node_id, attrs in nodes:
            street_graph.add_node(node_id, **attrs)
        
        # Crear conexiones
        edges = [(1, 2), (2, 3), (3, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 6), (6, 7), (7, 8)]
        for edge in edges:
            street_graph.add_edge(edge[0], edge[1], weight=random.uniform(0.5, 2.0))
        
        print(f"   Grafo creado: {len(street_graph.nodes)} nodos, {len(street_graph.edges)} aristas")
        
        # Crear environment
        environment = Environment(street_graph, num_vehicles=10)
        print("   Environment inicializado")
        
        # 2. Inicializar sistema de semáforos integrado
        print("\n2. Inicializando sistema de semáforos modular...")
        
        traffic_integration = await initialize_traffic_light_system(environment)
        
        print("   ✅ Sistema de semáforos integrado")
        print(f"   📊 Semáforos migrados: {len(traffic_integration.legacy_to_new_mapping)}")
        
        # 3. Mostrar estado inicial
        print("\n3. Estado inicial del sistema:")
        
        system_metrics = traffic_integration.get_system_metrics()
        print(f"   🚦 Semáforos operativos: {system_metrics['controller_status']['operational_lights']}")
        print(f"   🎯 Eficiencia promedio: {system_metrics['performance']['average_efficiency']:.2f}")
        print(f"   🔧 Optimización habilitada: {system_metrics['controller_status']['optimization_enabled']}")
        
        # 4. Simular operación durante un período
        print("\n4. Iniciando simulación de operación...")
        
        simulation_duration = 30  # segundos
        step_interval = 2  # segundos
        steps = simulation_duration // step_interval
        
        for step in range(steps):
            print(f"\n   --- Paso {step + 1}/{steps} ---")
            
            # Actualizar environment
            environment.update_state()
            
            # Obtener estados de semáforos
            traffic_states = traffic_integration.get_traffic_light_status()
            
            # Mostrar algunos estados
            for node_id, state in list(traffic_states.items())[:3]:  # Primeros 3
                print(f"   🚦 Nodo {node_id}: {state['state'].upper()} "
                      f"({state['time_in_phase']:.1f}s, eficiencia: {state['efficiency']:.2f})")
            
            # Simular vehículo de emergencia ocasionalmente
            if step == 5:  # En el paso 5
                print("\n   🚨 SIMULANDO VEHÍCULO DE EMERGENCIA")
                
                emergency_vehicle = {
                    "lat": 23.1146,
                    "lon": -82.3676,
                    "speed": 60,
                    "emergency": True
                }
                
                success = await traffic_integration.handle_emergency_vehicle(emergency_vehicle)
                if success:
                    print("   ✅ Vehículo de emergencia procesado - preempción activada")
                else:
                    print("   ❌ No se pudo procesar vehículo de emergencia")
            
            # Simular modificación manual ocasionalmente
            if step == 10:  # En el paso 10
                print("\n   🔧 SIMULANDO MODIFICACIÓN MANUAL")
                
                # Modificar primer semáforo a verde por 30 segundos
                first_node = list(traffic_states.keys())[0]
                success = await traffic_integration.modify_traffic_light_external(
                    first_node, "green", 30
                )
                
                if success:
                    print(f"   ✅ Semáforo nodo {first_node} modificado manualmente")
                else:
                    print(f"   ❌ No se pudo modificar semáforo nodo {first_node}")
            
            # Mostrar métricas cada 5 pasos
            if (step + 1) % 5 == 0:
                current_metrics = traffic_integration.get_system_metrics()
                optimizations = current_metrics['optimization_summary'].get('total_optimizations', 0)
                print(f"   📈 Optimizaciones ejecutadas: {optimizations}")
            
            await asyncio.sleep(step_interval)
        
        # 5. Ejecutar optimización manual
        print("\n5. Ejecutando optimización manual de la red...")
        
        # Forzar optimización
        await traffic_integration._run_network_optimization()
        
        # Mostrar resultados
        final_metrics = traffic_integration.get_system_metrics()
        optimization_summary = final_metrics['optimization_summary']
        
        print(f"   ✅ Optimización completada")
        print(f"   📊 Total optimizaciones: {optimization_summary.get('total_optimizations', 0)}")
        print(f"   📈 Mejor mejora obtenida: {optimization_summary.get('best_improvement', 0):.1f}%")
        
        if optimization_summary.get('total_optimizations', 0) > 0:
            print(f"   ⭐ Mejora promedio: {optimization_summary.get('average_improvement', 0):.1f}%")
        
        # 6. Demostrar integración con vehículos
        print("\n6. Demostrando integración con vehículos...")
        
        if environment.civilian_vehicles:
            # Tomar primer vehículo como ejemplo
            sample_vehicle = environment.civilian_vehicles[0]
            
            # Obtener semáforos visibles para este vehículo
            visible_lights = get_traffic_light_data_for_vehicle(
                traffic_integration, sample_vehicle
            )
            
            print(f"   🚗 Vehículo {sample_vehicle.agent_id}:")
            print(f"      Posición: ({sample_vehicle.lat:.6f}, {sample_vehicle.lon:.6f})")
            print(f"      Semáforos visibles: {len(visible_lights)}")
            
            for light_id, light_data in visible_lights.items():
                print(f"      🚦 {light_id}: {light_data['state'].upper()} "
                      f"(distancia: {light_data['distance']:.0f}m)")
        
        # 7. Mostrar resumen final
        print("\n7. Resumen final del sistema:")
        
        final_status = traffic_integration.get_system_metrics()
        
        print("   📋 Estado del sistema:")
        print(f"      - Sistema integrado: ✅")
        print(f"      - Semáforos operativos: {final_status['controller_status']['operational_lights']}")
        print(f"      - Coordinación habilitada: {final_status['controller_status']['coordination_enabled']}")
        print(f"      - Corredores verdes: {final_status['controller_status']['green_corridors']}")
        print(f"      - Eficiencia promedio: {final_status['performance']['average_efficiency']:.2f}")
        
        print("\n   🎯 Beneficios del sistema modular:")
        print("      ✅ Lógica inteligente adaptativa")
        print("      ✅ Optimización automática")
        print("      ✅ Manejo de emergencias")
        print("      ✅ Coordinación entre semáforos")
        print("      ✅ Métricas de rendimiento")
        print("      ✅ Integración sin conflictos")
        
        # 8. Limpieza
        print("\n8. Cerrando sistema...")
        await traffic_integration.shutdown()
        print("   ✅ Sistema cerrado correctamente")
        
    except Exception as e:
        logger.error(f"Error en demo: {e}")
        raise
    
    print("\n=== Demo completada exitosamente ===")


async def demo_optimization_algorithms():
    """
    Demuestra los diferentes algoritmos de optimización disponibles
    """
    print("\n=== Demo: Algoritmos de Optimización ===\n")
    
    from multiagent.traffic_lights.traffic_light_optimization import TrafficLightOptimizer
    from multiagent.traffic_lights.traffic_light_models import IntersectionData, TrafficFlow, TrafficDirection
    
    # Crear optimizador
    optimizer = TrafficLightOptimizer()
    
    # Crear intersección de prueba
    intersection = IntersectionData(
        intersection_id="test_intersection",
        node_id=999,
        latitude=23.1136,
        longitude=-82.3666
    )
    
    # Añadir flujos de tráfico simulados
    intersection.traffic_flows = {
        TrafficDirection.NORTH: TrafficFlow(
            direction=TrafficDirection.NORTH,
            vehicle_count=15,
            queue_length=5,
            average_speed=25.0
        ),
        TrafficDirection.SOUTH: TrafficFlow(
            direction=TrafficDirection.SOUTH,
            vehicle_count=12,
            queue_length=3,
            average_speed=30.0
        ),
        TrafficDirection.EAST: TrafficFlow(
            direction=TrafficDirection.EAST,
            vehicle_count=8,
            queue_length=2,
            average_speed=35.0
        ),
        TrafficDirection.WEST: TrafficFlow(
            direction=TrafficDirection.WEST,
            vehicle_count=10,
            queue_length=4,
            average_speed=28.0
        )
    }
    
    # Probar diferentes algoritmos
    algorithms = ["webster", "adaptive", "hill_climbing"]
    
    for algorithm in algorithms:
        print(f"Probando algoritmo: {algorithm}")
        
        try:
            result = await optimizer.optimize_intersection(intersection, algorithm)
            
            print(f"   ✅ Optimización completada")
            print(f"   📈 Mejora: {result.improvement_percent:.1f}%")
            print(f"   ⏱️  Tiempo: {result.execution_time:.2f}s")
            print(f"   ⚙️  Parámetros: {result.parameters_used}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    # Mostrar resumen de optimizaciones
    summary = optimizer.get_optimization_summary()
    print("📊 Resumen de optimizaciones:")
    print(f"   Total: {summary['total_optimizations']}")
    print(f"   Mejora promedio: {summary['average_improvement']:.1f}%")
    print(f"   Mejor mejora: {summary['best_improvement']:.1f}%")


if __name__ == "__main__":
    print("Iniciando demo del Sistema de Semáforos Modular...\n")
    
    # Ejecutar demo principal
    asyncio.run(demo_traffic_light_integration())
    
    # Ejecutar demo de optimización
    asyncio.run(demo_optimization_algorithms())
    
    print("\n¡Todas las demos completadas exitosamente! 🎉")
