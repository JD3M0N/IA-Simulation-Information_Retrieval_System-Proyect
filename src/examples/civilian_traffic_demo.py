"""
Ejemplo de integración del CivilianTrafficAgent con Environment
Demuestra cómo usar el agente de tráfico civil en la simulación
"""

import asyncio
import networkx as nx
import random
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Importar las clases necesarias
from multiagent.environment import Environment, WeatherCondition, TrafficEventType
from multiagent.civilian_traffic import CivilianTrafficAgent, CivilianBehavior
from multi_agent.communication import communication_manager

async def demo_civilian_traffic_simulation():
    """
    Demostración de la simulación con agentes de tráfico civil
    """
    print("=== Iniciando Simulación de Tráfico Civil ===")
    
    # 1. Crear grafo de calles simple
    street_graph = nx.Graph()
    
    # Agregar nodos (intersecciones)
    nodes = [(40.7128, -74.0060), (40.7138, -74.0070), (40.7148, -74.0080), 
             (40.7158, -74.0090), (40.7168, -74.0100)]
    
    for i, (lat, lon) in enumerate(nodes):
        street_graph.add_node(i, lat=lat, lon=lon)
    
    # Agregar aristas (calles)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4)]
    for edge in edges:
        street_graph.add_edge(edge[0], edge[1], weight=random.uniform(0.5, 2.0))
    
    print(f"Grafo creado con {len(street_graph.nodes)} nodos y {len(street_graph.edges)} aristas")
    
    # 2. Crear Environment
    environment = Environment(street_graph, num_vehicles=5)
    
    # Registrar el environment en el sistema de comunicación
    await communication_manager.register_agent_id("environment", "environment")
    
    print("Environment inicializado")
    
    # 3. Crear agentes de tráfico civil
    civilian_agents = []
    behaviors = list(CivilianBehavior)
    
    for i in range(5):
        # Posición inicial aleatoria
        initial_pos = (40.7128 + random.uniform(-0.01, 0.01), 
                      -74.0060 + random.uniform(-0.01, 0.01))
        
        # Comportamiento aleatorio
        behavior = random.choice(behaviors)
        
        # Crear agente
        agent = CivilianTrafficAgent(
            agent_id=f"civilian_{i}",
            initial_position=initial_pos,
            behavior=behavior
        )
        
        # Asignar nodo inicial y destino
        agent.current_node = random.choice(list(street_graph.nodes))
        agent.set_destination_and_route(
            target_node=random.choice(list(street_graph.nodes)),
            route=[agent.current_node, random.choice(list(street_graph.nodes))]
        )
        
        civilian_agents.append(agent)
        
        # Registrar en el sistema de comunicación
        await communication_manager.register_agent(agent)
        
        print(f"Agente creado: {agent.agent_id} - Comportamiento: {behavior.value}")
    
    # 4. Iniciar comunicación
    await communication_manager.start()
    print("Sistema de comunicación iniciado")
    
    # 5. Ejecutar simulación
    print("\n=== Iniciando Ciclo de Simulación ===")
    
    try:
        for step in range(10):  # 10 pasos de simulación
            print(f"\n--- Paso {step + 1} ---")
            
            # Actualizar Environment
            environment.update_state()
            environment_state = environment.get_perception_for_agent("civilian")
            
            # Simular eventos ocasionales
            if random.random() < 0.3:  # 30% probabilidad
                environment.generate_traffic_event(
                    event_type=random.choice(list(TrafficEventType)),
                    location=(40.7128 + random.uniform(-0.01, 0.01), 
                             -74.0060 + random.uniform(-0.01, 0.01)),
                    severity=random.randint(1, 5)
                )
            
            # Cambiar clima ocasionalmente
            if random.random() < 0.2:  # 20% probabilidad
                new_weather = random.choice(list(WeatherCondition))
                environment.weather_state.condition = new_weather
                print(f"Cambio climático: {new_weather.value}")
            
            # Procesar cada agente
            for agent in civilian_agents:
                try:
                    # Ciclo Percepción-Decisión-Acción
                    perception = await agent.perceive(environment_state)
                    decision = await agent.decide(perception)
                    success = await agent.act(decision)
                    
                    # Procesar mensajes
                    await agent.process_messages()
                    
                    # Mostrar estado del agente
                    status = agent.get_vehicle_status()
                    print(f"  {agent.agent_id}: Velocidad={status['current_speed']:.1f}, "
                          f"Estado={status['movement_state']}, "
                          f"Nodo={status['current_node']}")
                    
                except Exception as e:
                    print(f"Error procesando agente {agent.agent_id}: {e}")
            
            # Actualizar métricas del environment
            environment.update_metrics()
            
            # Pequeña pausa
            await asyncio.sleep(1)
        
        # 6. Mostrar estadísticas finales
        print("\n=== Estadísticas Finales ===")
        
        # Métricas del environment
        env_metrics = environment.get_metrics()
        print(f"Eventos de tráfico generados: {env_metrics['total_events']}")
        print(f"Vehículos activos: {env_metrics['active_vehicles']}")
        print(f"Tiempo de simulación: {env_metrics['simulation_time']:.2f}s")
        
        # Métricas de los agentes
        for agent in civilian_agents:
            metrics = agent.get_vehicle_status()["metrics"]
            print(f"\n{agent.agent_id}:")
            print(f"  - Distancia recorrida: {metrics['distance_traveled']:.2f} km")
            print(f"  - Paradas realizadas: {metrics['stops_count']}")
            print(f"  - Cambios de ruta: {metrics['route_changes']}")
            print(f"  - Respuestas a emergencias: {metrics['emergency_responses']}")
    
    except KeyboardInterrupt:
        print("\nSimulación interrumpida por el usuario")
    
    except Exception as e:
        print(f"Error en la simulación: {e}")
    
    finally:
        # 7. Limpieza
        print("\n=== Finalizando Simulación ===")
        
        # Detener agentes
        for agent in civilian_agents:
            await agent.stop()
        
        # Detener comunicación
        await communication_manager.stop()
        
        print("Simulación finalizada")

async def demo_emergency_scenario():
    """
    Demonstración de escenario de emergencia
    """
    print("\n=== Escenario de Emergencia ===")
    
    # Crear grafo simple
    street_graph = nx.Graph()
    street_graph.add_nodes_from(range(5))
    street_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    
    # Crear environment y agente
    environment = Environment(street_graph, num_vehicles=3)
    
    agent = CivilianTrafficAgent(
        agent_id="civilian_emergency_test",
        initial_position=(40.7128, -74.0060),
        behavior=CivilianBehavior.NORMAL
    )
    
    # Registrar agente
    await communication_manager.register_agent(agent)
    await communication_manager.start()
    
    try:
        # Estado inicial
        print(f"Estado inicial del agente: {agent.movement_state.value}")
        
        # Simular emergencia de accidente
        print("Generando evento de accidente...")
        environment.generate_traffic_event(
            event_type=TrafficEventType.ACCIDENT,
            location=(40.7130, -74.0062),  # Cerca del agente
            severity=5  # Severidad máxima
        )
        
        # Proceso de percepción y respuesta
        environment_state = environment.get_perception_for_agent("civilian")
        perception = await agent.perceive(environment_state)
        decision = await agent.decide(perception)
        await agent.act(decision)
        
        print(f"Estado después del accidente: {agent.movement_state.value}")
        print(f"Velocidad: {agent.current_speed:.1f} km/h")
        
        # Simular vehículo de emergencia
        print("Simulando mensaje de vehículo de emergencia...")
        
        await communication_manager.send_to_topic(
            "emergency",
            "emergency_vehicle",
            "emergency",
            {
                "emergency_type": "emergency_vehicle",
                "location": agent.position,
                "priority": "high"
            }
        )
        
        # Procesar mensaje de emergencia
        await agent.process_messages()
        
        print(f"Estado final: {agent.movement_state.value}")
        print(f"Respuestas a emergencias: {agent.emergency_responses}")
        
    finally:
        await agent.stop()
        await communication_manager.stop()

async def demo_weather_adaptation():
    """
    Demonstración de adaptación climática
    """
    print("\n=== Adaptación Climática ===")
    
    # Crear grafo y environment
    street_graph = nx.Graph()
    street_graph.add_nodes_from(range(3))
    street_graph.add_edges_from([(0, 1), (1, 2)])
    
    environment = Environment(street_graph)
    
    # Crear agente cauteloso
    agent = CivilianTrafficAgent(
        agent_id="weather_test_agent",
        initial_position=(40.7128, -74.0060),
        behavior=CivilianBehavior.CAUTIOUS
    )
    
    await communication_manager.register_agent(agent)
    await communication_manager.start()
    
    try:
        print(f"Velocidad inicial: {agent.current_speed:.1f} km/h")
        
        # Probar diferentes condiciones climáticas
        weather_conditions = [
            WeatherCondition.CLEAR,
            WeatherCondition.LIGHT_RAIN,
            WeatherCondition.HEAVY_RAIN,
            WeatherCondition.FOG
        ]
        
        for weather in weather_conditions:
            print(f"\nCambiando clima a: {weather.value}")
            environment.weather_state.condition = weather
            
            if weather == WeatherCondition.HEAVY_RAIN:
                environment.weather_state.precipitation = 15.0
                environment.weather_state.visibility = 3.0
            elif weather == WeatherCondition.LIGHT_RAIN:
                environment.weather_state.precipitation = 5.0
                environment.weather_state.visibility = 7.0
            elif weather == WeatherCondition.FOG:
                environment.weather_state.precipitation = 0.0
                environment.weather_state.visibility = 2.0
            else:  # CLEAR
                environment.weather_state.precipitation = 0.0
                environment.weather_state.visibility = 10.0
            
            # Procesar adaptación
            environment_state = environment.get_perception_for_agent("civilian")
            perception = await agent.perceive(environment_state)
            decision = await agent.decide(perception)
            await agent.act(decision)
            
            print(f"  Velocidad adaptada: {agent.current_speed:.1f} km/h")
            print(f"  Adaptaciones climáticas: {agent.weather_adaptations}")
    
    finally:
        await agent.stop()
        await communication_manager.stop()

if __name__ == "__main__":
    print("Ejemplos de Simulación con CivilianTrafficAgent")
    print("=" * 50)
    
    # Ejecutar demostraciones
    asyncio.run(demo_civilian_traffic_simulation())
    asyncio.run(demo_emergency_scenario())
    asyncio.run(demo_weather_adaptation())
    
    print("\nTodas las demostraciones completadas.")
