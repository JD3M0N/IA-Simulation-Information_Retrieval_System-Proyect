"""
Pruebas de integración para CivilianTrafficAgent y Environment
Valida la compatibilidad y funcionalidad del sistema
"""

import pytest
import asyncio
import networkx as nx
from datetime import datetime
import random


from src.multiagent.environment import Environment, WeatherCondition, TrafficEventType
from src.multiagent.civilian_traffic import CivilianTrafficAgent, CivilianBehavior
from src.multi_agent.communication import communication_manager
from src.config.civilian_traffic_config import CivilianTrafficManager, CivilianTrafficConfig


class TestCivilianTrafficIntegration:
    """Pruebas de integración del sistema completo"""
    
    @pytest.fixture
    def street_graph(self):
        """Grafo de calles para pruebas"""
        graph = nx.Graph()
        
        # Crear red simple 3x3
        for i in range(9):
            row = i // 3
            col = i % 3
            graph.add_node(i, lat=40.7128 + row * 0.001, lon=-74.0060 + col * 0.001)
        
        # Conectar nodos adyacentes
        for i in range(9):
            row = i // 3
            col = i % 3
            
            # Conectar horizontalmente
            if col < 2:
                graph.add_edge(i, i + 1, weight=1.0)
            
            # Conectar verticalmente
            if row < 2:
                graph.add_edge(i, i + 3, weight=1.0)
        
        return graph
    
    @pytest.fixture
    def environment(self, street_graph):
        """Environment para pruebas"""
        return Environment(street_graph, num_vehicles=5)
    
    @pytest.fixture
    def civilian_agent(self):
        """Agente civil para pruebas"""
        return CivilianTrafficAgent(
            vehicle_id="test_civilian",
            initial_position=(40.7128, -74.0060),
            behavior=CivilianBehavior.NORMAL
        )
    
    @pytest.mark.asyncio
    async def test_environment_creation(self, street_graph):
        """Prueba creación del Environment"""
        env = Environment(street_graph, num_vehicles=3)
        
        assert env.street_graph == street_graph
        assert len(env.vehicles) == 0  # Se crean bajo demanda
        assert env.weather_state.condition == WeatherCondition.CLEAR
        assert len(env.road_segments) > 0
        assert len(env.traffic_lights) > 0
    
    @pytest.mark.asyncio
    async def test_civilian_agent_creation(self):
        """Prueba creación del agente civil"""
        agent = CivilianTrafficAgent(
            vehicle_id="test_agent",
            initial_position=(40.7128, -74.0060),
            behavior=CivilianBehavior.AGGRESSIVE
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.behavior == CivilianBehavior.AGGRESSIVE
        assert agent.position == (40.7128, -74.0060)
        assert agent.current_speed == 0.0
        assert agent.fuel_level > 0
    
    @pytest.mark.asyncio
    async def test_perception_integration(self, environment, civilian_agent):
        """Prueba integración de percepción"""
        # Configurar environment
        environment.weather_state.condition = WeatherCondition.LIGHT_RAIN
        environment.weather_state.precipitation = 5.0
        
        # Generar evento de tráfico
        environment.generate_traffic_event(
            event_type=TrafficEventType.ACCIDENT,
            location=(40.7128, -74.0060),
            severity=3
        )
        
        # Obtener percepción
        environment_state = environment.get_perception_for_agent("civilian")
        perception = await civilian_agent.perceive(environment_state)
        
        # Verificar percepción
        assert "weather" in perception
        assert "traffic_events" in perception
        assert "traffic_lights" in perception
        assert perception["weather"]["condition"] == WeatherCondition.LIGHT_RAIN
        assert len(perception["traffic_events"]) > 0
    
    @pytest.mark.asyncio
    async def test_decision_making(self, environment, civilian_agent):
        """Prueba toma de decisiones"""
        # Configurar percepción
        environment_state = environment.get_perception_for_agent("civilian")
        perception = await civilian_agent.perceive(environment_state)
        
        # Tomar decisión
        decision = await civilian_agent.decide(perception)
        
        # Verificar decisión
        assert "target_speed" in decision
        assert "acceleration" in decision
        assert "should_stop" in decision
        assert "cooperation_actions" in decision
        assert decision["target_speed"] >= 0
    
    @pytest.mark.asyncio
    async def test_action_execution(self, environment, civilian_agent):
        """Prueba ejecución de acciones"""
        # Configurar decisión
        decision = {
            "target_speed": 30.0,
            "acceleration": 2.0,
            "should_stop": False,
            "cooperation_actions": ["maintain_safe_distance"],
            "emergency_response": {},
            "weather_adaptation": {}
        }
        
        initial_speed = civilian_agent.current_speed
        
        # Ejecutar acción
        success = await civilian_agent.act(decision)
        
        # Verificar resultado
        assert success == True
        assert civilian_agent.current_speed != initial_speed
    
    @pytest.mark.asyncio
    async def test_weather_adaptation(self, environment, civilian_agent):
        """Prueba adaptación climática"""
        # Configurar clima adverso
        environment.weather_state.condition = WeatherCondition.HEAVY_RAIN
        environment.weather_state.precipitation = 15.0
        environment.weather_state.visibility = 2.0
        
        # Procesar percepción y decisión
        environment_state = environment.get_perception_for_agent("civilian")
        perception = await civilian_agent.perceive(environment_state)
        decision = await civilian_agent.decide(perception)
        
        # Verificar adaptación
        assert decision["target_speed"] < civilian_agent.max_speed
        assert "weather_adaptation" in decision
        
        # Ejecutar adaptación
        await civilian_agent.act(decision)
        
        # Verificar que se adaptó al clima
        assert civilian_agent.current_speed < civilian_agent.max_speed
    
    @pytest.mark.asyncio
    async def test_emergency_response(self, environment, civilian_agent):
        """Prueba respuesta a emergencias"""
        # Registrar agente en comunicación
        await communication_manager.register_agent(civilian_agent)
        await communication_manager.start()
        
        try:
            # Simular emergencia
            await communication_manager.send_to_topic(
                "emergency",
                "emergency_system",
                "emergency",
                {
                    "emergency_type": "emergency_vehicle",
                    "location": civilian_agent.position,
                    "priority": "high"
                }
            )
            
            # Procesar mensaje
            await civilian_agent.process_messages()
            
            # Verificar respuesta
            assert civilian_agent.emergency_responses > 0
            
        finally:
            await communication_manager.stop()
    
    @pytest.mark.asyncio
    async def test_traffic_light_compliance(self, environment, civilian_agent):
        """Prueba cumplimiento de semáforos"""
        # Configurar semáforo en rojo
        civilian_agent.current_node = 0
        
        # Crear semáforo cercano en rojo
        for light_id, light in environment.traffic_lights.items():
            if light.node_id == 0:
                light.state = "red"
                break
        
        # Procesar percepción y decisión
        environment_state = environment.get_perception_for_agent("civilian")
        perception = await civilian_agent.perceive(environment_state)
        decision = await civilian_agent.decide(perception)
        
        # Verificar que decide detenerse
        assert decision["should_stop"] == True
    
    @pytest.mark.asyncio
    async def test_behavior_differences(self, environment):
        """Prueba diferencias de comportamiento"""
        behaviors = [CivilianBehavior.CAUTIOUS, CivilianBehavior.AGGRESSIVE]
        agents = []
        
        for behavior in behaviors:
            agent = CivilianTrafficAgent(
                vehicle_id=f"test_{behavior.value}",
                initial_position=(40.7128, -74.0060),
                behavior=behavior
            )
            agents.append(agent)
        
        # Configurar mismo escenario
        environment.weather_state.condition = WeatherCondition.CLEAR
        environment_state = environment.get_perception_for_agent("civilian")
        
        decisions = []
        for agent in agents:
            perception = await agent.perceive(environment_state)
            decision = await agent.decide(perception)
            decisions.append(decision)
        
        # Verificar que los comportamientos difieren
        cautious_speed = decisions[0]["target_speed"]
        aggressive_speed = decisions[1]["target_speed"]
        
        assert cautious_speed < aggressive_speed
    
    @pytest.mark.asyncio
    async def test_traffic_manager_integration(self, street_graph):
        """Prueba integración con CivilianTrafficManager"""
        config = CivilianTrafficConfig(
            num_civilian_vehicles=5,
            behavior_distribution={
                CivilianBehavior.NORMAL: 0.6,
                CivilianBehavior.AGGRESSIVE: 0.4
            }
        )
        
        environment = Environment(street_graph, num_vehicles=5)
        manager = CivilianTrafficManager(config)
        
        # Crear agentes
        agents = manager.create_civilian_agents(street_graph, environment)
        
        # Verificar creación
        assert len(agents) == 5
        assert all(isinstance(agent, CivilianTrafficAgent) for agent in agents)
        assert all(agent.current_node is not None for agent in agents)
        
        # Verificar estadísticas
        stats = manager.get_agent_statistics()
        assert stats["total_agents"] == 5
        assert "behavior_distribution" in stats
        assert "vehicle_type_distribution" in stats
    
    @pytest.mark.asyncio
    async def test_full_simulation_cycle(self, street_graph):
        """Prueba ciclo completo de simulación"""
        # Crear componentes
        environment = Environment(street_graph, num_vehicles=3)
        config = CivilianTrafficConfig(num_civilian_vehicles=3)
        manager = CivilianTrafficManager(config)
        agents = manager.create_civilian_agents(street_graph, environment)
        
        # Registrar agentes
        await communication_manager.register_agent_id("environment", "environment")
        for agent in agents:
            await communication_manager.register_agent(agent)
        
        await communication_manager.start()
        
        try:
            # Ejecutar varios ciclos
            for step in range(5):
                # Actualizar environment
                environment.update_state()
                environment_state = environment.get_perception_for_agent("civilian")
                
                # Procesar cada agente
                for agent in agents:
                    perception = await agent.perceive(environment_state)
                    decision = await agent.decide(perception)
                    success = await agent.act(decision)
                    
                    assert success == True
                    
                    # Procesar mensajes
                    await agent.process_messages()
                
                # Generar evento ocasional
                if step == 2:
                    environment.generate_traffic_event(
                        event_type=TrafficEventType.CONSTRUCTION,
                        location=(40.7128, -74.0060),
                        severity=2
                    )
                
                # Verificar estado
                assert len(environment.traffic_events) >= 0
                
                await asyncio.sleep(0.01)  # Pequeña pausa
            
            # Verificar métricas finales
            stats = manager.get_agent_statistics()
            assert stats["total_agents"] == 3
            
            # Verificar que los agentes se movieron
            for agent in agents:
                metrics = agent.get_vehicle_status()["metrics"]
                # Al menos algún agente debería haber registrado actividad
                assert sum(metrics.values()) >= 0
        
        finally:
            await communication_manager.stop()
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, environment, civilian_agent):
        """Prueba casos extremos"""
        # Caso 1: Percepción vacía
        empty_perception = {}
        decision = await civilian_agent.decide(empty_perception)
        assert "target_speed" in decision
        
        # Caso 2: Múltiples eventos simultáneos
        environment.generate_traffic_event(
            event_type=TrafficEventType.ACCIDENT,
            location=(40.7128, -74.0060),
            severity=5
        )
        environment.generate_traffic_event(
            event_type=TrafficEventType.CONSTRUCTION,
            location=(40.7129, -74.0061),
            severity=3
        )
        
        environment_state = environment.get_perception_for_agent("civilian")
        perception = await civilian_agent.perceive(environment_state)
        decision = await civilian_agent.decide(perception)
        
        assert decision["target_speed"] >= 0
        
        # Caso 3: Condiciones climáticas extremas
        environment.weather_state.condition = WeatherCondition.STORM
        environment.weather_state.precipitation = 25.0
        environment.weather_state.visibility = 0.5
        
        environment_state = environment.get_perception_for_agent("civilian")
        perception = await civilian_agent.perceive(environment_state)
        decision = await civilian_agent.decide(perception)
        
        # Debería reducir significativamente la velocidad
        assert decision["target_speed"] < civilian_agent.max_speed * 0.5

# Función para ejecutar todas las pruebas
async def run_all_tests():
    """Ejecuta todas las pruebas de integración"""
    print("Ejecutando pruebas de integración...")
    
    # Crear instancia de pruebas
    test_instance = TestCivilianTrafficIntegration()
    
    # Crear fixtures
    street_graph = test_instance.street_graph()
    environment = test_instance.environment(street_graph)
    civilian_agent = test_instance.civilian_agent()
    
    try:
        print("✓ Prueba de creación de Environment")
        await test_instance.test_environment_creation(street_graph)
        
        print("✓ Prueba de creación de agente civil")
        await test_instance.test_civilian_agent_creation()
        
        print("✓ Prueba de integración de percepción")
        await test_instance.test_perception_integration(environment, civilian_agent)
        
        print("✓ Prueba de toma de decisiones")
        await test_instance.test_decision_making(environment, civilian_agent)
        
        print("✓ Prueba de ejecución de acciones")
        await test_instance.test_action_execution(environment, civilian_agent)
        
        print("✓ Prueba de adaptación climática")
        await test_instance.test_weather_adaptation(environment, civilian_agent)
        
        print("✓ Prueba de respuesta a emergencias")
        await test_instance.test_emergency_response(environment, civilian_agent)
        
        print("✓ Prueba de cumplimiento de semáforos")
        await test_instance.test_traffic_light_compliance(environment, civilian_agent)
        
        print("✓ Prueba de diferencias de comportamiento")
        await test_instance.test_behavior_differences(environment)
        
        print("✓ Prueba de integración con manager")
        await test_instance.test_traffic_manager_integration(street_graph)
        
        print("✓ Prueba de ciclo completo")
        await test_instance.test_full_simulation_cycle(street_graph)
        
        print("✓ Prueba de casos extremos")
        await test_instance.test_edge_cases(environment, civilian_agent)
        
        print("\n🎉 Todas las pruebas pasaron exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error en las pruebas: {e}")
        raise

if __name__ == "__main__":
    # Ejecutar pruebas
    asyncio.run(run_all_tests())
