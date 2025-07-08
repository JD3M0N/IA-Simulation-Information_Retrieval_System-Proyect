"""
Script de prueba rápida del sistema BDI
Verificación básica de funcionalidad
"""

import sys
import asyncio
import networkx as nx

# Añadir paths
sys.path.append("src")
sys.path.append("src/multiagent")

def test_bdi_core():
    """Prueba componentes básicos BDI"""
    print("🧪 Probando componentes BDI básicos...")
    
    try:
        from bdi_core import BDIAgent, Belief, Desire, BeliefType, DesireType
        
        # Crear agente BDI básico
        agent = BDIAgent("test_agent")
        print(f"✅ Agente BDI creado: {agent.agent_id}")
        
        # Crear creencia
        belief = Belief(
            belief_id="test_belief",
            belief_type=BeliefType.VEHICLE_INFO,
            content={"speed": 50, "fuel": 80}
        )
        agent.belief_base.add_belief(belief)
        print(f"✅ Creencia añadida: {belief.belief_id}")
        
        # Crear deseo
        desire = Desire(
            desire_id="test_desire",
            desire_type=DesireType.SAVE_FUEL,
            priority=0.8
        )
        agent.desire_set.add_desire(desire)
        print(f"✅ Deseo añadido: {desire.desire_id}")
        
        # Verificar estado
        status = agent.get_status()
        print(f"✅ Estado del agente: {status['beliefs_count']} creencias, {status['desires_count']} deseos")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba BDI: {e}")
        return False

def test_delivery_truck():
    """Prueba camión de reparto BDI"""
    print("\n🚛 Probando camión de reparto BDI...")
    
    try:
        from delivery_truck_bdi import DeliveryTruckBDI
        
        # Crear camión
        truck = DeliveryTruckBDI(
            agent_id="test_truck",
            initial_node=1,
            capacity=1000
        )
        print(f"✅ Camión BDI creado: {truck.agent_id}")
        
        # Verificar propiedades
        print(f"   📍 Nodo inicial: {truck.current_node}")
        print(f"   📦 Capacidad: {truck.capacity} kg")
        print(f"   ⛽ Combustible: {truck.fuel_level}%")
        
        # Verificar estado BDI
        status = truck.get_status()
        print(f"✅ Estado BDI: {status['beliefs_count']} creencias, {status['desires_count']} deseos")
        
        # Verificar estado de entrega
        delivery_status = truck.get_delivery_status()
        print(f"✅ Estado de entrega obtenido correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de camión: {e}")
        return False

def test_communication():
    """Prueba sistema de comunicación"""
    print("\n📡 Probando sistema de comunicación...")
    
    try:
        from communication_system import communication_manager, MessageType
        from delivery_truck_bdi import DeliveryTruckBDI
        
        # Crear dos camiones
        truck1 = DeliveryTruckBDI("truck1", 1, 1000)
        truck2 = DeliveryTruckBDI("truck2", 2, 1000)
        
        # Registrar en comunicación
        communication_manager.register_agent(truck1)
        communication_manager.register_agent(truck2)
        print("✅ Agentes registrados en comunicación")
        
        # Enviar mensaje
        success = communication_manager.send_message(
            "truck1",
            "truck2", 
            MessageType.ROUTE_INFO,
            {"route": [1, 2, 3], "test": True}
        )
        print(f"✅ Mensaje enviado: {success}")
        
        # Obtener estadísticas
        stats = communication_manager.get_communication_stats()
        print(f"✅ Estadísticas: {stats['active_agents']} agentes activos")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de comunicación: {e}")
        return False

def test_environment_integration():
    """Prueba integración con entorno"""
    print("\n🏗️ Probando integración con entorno...")
    
    try:
        from environment import Environment
        
        # Crear grafo simple
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        
        # Añadir coordenadas
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['lat'] = 23.1 + i * 0.001
            G.nodes[node]['lon'] = -82.3 + i * 0.001
        
        # Crear entorno
        env = Environment(G, num_vehicles=2)
        print(f"✅ Entorno creado con {len(env.street_graph.nodes)} nodos")
        
        # Añadir camión BDI
        success = env.add_bdi_delivery_truck(
            truck_id="test_truck",
            start_node=1,
            capacity=1000,
            delivery_locations=[2, 3]
        )
        print(f"✅ Camión BDI añadido: {success}")
        
        # Verificar estado
        status = env.get_bdi_trucks_status()
        print(f"✅ Estado obtenido: {len(status)} camiones")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de entorno: {e}")
        return False

async def test_bdi_cycle():
    """Prueba ciclo BDI completo"""
    print("\n🔄 Probando ciclo BDI completo...")
    
    try:
        from delivery_truck_bdi import DeliveryTruckBDI
        import networkx as nx
        
        # Crear grafo simple
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['lat'] = 23.1 + i * 0.001
            G.nodes[node]['lon'] = -82.3 + i * 0.001
            
        # Crear camión
        truck = DeliveryTruckBDI("test_truck", 1, 1000)
        truck.street_graph = G
        
        # Asignar entregas
        truck.assign_delivery_route([2, 3])
        
        # Datos del entorno simulados
        env_data = {
            "weather": {"condition": "clear", "temperature": 25},
            "road_network": {"congestion": {}, "traffic_lights": {}},
            "vehicles": {}
        }
        
        # Ejecutar ciclo BDI
        await truck.bdi_cycle(env_data)
        print("✅ Ciclo BDI ejecutado correctamente")
        
        # Verificar métricas
        status = truck.get_status()
        print(f"✅ Ciclos ejecutados: {status['cycle_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de ciclo BDI: {e}")
        return False

async def main():
    """Función principal de pruebas"""
    print("🚛 SISTEMA BDI - PRUEBAS RÁPIDAS")
    print("=" * 50)
    
    tests = [
        ("BDI Core", test_bdi_core),
        ("Delivery Truck", test_delivery_truck),
        ("Communication", test_communication),
        ("Environment Integration", test_environment_integration),
        ("BDI Cycle", test_bdi_cycle)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTADOS: {passed}/{total} pruebas exitosas")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas exitosas! Sistema BDI listo para usar.")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisar implementación.")
    
    print("\n💡 Para ejecutar la demostración completa:")
    print("   python src/multiagent/bdi_demo.py")

if __name__ == "__main__":
    asyncio.run(main())
