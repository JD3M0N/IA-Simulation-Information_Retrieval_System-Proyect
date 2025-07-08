"""
Script de Demostración del Sistema Modular de Semáforos
Ejecuta una simulación completa mostrando las capacidades del sistema
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
import sys
import os

# Añadir rutas del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.multiagent.environment import Environment
from src.multiagent.traffic_lights import (
    initialize_server_traffic_lights,
    get_server_traffic_lights_data, 
    modify_server_traffic_light,
    get_server_traffic_metrics,
    server_traffic_manager
)


async def create_demo_environment():
    """Crea un entorno de demostración"""
    import networkx as nx
    
    print("🏗️ Creando entorno de demostración...")
    
    # Crear grafo de calles
    street_graph = nx.Graph()
    
    # Añadir nodos representando intersecciones importantes
    nodes_data = [
        (101, {"lat": 23.1130, "lon": -82.3660, "name": "Plaza de Armas"}),
        (102, {"lat": 23.1140, "lon": -82.3670, "name": "Capitolio"}),
        (103, {"lat": 23.1150, "lon": -82.3680, "name": "Parque Central"}),
        (104, {"lat": 23.1160, "lon": -82.3690, "name": "Malecón"}),
        (105, {"lat": 23.1130, "lon": -82.3670, "name": "Cathedral"}),
        (106, {"lat": 23.1140, "lon": -82.3680, "name": "Teatro Nacional"}),
        (107, {"lat": 23.1150, "lon": -82.3690, "name": "Hotel Nacional"}),
        (108, {"lat": 23.1160, "lon": -82.3700, "name": "Universidad"})
    ]
    
    for node_id, attrs in nodes_data:
        street_graph.add_node(node_id, **attrs)
    
    # Crear conexiones entre intersecciones
    connections = [
        (101, 102, {"road_type": "primary", "max_speed": 50}),
        (102, 103, {"road_type": "primary", "max_speed": 50}),
        (103, 104, {"road_type": "primary", "max_speed": 60}),
        (101, 105, {"road_type": "secondary", "max_speed": 40}),
        (102, 106, {"road_type": "secondary", "max_speed": 40}),
        (103, 107, {"road_type": "secondary", "max_speed": 40}),
        (104, 108, {"road_type": "secondary", "max_speed": 40}),
        (105, 106, {"road_type": "tertiary", "max_speed": 30}),
        (106, 107, {"road_type": "tertiary", "max_speed": 30}),
        (107, 108, {"road_type": "tertiary", "max_speed": 30})
    ]
    
    for src, dst, attrs in connections:
        street_graph.add_edge(src, dst, **attrs)
    
    print(f"   ✅ Grafo creado: {len(street_graph.nodes)} intersecciones, {len(street_graph.edges)} calles")
    
    # Crear environment
    environment = Environment(street_graph, num_vehicles=15)
    print(f"   ✅ Environment inicializado con {environment.num_vehicles} vehículos")
    
    return environment


async def run_traffic_light_demo():
    """Ejecuta la demostración completa del sistema de semáforos"""
    print("=" * 60)
    print("🚦 DEMOSTRACIÓN SISTEMA MODULAR DE SEMÁFOROS")
    print("=" * 60)
    
    try:
        # 1. Crear entorno de demostración
        environment = await create_demo_environment()
        
        # 2. Inicializar sistema modular de semáforos
        print("\n🔧 Inicializando sistema modular de semáforos...")
        success = await initialize_server_traffic_lights(environment)
        
        if not success:
            print("❌ Error inicializando sistema de semáforos")
            return
        
        print("✅ Sistema de semáforos inicializado correctamente")
        
        # 3. Mostrar estado inicial
        print("\n📊 Estado inicial del sistema:")
        initial_metrics = get_server_traffic_metrics()
        
        if initial_metrics:
            controller_status = initial_metrics.get('controller_status', {})
            performance = initial_metrics.get('performance', {})
            
            print(f"   🚦 Semáforos operativos: {controller_status.get('operational_lights', 0)}")
            print(f"   🎯 Eficiencia promedio: {performance.get('average_efficiency', 0):.2f}")
            print(f"   🌐 Eficiencia de red: {performance.get('network_efficiency', 0):.2f}")
            print(f"   🔧 Optimización activa: {controller_status.get('optimization_enabled', False)}")
        
        # 4. Mostrar datos de semáforos
        print("\n🚦 Datos de semáforos activos:")
        traffic_data = get_server_traffic_lights_data()
        
        for i, traffic_light in enumerate(traffic_data[:5]):  # Mostrar solo los primeros 5
            print(f"   {i+1}. Nodo {traffic_light['node_id']}: "
                  f"Estado {traffic_light['state'].upper()}, "
                  f"Zona {traffic_light['zone']}, "
                  f"Dirección {traffic_light['direction']}")
            
            if traffic_light.get('adaptive'):
                print(f"      🧠 Modo adaptativo activado")
            if traffic_light.get('emergency_override'):
                print(f"      🚨 Override de emergencia activo")
        
        # 5. Simular cambios de estado
        print("\n🔄 Simulando operación del sistema...")
        
        for step in range(1, 6):
            print(f"\n--- Paso {step}/5 ---")
            
            # Obtener estado actual
            current_data = get_server_traffic_lights_data()
            if current_data:
                # Seleccionar un semáforo aleatorio para modificar
                selected_light = random.choice(current_data)
                node_id = selected_light['node_id']
                current_state = selected_light['state']
                
                # Cambiar a un estado diferente
                new_states = ['green', 'yellow', 'red']
                new_states.remove(current_state)
                new_state = random.choice(new_states)
                
                print(f"🔧 Cambiando semáforo {node_id}: {current_state} → {new_state}")
                
                # Aplicar cambio
                success = await modify_server_traffic_light(node_id, new_state, 15.0)
                if success:
                    print(f"   ✅ Cambio aplicado correctamente")
                else:
                    print(f"   ❌ Error aplicando cambio")
            
            # Mostrar métricas actualizadas cada 2 pasos
            if step % 2 == 0:
                print(f"\n📈 Métricas del paso {step}:")
                current_metrics = get_server_traffic_metrics()
                if current_metrics:
                    perf = current_metrics.get('performance', {})
                    print(f"   Eficiencia actual: {perf.get('average_efficiency', 0):.3f}")
                    print(f"   Procesados: {current_metrics.get('network_metrics', {}).get('total_vehicles_processed', 0)} vehículos")
            
            # Pausa entre pasos
            await asyncio.sleep(2)
        
        # 6. Demostrar características avanzadas
        print("\n🚨 Demostrando manejo de emergencias...")
        
        # Activar protocolo de emergencia en una ubicación
        emergency_lat, emergency_lon = 23.1145, -82.3675
        
        if hasattr(server_traffic_manager, 'handle_emergency_at_location'):
            emergency_success = await server_traffic_manager.handle_emergency_at_location(
                emergency_lat, emergency_lon, radius=0.005
            )
            
            if emergency_success:
                print(f"   ✅ Protocolo de emergencia activado en ({emergency_lat}, {emergency_lon})")
            else:
                print(f"   ⚠️ No se pudo activar protocolo de emergencia")
        
        # 7. Optimización de red
        print("\n⚡ Ejecutando optimización de red...")
        
        if hasattr(server_traffic_manager, 'optimize_traffic_network'):
            optimization_success = await server_traffic_manager.optimize_traffic_network()
            
            if optimization_success:
                print("   ✅ Optimización de red completada")
                
                # Mostrar métricas finales
                final_metrics = get_server_traffic_metrics()
                if final_metrics:
                    final_perf = final_metrics.get('performance', {})
                    print(f"   📊 Eficiencia final: {final_perf.get('average_efficiency', 0):.3f}")
            else:
                print("   ⚠️ Optimización no disponible")
        
        # 8. Resumen final
        print("\n" + "=" * 60)
        print("📋 RESUMEN DE LA DEMOSTRACIÓN")
        print("=" * 60)
        
        final_data = get_server_traffic_lights_data()
        final_metrics = get_server_traffic_metrics()
        
        print(f"✅ Semáforos gestionados: {len(final_data)}")
        print(f"✅ Sistema modular: {'Activo' if server_traffic_manager.is_ready() else 'Inactivo'}")
        
        if final_metrics:
            print(f"✅ Eficiencia del sistema: {final_metrics.get('performance', {}).get('average_efficiency', 0):.1%}")
            print(f"✅ Optimización habilitada: {'Sí' if final_metrics.get('controller_status', {}).get('optimization_enabled') else 'No'}")
        
        print("\n🎉 Demostración completada exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error durante la demostración: {e}")
        import traceback
        traceback.print_exc()


def demo_traffic_light_json_format():
    """Muestra el formato JSON para interacción con WebSocket"""
    print("\n" + "=" * 60)
    print("📡 FORMATOS JSON PARA WEBSOCKET")
    print("=" * 60)
    
    # Ejemplo de modificación de semáforo
    modify_example = {
        "type": "modify_traffic_light",
        "node_id": 101,
        "state": "green",
        "duration": 30.0
    }
    
    print("\n🔧 Modificar semáforo:")
    print(json.dumps(modify_example, indent=2))
    
    # Ejemplo de solicitud de métricas
    metrics_example = {
        "type": "get_traffic_metrics"
    }
    
    print("\n📊 Solicitar métricas:")
    print(json.dumps(metrics_example, indent=2))
    
    # Ejemplo de respuesta de estado
    response_example = {
        "type": "traffic_light_modified",
        "node_id": 101,
        "new_state": "green",
        "message": "Semáforo 101 cambiado a green"
    }
    
    print("\n✅ Respuesta de modificación:")
    print(json.dumps(response_example, indent=2))


if __name__ == "__main__":
    try:
        # Ejecutar demostración principal
        asyncio.run(run_traffic_light_demo())
        
        # Mostrar formatos de comunicación
        demo_traffic_light_json_format()
        
    except KeyboardInterrupt:
        print("\n👋 Demostración interrumpida por el usuario")
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        import traceback
        traceback.print_exc()
