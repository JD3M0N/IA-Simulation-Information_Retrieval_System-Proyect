"""
Demostración del Sistema BDI para Camiones de Reparto
Script de prueba que muestra las capacidades del sistema BDI
"""

import sys
import asyncio
import random
import networkx as nx
from typing import List, Dict, Any

# Añadir paths
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from environment import Environment
from delivery_truck_bdi import DeliveryTruckBDI
from communication_system import communication_manager, MessageType
from Environment_enums import WeatherCondition

class BDIDemo:
    """Demostración del sistema BDI"""
    
    def __init__(self):
        self.environment = None
        self.demo_trucks = []
        self.demo_running = False
    
    def create_demo_graph(self) -> nx.Graph:
        """Crea un grafo de demostración"""
        print("📊 Creando grafo de demostración...")
        
        # Crear grafo grid simple para demostración
        G = nx.grid_2d_graph(10, 10)
        
        # Convertir a grafo dirigido y añadir coordenadas
        street_graph = nx.DiGraph()
        
        for node in G.nodes():
            x, y = node
            # Convertir coordenadas grid a coordenadas geográficas ficticias
            lat = 23.1136 + (y * 0.001)  # La Habana como referencia
            lon = -82.3666 + (x * 0.001)
            
            # Convertir tupla a entero para compatibilidad
            node_id = x * 10 + y
            street_graph.add_node(node_id, lat=lat, lon=lon)
        
        # Añadir aristas con pesos
        for edge in G.edges():
            (x1, y1), (x2, y2) = edge
            node1_id = x1 * 10 + y1
            node2_id = x2 * 10 + y2
            
            # Añadir aristas bidireccionales
            weight = random.uniform(0.5, 2.0)
            max_speed = random.choice([30, 50, 60])
            
            street_graph.add_edge(node1_id, node2_id, 
                                weight=weight, 
                                max_speed=max_speed,
                                highway='residential')
            street_graph.add_edge(node2_id, node1_id, 
                                weight=weight, 
                                max_speed=max_speed,
                                highway='residential')
        
        print(f"✅ Grafo creado: {len(street_graph.nodes)} nodos, {len(street_graph.edges)} aristas")
        return street_graph
    
    def setup_environment(self):
        """Configura el entorno de simulación"""
        print("🏗️ Configurando entorno de simulación...")
        
        # Crear grafo de demostración
        demo_graph = self.create_demo_graph()
        
        # Crear entorno
        self.environment = Environment(demo_graph, num_vehicles=5)
        
        print("✅ Entorno configurado")
    
    async def create_demo_trucks(self):
        """Crea camiones de demostración"""
        print("🚛 Creando camiones de demostración...")
        
        # Obtener nodos disponibles
        nodes = list(self.environment.street_graph.nodes())
        
        # Configuraciones de camiones de demostración
        truck_configs = [
            {
                "id": "truck_fuel_saver",
                "start_node": random.choice(nodes),
                "capacity": 1000,
                "deliveries": random.sample(nodes, 5),
                "description": "Camión optimizado para ahorro de combustible"
            },
            {
                "id": "truck_time_optimizer", 
                "start_node": random.choice(nodes),
                "capacity": 1200,
                "deliveries": random.sample(nodes, 6),
                "description": "Camión optimizado para minimizar tiempo"
            },
            {
                "id": "truck_max_deliveries",
                "start_node": random.choice(nodes),
                "capacity": 800,
                "deliveries": random.sample(nodes, 8),
                "description": "Camión optimizado para maximizar entregas"
            },
            {
                "id": "truck_collaborator",
                "start_node": random.choice(nodes),
                "capacity": 900,
                "deliveries": random.sample(nodes, 4),
                "description": "Camión enfocado en colaboración"
            }
        ]
        
        # Crear camiones
        for config in truck_configs:
            success = self.environment.add_bdi_delivery_truck(
                truck_id=config["id"],
                start_node=config["start_node"],
                capacity=config["capacity"],
                delivery_locations=config["deliveries"]
            )
            
            if success:
                self.demo_trucks.append(config["id"])
                print(f"✅ {config['description']} creado: {config['id']}")
                print(f"   📍 Nodo inicial: {config['start_node']}")
                print(f"   📦 Entregas: {len(config['deliveries'])} ubicaciones")
            else:
                print(f"❌ Error creando {config['id']}")
        
        # Iniciar sistema de comunicación
        await self.environment.start_communication_system()
        
        print(f"✅ Sistema BDI inicializado con {len(self.demo_trucks)} camiones")
    
    async def run_demo_scenario(self, duration_seconds: int = 120):
        """Ejecuta un escenario de demostración"""
        print(f"🎬 Iniciando demostración BDI por {duration_seconds} segundos...")
        print("=" * 60)
        
        self.demo_running = True
        
        # Iniciar movimiento de camiones
        self.environment.start_bdi_trucks_movement()
        
        start_time = asyncio.get_event_loop().time()
        step_count = 0
        
        while self.demo_running and (asyncio.get_event_loop().time() - start_time) < duration_seconds:
            try:
                # Actualizar entorno
                delta_time = 1.0  # 1 segundo por paso
                await self.environment.update_bdi_trucks(delta_time)
                
                step_count += 1
                
                # Mostrar estado cada 20 pasos (20 segundos)
                if step_count % 20 == 0:
                    await self._show_demo_status(step_count)
                
                # Simular eventos ocasionales
                if step_count % 30 == 0:  # Cada 30 segundos
                    await self._simulate_random_event()
                
                # Pausa entre pasos
                await asyncio.sleep(1.0)
                
            except KeyboardInterrupt:
                print("\n⏹️ Demostración interrumpida por el usuario")
                break
            except Exception as e:
                print(f"❌ Error en demostración: {e}")
                break
        
        print("\n🏁 Demostración completada")
        await self._show_final_results()
    
    async def _show_demo_status(self, step: int):
        """Muestra el estado actual de la demostración"""
        print(f"\n📊 Estado de la simulación - Paso {step}")
        print("-" * 50)
        
        # Obtener estado de camiones BDI
        bdi_status = self.environment.get_bdi_trucks_status()
        
        for truck_id, status in bdi_status.items():
            if "error" in status:
                print(f"❌ {truck_id}: {status['error']}")
                continue
            
            bdi_info = status.get("bdi_status", {})
            delivery_info = status.get("metrics", {})
            
            print(f"🚛 {truck_id}:")
            print(f"   📍 Nodo: {status.get('current_node', 'N/A')}")
            print(f"   ⛽ Combustible: {status.get('fuel_level', 0):.1f}%")
            print(f"   📦 Entregas completadas: {delivery_info.get('deliveries_completed', 0)}")
            print(f"   🧠 Decisiones BDI: {bdi_info.get('decisions_made', 0)}")
            print(f"   🎯 Intenciones ejecutadas: {bdi_info.get('intentions_executed', 0)}")
        
        # Mostrar estadísticas de comunicación
        comm_stats = self.environment.get_communication_stats()
        if "error" not in comm_stats:
            print(f"\n📡 Comunicación:")
            print(f"   📨 Mensajes enviados: {comm_stats['stats']['messages_sent']}")
            print(f"   📬 Mensajes entregados: {comm_stats['stats']['messages_delivered']}")
            print(f"   📢 Mensajes broadcast: {comm_stats['stats']['broadcast_messages']}")
    
    async def _simulate_random_event(self):
        """Simula eventos aleatorios para probar el sistema BDI"""
        events = [
            "traffic_congestion",
            "weather_change", 
            "emergency_alert",
            "fuel_station_closed"
        ]
        
        event = random.choice(events)
        
        if event == "traffic_congestion":
            print("🚦 Evento: Congestión de tráfico detectada")
            # Simular actualización de tráfico via comunicación
            traffic_data = {
                "congestion_level": random.uniform(0.7, 1.0),
                "affected_areas": random.sample(list(self.environment.street_graph.nodes()), 3),
                "estimated_delay": random.randint(10, 30)
            }
            
            # Enviar actualización desde el primer camión
            if self.demo_trucks:
                communication_manager.broadcast_traffic_update(
                    self.demo_trucks[0], traffic_data
                )
        
        elif event == "emergency_alert":
            print("🚨 Evento: Alerta de emergencia")
            emergency_data = {
                "type": "emergency_vehicle",
                "location": random.choice(list(self.environment.street_graph.nodes())),
                "priority": "high",
                "estimated_duration": random.randint(5, 15)
            }
            
            if self.demo_trucks:
                communication_manager.broadcast_emergency(
                    self.demo_trucks[0], emergency_data
                )
        
        elif event == "weather_change":
            print("🌧️ Evento: Cambio climático")
            # Actualizar estado del clima en el entorno
            self.environment.weather_state.condition = random.choice([
                WeatherCondition.CLEAR, WeatherCondition.CLOUDY, 
                WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN
            ])
        
        elif event == "fuel_station_closed":
            print("⛽ Evento: Estación de combustible cerrada")
            # Este evento afectaría las decisiones de los agentes BDI
    
    async def _show_final_results(self):
        """Muestra resultados finales de la demostración"""
        print("\n" + "=" * 60)
        print("📈 RESULTADOS FINALES DE LA DEMOSTRACIÓN BDI")
        print("=" * 60)
        
        # Métricas del sistema
        metrics = self.environment.system_metrics
        print(f"\n🏭 Métricas del Sistema:")
        print(f"   🚛 Total camiones BDI: {len(self.demo_trucks)}")
        print(f"   🧠 Decisiones BDI totales: {metrics.get('bdi_decisions_made', 0)}")
        print(f"   🎯 Intenciones ejecutadas: {metrics.get('bdi_intentions_executed', 0)}")
        print(f"   🤝 Colaboraciones: {metrics.get('bdi_collaborations', 0)}")
        
        # Estado final de cada camión
        bdi_status = self.environment.get_bdi_trucks_status()
        print(f"\n🚛 Estado Final de Camiones:")
        
        total_deliveries = 0
        total_distance = 0.0
        total_fuel_consumed = 0.0
        
        for truck_id, status in bdi_status.items():
            if "error" in status:
                continue
            
            delivery_metrics = status.get("metrics", {})
            deliveries = delivery_metrics.get("deliveries_completed", 0)
            distance = delivery_metrics.get("total_distance", 0.0)
            fuel = delivery_metrics.get("fuel_consumed", 0.0)
            
            total_deliveries += deliveries
            total_distance += distance
            total_fuel_consumed += fuel
            
            print(f"   {truck_id}:")
            print(f"     📦 Entregas: {deliveries}")
            print(f"     🛣️ Distancia: {distance:.2f} km")
            print(f"     ⛽ Combustible: {fuel:.2f} L")
            print(f"     ⛽ Nivel final: {status.get('fuel_level', 0):.1f}%")
        
        # Totales
        print(f"\n📊 Totales:")
        print(f"   📦 Entregas completadas: {total_deliveries}")
        print(f"   🛣️ Distancia total: {total_distance:.2f} km")
        print(f"   ⛽ Combustible total: {total_fuel_consumed:.2f} L")
        
        if total_deliveries > 0:
            print(f"   📈 Eficiencia promedio: {total_distance/total_deliveries:.2f} km/entrega")
        
        # Estadísticas de comunicación
        comm_stats = self.environment.get_communication_stats()
        if "error" not in comm_stats:
            stats = comm_stats.get('stats', {})
            print(f"\n📡 Comunicación:")
            print(f"   📨 Mensajes enviados: {stats.get('messages_sent', 0)}")
            print(f"   📬 Mensajes entregados: {stats.get('messages_delivered', 0)}")
            print(f"   📢 Broadcasts: {stats.get('broadcast_messages', 0)}")
            
            if stats.get('messages_sent', 0) > 0:
                delivery_rate = stats.get('messages_delivered', 0) / stats.get('messages_sent', 1)
                print(f"   📊 Tasa de entrega: {delivery_rate:.1%}")
    
    async def stop_demo(self):
        """Detiene la demostración"""
        self.demo_running = False
        
        if self.environment:
            await self.environment.stop_communication_system()
        
        print("✅ Demostración detenida")
    
    async def run_interactive_demo(self):
        """Ejecuta demostración interactiva"""
        print("\n" + "=" * 60)
        print("🎮 DEMOSTRACIÓN INTERACTIVA BDI")
        print("=" * 60)
        print("Esta demostración muestra:")
        print("• ✅ Agentes BDI con creencias, deseos e intenciones")
        print("• 🧠 Toma de decisiones inteligente")
        print("• 🤝 Comunicación y colaboración entre agentes")
        print("• 🎯 Optimización de rutas, combustible y entregas")
        print("• 📡 Sistema de mensajes entre camiones")
        print("=" * 60)
        
        try:
            # Configurar entorno
            self.setup_environment()
            
            # Crear camiones
            await self.create_demo_trucks()
            
            # Ejecutar demostración
            await self.run_demo_scenario(120)  # 2 minutos
            
        except KeyboardInterrupt:
            print("\n⏹️ Demostración interrumpida")
        except Exception as e:
            print(f"❌ Error en demostración: {e}")
        finally:
            await self.stop_demo()

async def main():
    """Función principal"""
    demo = BDIDemo()
    await demo.run_interactive_demo()

if __name__ == "__main__":
    print("🚛 Sistema BDI para Camiones de Reparto")
    print("Desarrollado para IA-Simulation-Information_Retrieval_System")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"❌ Error crítico: {e}")
