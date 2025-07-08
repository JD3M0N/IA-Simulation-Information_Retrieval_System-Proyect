#!/usr/bin/env python3
"""
Script de prueba para verificar el movimiento de vehículos
Ejecuta el servidor con debug específico para movimiento de vehículos
"""

import asyncio
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.multiagent.environment import Environment
import networkx as nx
import json

async def test_vehicle_movement():
    """Prueba el movimiento de vehículos civiles"""
    print("🚀 Iniciando prueba de movimiento de vehículos...")
    
    # Crear un grafo simple para pruebas
    street_graph = nx.MultiDiGraph()
    
    # Agregar nodos con coordenadas
    nodes = [
        (0, {"lat": 23.1298784, "lon": -82.3490351}),
        (1, {"lat": 23.1300784, "lon": -82.3488351}),
        (2, {"lat": 23.1302784, "lon": -82.3486351}),
        (3, {"lat": 23.1304784, "lon": -82.3484351}),
        (4, {"lat": 23.1306784, "lon": -82.3482351}),
    ]
    
    for node_id, data in nodes:
        street_graph.add_node(node_id, **data)
    
    # Agregar aristas con pesos
    edges = [
        (0, 1, {"weight": 0.5}),
        (1, 2, {"weight": 0.3}),
        (2, 3, {"weight": 0.4}),
        (3, 4, {"weight": 0.6}),
        (4, 0, {"weight": 0.8}),
        (1, 3, {"weight": 0.7}),
        (2, 4, {"weight": 0.5}),
    ]
    
    for src, dst, data in edges:
        street_graph.add_edge(src, dst, **data)
    
    print(f"📊 Grafo creado con {len(street_graph.nodes)} nodos y {len(street_graph.edges)} aristas")
    
    # Crear entorno con pocos vehículos para prueba
    environment = Environment(street_graph, num_vehicles=3)
    
    print("🏗️ Entorno creado, ejecutando pasos de simulación...")
    
    # Ejecutar algunos pasos de simulación
    for step in range(20):
        print(f"\n--- Paso {step + 1} ---")
        
        # Ejecutar un paso
        await environment.step()
        
        # Obtener posiciones de vehículos
        positions = environment.get_vehicle_positions()
        
        print(f"📍 Posiciones de {len(positions)} vehículos:")
        for vehicle in positions:
            print(f"   🚗 {vehicle['id']}: ({vehicle['lat']:.6f}, {vehicle['lon']:.6f}) "
                  f"velocidad: {vehicle['speed']:.1f} km/h, estado: {vehicle['state']}")
        
        # Esperar un poco entre pasos
        await asyncio.sleep(0.1)
    
    print("\n✅ Prueba completada")

if __name__ == "__main__":
    asyncio.run(test_vehicle_movement())
