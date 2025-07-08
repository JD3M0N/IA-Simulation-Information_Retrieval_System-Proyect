"""
Ejemplo de uso del WeatherAgent en la simulación
Demuestra cómo interactuar con el agente de clima
"""

import asyncio
import networkx as nx
import random
from datetime import datetime, timedelta

from src.multiagent.environment import Environment
from src.multiagent.weather_agent import WeatherAgent
from src.multiagent.Environment_enums import WeatherCondition

async def demo_weather_agent():
    """Demostración del agente de clima"""
    
    print("🌤️ === DEMO: Agente de Clima en Simulación === 🌤️\n")
    
    # Crear un grafo simple para la demostración
    G = nx.grid_2d_graph(10, 10)
    # Agregar coordenadas a los nodos
    for node in G.nodes():
        G.nodes[node]['lat'] = node[0] * 0.001  # Simulamos coordenadas
        G.nodes[node]['lon'] = node[1] * 0.001
    
    # Agregar pesos a las aristas
    for edge in G.edges():
        G.edges[edge]['weight'] = random.uniform(1.0, 5.0)
    
    # Crear el entorno con agente de clima integrado
    print("📍 Creando entorno de simulación...")
    environment = Environment(street_graph=G, num_vehicles=5)
    
    print(f"✅ Entorno creado con agente de clima: {environment.weather_agent}")
    print(f"🌡️ Clima inicial: {environment.weather_agent}")
    print()
    
    # Mostrar estado inicial del clima
    print("=== ESTADO INICIAL DEL CLIMA ===")
    current_weather = environment.weather_agent.get_current_weather()
    print(f"🌤️ Condición: {current_weather['condition'].value}")
    print(f"🌡️ Temperatura: {current_weather['temperature']:.1f}°C")
    print(f"💧 Humedad: {current_weather['humidity']:.1f}%")
    print(f"💨 Viento: {current_weather['wind_speed']:.1f} km/h")
    print(f"🌧️ Precipitación: {current_weather['precipitation']:.1f} mm/h")
    print(f"👁️ Visibilidad: {current_weather['visibility']:.1f} km")
    print()
    
    # Mostrar factores de impacto
    print("=== FACTORES DE IMPACTO EN EL TRÁFICO ===")
    impact_factors = environment.get_weather_impact_factors()
    print(f"🚗 Factor velocidad: {impact_factors.get('speed_factor', 1.0):.2f}")
    print(f"👁️ Factor visibilidad: {impact_factors.get('visibility_factor', 1.0):.2f}")
    print(f"⚠️ Riesgo accidentes: {impact_factors.get('accident_risk', 1.0):.2f}")
    print(f"⛽ Consumo combustible: {impact_factors.get('fuel_consumption', 1.0):.2f}")
    print()
    
    # Simular varios pasos de tiempo
    print("=== SIMULACIÓN DE EVOLUCIÓN CLIMÁTICA ===")
    for step in range(5):
        print(f"\n--- Paso {step + 1} ---")
        
        # Avanzar la simulación
        await environment.step()
        
        # Mostrar cambios en el clima
        current_weather = environment.weather_agent.get_current_weather()
        print(f"🌤️ Condición: {current_weather['condition'].value}")
        print(f"🌡️ Temperatura: {current_weather['temperature']:.1f}°C")
        print(f"💧 Humedad: {current_weather['humidity']:.1f}%")
        
        # Mostrar métricas del agente
        metrics = environment.get_weather_agent_metrics()
        print(f"📊 Actualizaciones totales: {metrics['total_updates']}")
        print(f"⚡ Eventos extremos: {metrics['extreme_weather_events']}")
        print(f"🌧️ Eventos precipitación: {metrics['precipitation_events']}")
        
        # Verificar si hay eventos extremos
        if environment.weather_agent.current_extreme_event:
            print(f"🚨 EVENTO EXTREMO ACTIVO: {environment.weather_agent.current_extreme_event.value}")
            print(f"⏰ Finaliza: {environment.weather_agent.extreme_event_end_time}")
        
        await asyncio.sleep(1)  # Pausa para visualización
    
    print("\n=== PRONÓSTICO DEL CLIMA ===")
    # Mostrar pronóstico para las próximas horas
    for hour in range(1, 6):
        forecast = environment.get_weather_forecast(hour)
        if forecast:
            print(f"🕐 +{hour}h: {forecast['condition']} - {forecast['temperature']:.1f}°C "
                  f"(confianza: {forecast['confidence']:.0%})")
    
    print("\n=== HISTORIAL CLIMÁTICO ===")
    # Mostrar historial reciente
    history = environment.get_weather_history(hours=1)
    print(f"📈 Registros en historial: {len(history)}")
    if history:
        last_record = history[-1]
        print(f"🕐 Último registro: {last_record['condition'].value} - {last_record['temperature']:.1f}°C")
    
    print("\n=== EXPORTACIÓN DE DATOS ===")
    # Exportar todos los datos del clima
    weather_data = environment.export_weather_data()
    print(f"📊 Datos exportados del agente: {weather_data['agent_id']}")
    print(f"🌡️ Condición actual: {weather_data['current_weather']['condition'].value}")
    print(f"🔮 Pronósticos disponibles: {len(weather_data['forecasts'])}")
    print(f"📈 Tamaño del historial: {weather_data['history_size']}")
    
    print("\n🎯 === DEMO COMPLETADA === 🎯")
    print("El agente de clima está funcionando correctamente y proporciona:")
    print("✅ Evolución realista del clima")
    print("✅ Factores de impacto en el tráfico")
    print("✅ Pronósticos con confianza")
    print("✅ Historial para análisis")
    print("✅ Eventos climáticos extremos")
    print("✅ Integración completa con el entorno")

async def demo_weather_agent_standalone():
    """Demostración del agente de clima como entidad independiente"""
    
    print("\n🌦️ === DEMO: WeatherAgent Independiente === 🌦️\n")
    
    # Crear agente de clima independiente
    weather_agent = WeatherAgent(
        agent_id="demo_weather",
        location=(19.4326, -99.1332)  # Ciudad de México como ejemplo
    )
    
    print(f"🌍 Agente de clima creado para ubicación: {weather_agent.location}")
    print(f"📊 Estado inicial: {weather_agent}")
    
    # Simular ambiente básico
    mock_environment = {
        "current_time": datetime.now(),
        "system_metrics": {
            "total_vehicles": 25,
            "average_speed": 45.0
        }
    }
    
    print("\n=== SIMULACIÓN INDEPENDIENTE ===")
    for i in range(3):
        print(f"\n--- Actualización {i + 1} ---")
        
        # Actualizar agente
        await weather_agent.next_step(mock_environment)
        
        # Mostrar estado
        current = weather_agent.get_current_weather()
        print(f"🌤️ {current['condition'].value} - {current['temperature']:.1f}°C")
        
        # Mostrar impactos
        impacts = weather_agent.get_weather_impact_factors()
        print(f"🚗 Impacto velocidad: {impacts['speed_factor']:.2f}")
        
        await asyncio.sleep(0.5)
    
    print(f"\n📈 Métricas finales: {weather_agent.metrics}")

if __name__ == "__main__":
    # Ejecutar demos
    asyncio.run(demo_weather_agent())
    asyncio.run(demo_weather_agent_standalone())
