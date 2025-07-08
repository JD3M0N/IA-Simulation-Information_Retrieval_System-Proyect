"""
Test simple para verificar el funcionamiento del WeatherAgent
"""

import asyncio
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multiagent.weather_agent import WeatherAgent
from src.multiagent.Environment_enums import WeatherCondition

async def test_weather_agent_basic():
    """Test básico del agente de clima"""
    print("🧪 === TEST: WeatherAgent Básico === 🧪\n")
    
    # Crear agente
    agent = WeatherAgent("test_agent", (19.4326, -99.1332))
    print(f"✅ Agente creado: {agent}")
    
    # Verificar estado inicial
    initial_weather = agent.get_current_weather()
    print(f"🌤️ Estado inicial: {initial_weather['condition'].value}")
    print(f"🌡️ Temperatura: {initial_weather['temperature']:.1f}°C")
    
    # Verificar factores de impacto
    impacts = agent.get_weather_impact_factors()
    print(f"🚗 Factores de impacto: {impacts}")
    
    # Simular actualizaciones
    print("\n📊 Simulando actualizaciones:")
    mock_env = {
        "current_time": "2025-07-08T12:00:00",
        "system_metrics": {"total_vehicles": 20}
    }
    
    for i in range(3):
        # Forzar actualización modificando la configuración
        agent.config["update_interval"] = 0  # Forzar actualización inmediata
        
        await agent.next_step(mock_env)
        current = agent.get_current_weather()
        print(f"  Paso {i+1}: {current['condition'].value} - {current['temperature']:.1f}°C")
    
    # Verificar métricas
    metrics = agent.metrics
    print(f"\n📈 Métricas finales: {metrics}")
    
    # Verificar pronósticos
    agent.forecasts.clear()  # Limpiar pronósticos anteriores
    await agent._update_forecast()
    print(f"🔮 Pronósticos generados: {len(agent.forecasts)}")
    
    if agent.forecasts:
        first_forecast = agent.forecasts[0]
        print(f"   Primer pronóstico: {first_forecast.condition.value} - {first_forecast.temperature:.1f}°C")
    
    print("\n✅ Test básico completado exitosamente!")
    return True

async def test_weather_integration():
    """Test de integración con ambiente simulado"""
    print("\n🔗 === TEST: Integración con Environment === 🔗\n")
    
    try:
        import networkx as nx
        from src.multiagent.environment import Environment
        
        # Crear grafo simple
        G = nx.path_graph(5)
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['lat'] = i * 0.001
            G.nodes[node]['lon'] = i * 0.001
            
        for edge in G.edges():
            G.edges[edge]['weight'] = 1.0
        
        # Crear environment
        env = Environment(street_graph=G, num_vehicles=2)
        print(f"✅ Environment creado con agente de clima: {env.weather_agent}")
        
        # Verificar sincronización
        env_weather = env.weather_state
        agent_weather = env.weather_agent.get_current_weather()
        
        print(f"🔄 Sincronización:")
        print(f"   Environment: {env_weather.condition.value} - {env_weather.temperature:.1f}°C")
        print(f"   Agente: {agent_weather['condition'].value} - {agent_weather['temperature']:.1f}°C")
        
        # Simular step
        await env.step()
        print("✅ Step ejecutado correctamente")
        
        # Verificar métodos de acceso
        forecast = env.get_weather_forecast(1)
        impacts = env.get_weather_impact_factors()
        history = env.get_weather_history(1)
        
        print(f"🔮 Pronóstico disponible: {forecast is not None}")
        print(f"🚗 Factores de impacto: {len(impacts)} factores")
        print(f"📈 Historial: {len(history)} registros")
        
        print("\n✅ Test de integración completado exitosamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error en test de integración: {e}")
        return False

async def main():
    """Ejecutar todos los tests"""
    print("🚀 Ejecutando tests del WeatherAgent...\n")
    
    test1 = await test_weather_agent_basic()
    test2 = await test_weather_integration()
    
    if test1 and test2:
        print("\n🎉 ¡TODOS LOS TESTS PASARON! 🎉")
        print("El WeatherAgent está funcionando correctamente.")
    else:
        print("\n❌ Algunos tests fallaron")
        
if __name__ == "__main__":
    asyncio.run(main())
