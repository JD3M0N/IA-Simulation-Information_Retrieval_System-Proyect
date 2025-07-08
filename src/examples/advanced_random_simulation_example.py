"""
Ejemplo de uso del generador avanzado de variables aleatorias
para simulación de tráfico y entregas más realista
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from multiagent.environment import AdvancedRandomGenerator
from multiagent.Environment_enums import WeatherCondition


def demo_delivery_demand_simulation():
    """Demostración de generación de demanda de paquetes durante un día"""
    print("=== Simulación de Demanda de Paquetes ===")
    
    rng = AdvancedRandomGenerator(seed=42)
    
    # Simular demanda por hora durante 24 horas
    hours = list(range(24))
    demands = []
    
    for hour in hours:
        # Simular demanda base de 50 paquetes/hora
        base_demand = 50.0
        
        # Factores por hora del día
        if 9 <= hour <= 17:  # Horas comerciales
            hour_factor = 1.5
        elif 18 <= hour <= 20:  # Pico residencial
            hour_factor = 2.0
        elif 7 <= hour <= 8 or 21 <= hour <= 22:
            hour_factor = 1.2
        else:
            hour_factor = 0.3
        
        adjusted_demand = base_demand * hour_factor
        demand = rng.poisson(adjusted_demand)
        demands.append(demand)
        
        print(f"Hora {hour:2d}: {demand:3d} paquetes (factor: {hour_factor:.1f})")
    
    total_day = sum(demands)
    print(f"\nTotal diario: {total_day} paquetes")
    print(f"Promedio por hora: {total_day/24:.1f} paquetes")
    
    return hours, demands


def demo_delivery_times_with_uncertainty():
    """Demostración de tiempos de entrega con incertidumbre"""
    print("\n=== Simulación de Tiempos de Entrega ===")
    
    rng = AdvancedRandomGenerator(seed=123)
    
    # Diferentes escenarios de distancia
    distances = [2, 5, 10, 15, 25]  # km
    traffic_scenarios = [
        ("Normal", 1.0),
        ("Congestión leve", 1.3),
        ("Congestión severa", 2.0),
        ("Tráfico extremo", 3.0)
    ]
    
    print("Tiempos de entrega (minutos) por distancia y tráfico:")
    print("Distancia\\Tráfico", end="")
    for scenario_name, _ in traffic_scenarios:
        print(f"\t{scenario_name[:10]}", end="")
    print()
    
    for distance in distances:
        print(f"{distance:3d} km", end="\t\t")
        for scenario_name, traffic_factor in traffic_scenarios:
            # Tiempo base: 30 km/h promedio
            base_time = (distance / 30.0) * 60.0
            
            # Variabilidad normal
            time_variation = rng.normal(1.0, 0.2)
            time_variation = max(0.5, min(2.0, time_variation))
            
            total_time = base_time * time_variation * traffic_factor
            
            # Retrasos ocasionales (15% probabilidad)
            if rng.binomial(1, 0.15):
                delay = rng.exponential(10)
                total_time += delay
            
            print(f"{total_time:6.1f}", end="\t")
        print()


def demo_vehicle_reliability():
    """Demostración de confiabilidad de vehículos"""
    print("\n=== Simulación de Confiabilidad de Vehículos ===")
    
    rng = AdvancedRandomGenerator(seed=456)
    
    # Diferentes vehículos con características
    vehicles = [
        ("Camión nuevo", 1.0, 0.95),
        ("Camión 3 años", 3.0, 0.85),
        ("Camión viejo", 7.0, 0.70),
        ("Van nueva", 0.5, 0.98),
        ("Van usada", 5.0, 0.75)
    ]
    
    print("Simulación de fallos en 365 días:")
    print("Vehículo\t\tFallos/año\tDisponibilidad")
    
    for name, age, maintenance in vehicles:
        failures = 0
        for day in range(365):
            # Parámetros Weibull
            shape = 1.5
            scale = 10.0 / age
            
            time_to_failure = rng.weibull(shape, scale) * maintenance
            daily_failure_prob = 1.0 / (time_to_failure * 365.25)
            
            if rng.binomial(1, min(0.1, daily_failure_prob)):
                failures += 1
        
        availability = (365 - failures) / 365 * 100
        print(f"{name:15s}\t{failures:3d}\t\t{availability:5.1f}%")


def demo_fuel_consumption_patterns():
    """Demostración de patrones de consumo de combustible"""
    print("\n=== Simulación de Consumo de Combustible ===")
    
    rng = AdvancedRandomGenerator(seed=789)
    
    # Escenarios de prueba
    distances = [50, 100, 200, 500]  # km
    weather_conditions = [
        WeatherCondition.CLEAR,
        WeatherCondition.LIGHT_RAIN,
        WeatherCondition.HEAVY_RAIN,
        WeatherCondition.STORM
    ]
    
    vehicle_types = ["truck", "van", "car"]
    
    print("Consumo de combustible por 100km (litros):")
    
    for vehicle_type in vehicle_types:
        print(f"\n{vehicle_type.upper()}:")
        print("Distancia\\Clima", end="")
        for condition in weather_conditions:
            print(f"\t{condition.value[:8]}", end="")
        print()
        
        for distance in distances:
            print(f"{distance:3d} km", end="\t")
            for condition in weather_conditions:
                # Consumo base
                base_consumption = {"truck": 25.0, "van": 15.0, "car": 8.0}[vehicle_type]
                
                # Factor climático
                weather_factors = {
                    WeatherCondition.CLEAR: 1.0,
                    WeatherCondition.LIGHT_RAIN: 1.1,
                    WeatherCondition.HEAVY_RAIN: 1.25,
                    WeatherCondition.STORM: 1.4
                }
                weather_factor = weather_factors[condition]
                
                # Variabilidad
                consumption_variation = rng.gamma(2, 0.4)
                consumption_per_100km = base_consumption * weather_factor * consumption_variation
                
                print(f"{consumption_per_100km:8.1f}", end="")
            print()


def demo_traffic_congestion_patterns():
    """Demostración de patrones de congestión durante el día"""
    print("\n=== Simulación de Congestión de Tráfico ===")
    
    rng = AdvancedRandomGenerator(seed=101112)
    
    hours = list(range(24))
    road_types = ["motorway", "primary", "residential"]
    
    print("Factores de congestión por hora y tipo de vía:")
    print("Hora", end="")
    for road_type in road_types:
        print(f"\t{road_type[:10]}", end="")
    print()
    
    for hour in hours:
        print(f"{hour:2d}:00", end="")
        for road_type in road_types:
            # Factores por hora
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Pico
                peak_factor = 1.0 + rng.beta(2, 5) * 3.0
            elif 10 <= hour <= 16:  # Normal
                peak_factor = 1.0 + rng.beta(5, 2) * 1.5
            elif 20 <= hour <= 22:  # Noche
                peak_factor = 1.0 + rng.beta(3, 7) * 1.0
            else:  # Madrugada
                peak_factor = 1.0 + rng.beta(8, 2) * 0.3
            
            # Factor por tipo de vía
            road_factors = {"motorway": 0.8, "primary": 1.0, "residential": 1.2}
            road_factor = road_factors.get(road_type, 1.0)
            
            congestion = peak_factor * road_factor
            print(f"\t{congestion:8.2f}", end="")
        print()


def monte_carlo_cost_analysis():
    """Análisis Monte Carlo de costos de entrega"""
    print("\n=== Análisis Monte Carlo de Costos ===")
    
    rng = AdvancedRandomGenerator(seed=131415)
    
    # Parámetros del análisis
    n_simulations = 1000
    base_cost_per_km = 2.5  # €/km
    penalty_per_minute_delay = 0.5  # €/min
    
    costs = []
    
    for _ in range(n_simulations):
        # Generar escenario aleatorio
        distance = rng.gamma(2, 5)  # Distancias típicas urbanas
        traffic_factor = 1.0 + rng.exponential(0.3)  # Factor de tráfico
        
        # Tiempo base y variaciones
        base_time = (distance / 30.0) * 60  # minutos
        actual_time = base_time * traffic_factor * rng.normal(1.0, 0.15)
        
        # Retrasos excepcionales
        if rng.binomial(1, 0.1):  # 10% probabilidad
            delay = rng.exponential(20)
            actual_time += delay
        
        # Costo total
        transport_cost = distance * base_cost_per_km
        delay_penalty = max(0, actual_time - base_time) * penalty_per_minute_delay
        total_cost = transport_cost + delay_penalty
        
        costs.append(total_cost)
    
    # Análisis estadístico
    costs = np.array(costs)
    print(f"Número de simulaciones: {n_simulations}")
    print(f"Costo promedio: {np.mean(costs):.2f} €")
    print(f"Desviación estándar: {np.std(costs):.2f} €")
    print(f"Costo mínimo: {np.min(costs):.2f} €")
    print(f"Costo máximo: {np.max(costs):.2f} €")
    print(f"Percentil 95: {np.percentile(costs, 95):.2f} €")
    print(f"Percentil 99: {np.percentile(costs, 99):.2f} €")
    
    return costs


def create_visualization_plots():
    """Crear gráficos de visualización de las distribuciones"""
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        rng = AdvancedRandomGenerator(seed=42)
        
        # 1. Distribución de demanda horaria
        hours, demands = demo_delivery_demand_simulation()
        ax1.bar(hours, demands, alpha=0.7, color='skyblue')
        ax1.set_title('Demanda de Paquetes por Hora')
        ax1.set_xlabel('Hora del día')
        ax1.set_ylabel('Número de paquetes')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribución de tiempos de entrega
        delivery_times = []
        for _ in range(1000):
            distance = rng.gamma(2, 5)
            traffic_factor = 1.0 + rng.exponential(0.3)
            base_time = (distance / 30.0) * 60
            time = base_time * traffic_factor * rng.normal(1.0, 0.15)
            if rng.binomial(1, 0.1):
                time += rng.exponential(20)
            delivery_times.append(time)
        
        ax2.hist(delivery_times, bins=50, alpha=0.7, color='lightgreen', density=True)
        ax2.set_title('Distribución de Tiempos de Entrega')
        ax2.set_xlabel('Tiempo (minutos)')
        ax2.set_ylabel('Densidad')
        ax2.grid(True, alpha=0.3)
        
        # 3. Factores de congestión por hora
        congestion_factors = []
        hours_extended = []
        for hour in range(24):
            for _ in range(10):  # 10 muestras por hora
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    factor = 1.0 + rng.beta(2, 5) * 3.0
                elif 10 <= hour <= 16:
                    factor = 1.0 + rng.beta(5, 2) * 1.5
                elif 20 <= hour <= 22:
                    factor = 1.0 + rng.beta(3, 7) * 1.0
                else:
                    factor = 1.0 + rng.beta(8, 2) * 0.3
                
                congestion_factors.append(factor)
                hours_extended.append(hour)
        
        ax3.scatter(hours_extended, congestion_factors, alpha=0.6, color='orange', s=10)
        ax3.set_title('Factores de Congestión por Hora')
        ax3.set_xlabel('Hora del día')
        ax3.set_ylabel('Factor de congestión')
        ax3.grid(True, alpha=0.3)
        
        # 4. Análisis de costos Monte Carlo
        costs = monte_carlo_cost_analysis()
        ax4.hist(costs, bins=50, alpha=0.7, color='lightcoral', density=True)
        ax4.axvline(np.mean(costs), color='red', linestyle='--', label=f'Media: {np.mean(costs):.2f}€')
        ax4.axvline(np.percentile(costs, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(costs, 95):.2f}€')
        ax4.set_title('Distribución de Costos de Entrega')
        ax4.set_xlabel('Costo (€)')
        ax4.set_ylabel('Densidad')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_random_simulation_results.png', dpi=300, bbox_inches='tight')
        print("\n📊 Gráficos guardados en 'advanced_random_simulation_results.png'")
        
        # Mostrar gráficos si es posible
        plt.show()
        
    except ImportError:
        print("\n⚠️ matplotlib no disponible - no se pueden crear gráficos")


def main():
    """Función principal que ejecuta todas las demostraciones"""
    print("🎲 DEMOSTRACIÓN DE GENERACIÓN AVANZADA DE VARIABLES ALEATORIAS")
    print("=" * 70)
    
    # Ejecutar todas las demostraciones
    demo_delivery_demand_simulation()
    demo_delivery_times_with_uncertainty()
    demo_vehicle_reliability()
    demo_fuel_consumption_patterns()
    demo_traffic_congestion_patterns()
    monte_carlo_cost_analysis()
    
    # Crear visualizaciones
    create_visualization_plots()
    
    print("\n" + "=" * 70)
    print("✅ Demostración completada")
    print("\nEsta implementación usa métodos estadísticos fundamentales:")
    print("• Transformada inversa para exponencial y Weibull")
    print("• Box-Muller para distribución normal")
    print("• Algoritmo de Knuth para Poisson")
    print("• Método de Marsaglia-Tsang para Gamma")
    print("• Relaciones matemáticas para Beta y Log-normal")
    print("\n🔬 Esto hace la simulación más robusta y realista que usar random() básico")


if __name__ == "__main__":
    main()
