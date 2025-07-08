"""
Análisis del Comportamiento de Vehículos con Bootstrapping y Pruebas de Hipótesis
================================================================================

Este módulo realiza un análisis estadístico exhaustivo del comportamiento de los vehículos
en la simulación de tráfico, utilizando técnicas de bootstrapping para estimar distribuciones
y realizar pruebas de hipótesis sobre diferentes aspectos del movimiento vehicular.

Autor: Sistema de Análisis de Tráfico IA
Fecha: Julio 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, shapiro, anderson
import networkx as nx
import random
import math
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para los gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Parámetros de simulación
SIMULATION_STEPS = 1000
NUM_VEHICLES_TEST = 500
BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95

# Tipos de vehículos con sus características
VEHICLE_TYPES = {
    "normal": {"speed_factor": (0.75, 0.9), "aggression": 0.5, "color": "blue"},
    "agresivo": {"speed_factor": (0.85, 1.05), "aggression": 0.9, "color": "red"},
    "cauteloso": {"speed_factor": (0.7, 0.85), "aggression": 0.2, "color": "green"},
    "lento": {"speed_factor": (0.5, 0.7), "aggression": 0.3, "color": "orange"},
    "rápido": {"speed_factor": (0.9, 1.0), "aggression": 0.7, "color": "purple"}
}

class VehicleBehaviorSimulator:
    """Simulador independiente para analizar el comportamiento de vehículos"""
    
    def __init__(self, num_vehicles=NUM_VEHICLES_TEST):
        self.num_vehicles = num_vehicles
        self.vehicles = {}
        self.vehicle_metrics = {}
        self.simulation_data = []
        self.street_graph = self._create_test_graph()
        self.all_nodes = list(self.street_graph.nodes())
        self.current_step = 0
        
    def _create_test_graph(self):
        """Crea un grafo de prueba para la simulación"""
        graph = nx.MultiDiGraph()
        
        # Crear una cuadrícula de calles
        grid_size = 10
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = i * grid_size + j
                lat = 23.1136 + i * 0.001
                lon = -82.3666 + j * 0.001
                graph.add_node(node_id, lat=lat, lon=lon)
                
                # Conectar con nodos adyacentes
                if j < grid_size - 1:  # Conexión horizontal
                    next_node = i * grid_size + (j + 1)
                    speed = random.choice([30, 50, 70, 90])
                    highway_type = random.choice(["residential", "secondary", "primary", "tertiary"])
                    graph.add_edge(node_id, next_node, 
                                 max_speed=speed, min_speed=speed*0.7, 
                                 highway_type=highway_type, weight=0.001)
                    graph.add_edge(next_node, node_id, 
                                 max_speed=speed, min_speed=speed*0.7, 
                                 highway_type=highway_type, weight=0.001)
                
                if i < grid_size - 1:  # Conexión vertical
                    next_node = (i + 1) * grid_size + j
                    speed = random.choice([30, 50, 70, 90])
                    highway_type = random.choice(["residential", "secondary", "primary", "tertiary"])
                    graph.add_edge(node_id, next_node, 
                                 max_speed=speed, min_speed=speed*0.7, 
                                 highway_type=highway_type, weight=0.001)
                    graph.add_edge(next_node, node_id, 
                                 max_speed=speed, min_speed=speed*0.7, 
                                 highway_type=highway_type, weight=0.001)
        
        return graph
    
    def initialize_vehicles(self):
        """Inicializa los vehículos con diferentes comportamientos"""
        self.vehicles = {}
        self.vehicle_metrics = {}
        
        for i in range(self.num_vehicles):
            vid = f"vehicle_{i}"
            vehicle_type = random.choice(list(VEHICLE_TYPES.keys()))
            start_node = random.choice(self.all_nodes)
            node_data = self.street_graph.nodes[start_node]
            
            # Configuración del vehículo
            speed_range = VEHICLE_TYPES[vehicle_type]["speed_factor"]
            speed_factor = random.uniform(*speed_range)
            
            self.vehicles[vid] = {
                "lat": float(node_data['lat']),
                "lon": float(node_data['lon']),
                "current_node": start_node,
                "next_node": None,
                "previous_node": None,
                "progress": 0.0,
                "type": vehicle_type,
                "speed_factor": speed_factor,
                "aggression": VEHICLE_TYPES[vehicle_type]["aggression"],
                "distance_traveled": 0.0,
                "stops_count": 0,
                "speed_changes": 0,
                "average_speed": 0.0,
                "max_speed_reached": 0.0,
                "time_in_congestion": 0,
                "route_efficiency": 1.0
            }
            
            # Inicializar métricas
            self.vehicle_metrics[vid] = {
                "speeds": [],
                "positions": [],
                "accelerations": [],
                "distances": [],
                "wait_times": [],
                "route_changes": 0
            }
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calcula distancia haversine entre dos puntos"""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c * 1000  # en metros
    
    def _assign_next_node(self, vehicle_id):
        """Asigna el siguiente nodo basado en el comportamiento del vehículo"""
        current = self.vehicles[vehicle_id]["current_node"]
        previous = self.vehicles[vehicle_id]["previous_node"]
        vehicle_type = self.vehicles[vehicle_id]["type"]
        aggression = self.vehicles[vehicle_id]["aggression"]
        
        neighbors = list(self.street_graph.neighbors(current))
        
        if not neighbors:
            return None
        
        # Filtrar nodo anterior si hay opciones
        if previous and len(neighbors) > 1:
            neighbors = [n for n in neighbors if n != previous]
        
        if not neighbors:
            return random.choice(list(self.street_graph.neighbors(current)))
        
        # Selección basada en agresividad
        if aggression > 0.7:  # Vehículos agresivos prefieren rutas más rápidas
            weights = []
            for neighbor in neighbors:
                edge_data = self.street_graph.get_edge_data(current, neighbor, 0)
                max_speed = edge_data.get('max_speed', 50)
                weights.append(max_speed)
        else:  # Vehículos cautelosos prefieren rutas más seguras
            weights = [1] * len(neighbors)  # Selección uniforme
        
        # Selección ponderada
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
            return np.random.choice(neighbors, p=weights)
        else:
            return random.choice(neighbors)
    
    def update_vehicle_position(self, vehicle_id):
        """Actualiza la posición de un vehículo específico"""
        vehicle = self.vehicles[vehicle_id]
        current_node = vehicle["current_node"]
        next_node = vehicle["next_node"]
        
        # Asignar siguiente nodo si no existe
        if next_node is None:
            next_node = self._assign_next_node(vehicle_id)
            if next_node is None:
                return
            vehicle["next_node"] = next_node
            vehicle["progress"] = 0.0
            self.vehicle_metrics[vehicle_id]["route_changes"] += 1
        
        # Obtener información de la calle
        edge_data = self.street_graph.get_edge_data(current_node, next_node, 0)
        max_speed = edge_data.get('max_speed', 50)
        min_speed = edge_data.get('min_speed', 30)
        
        # Calcular velocidad basada en el tipo de vehículo
        base_speed = vehicle["speed_factor"] * max_speed * 0.0001
        
        # Simular congestión aleatoria
        congestion_factor = random.uniform(0.5, 1.0)
        current_speed = base_speed * congestion_factor
        
        # Registrar métricas
        self.vehicle_metrics[vehicle_id]["speeds"].append(current_speed * 10000)  # Convertir a unidades más legibles
        
        # Calcular aceleración (diferencia con velocidad anterior)
        if len(self.vehicle_metrics[vehicle_id]["speeds"]) > 1:
            prev_speed = self.vehicle_metrics[vehicle_id]["speeds"][-2]
            acceleration = current_speed * 10000 - prev_speed
            self.vehicle_metrics[vehicle_id]["accelerations"].append(acceleration)
        
        # Actualizar progreso
        vehicle["progress"] += current_speed
        
        # Obtener coordenadas
        current_lat = float(self.street_graph.nodes[current_node]['lat'])
        current_lon = float(self.street_graph.nodes[current_node]['lon'])
        next_lat = float(self.street_graph.nodes[next_node]['lat'])
        next_lon = float(self.street_graph.nodes[next_node]['lon'])
        
        if vehicle["progress"] >= 1.0:
            # Llegó al siguiente nodo
            vehicle["previous_node"] = current_node
            vehicle["current_node"] = next_node
            vehicle["next_node"] = None
            vehicle["lat"] = next_lat
            vehicle["lon"] = next_lon
            vehicle["progress"] = 0.0
            
            # Calcular distancia recorrida
            distance = self._calculate_distance(current_lat, current_lon, next_lat, next_lon)
            vehicle["distance_traveled"] += distance
            self.vehicle_metrics[vehicle_id]["distances"].append(distance)
            
        else:
            # Interpolación lineal
            vehicle["lat"] = current_lat + (next_lat - current_lat) * vehicle["progress"]
            vehicle["lon"] = current_lon + (next_lon - current_lon) * vehicle["progress"]
        
        # Registrar posición
        self.vehicle_metrics[vehicle_id]["positions"].append((vehicle["lat"], vehicle["lon"]))
        
        # Actualizar métricas del vehículo
        vehicle["max_speed_reached"] = max(vehicle["max_speed_reached"], current_speed * 10000)
        if len(self.vehicle_metrics[vehicle_id]["speeds"]) > 0:
            vehicle["average_speed"] = np.mean(self.vehicle_metrics[vehicle_id]["speeds"])
    
    def run_simulation(self, steps=SIMULATION_STEPS):
        """Ejecuta la simulación por un número determinado de pasos"""
        print(f"🚗 Iniciando simulación con {self.num_vehicles} vehículos por {steps} pasos...")
        
        self.initialize_vehicles()
        
        for step in range(steps):
            self.current_step = step
            
            # Actualizar todos los vehículos
            for vehicle_id in self.vehicles.keys():
                self.update_vehicle_position(vehicle_id)
            
            # Recopilar datos cada 10 pasos para análisis
            if step % 10 == 0:
                step_data = {
                    "step": step,
                    "vehicles": {}
                }
                
                for vid, vehicle in self.vehicles.items():
                    step_data["vehicles"][vid] = {
                        "type": vehicle["type"],
                        "speed": self.vehicle_metrics[vid]["speeds"][-1] if self.vehicle_metrics[vid]["speeds"] else 0,
                        "distance_traveled": vehicle["distance_traveled"],
                        "average_speed": vehicle["average_speed"],
                        "max_speed": vehicle["max_speed_reached"],
                        "position": (vehicle["lat"], vehicle["lon"])
                    }
                
                self.simulation_data.append(step_data)
            
            if step % 100 == 0:
                print(f"   Paso {step}/{steps} completado...")
        
        print("✅ Simulación completada!")
    
    def get_behavior_metrics(self):
        """Extrae métricas de comportamiento de todos los vehículos"""
        metrics_by_type = {}
        
        for vehicle_type in VEHICLE_TYPES.keys():
            metrics_by_type[vehicle_type] = {
                "speeds": [],
                "distances": [],
                "accelerations": [],
                "route_changes": [],
                "average_speeds": [],
                "max_speeds": []
            }
        
        for vid, vehicle in self.vehicles.items():
            vtype = vehicle["type"]
            metrics = self.vehicle_metrics[vid]
            
            metrics_by_type[vtype]["speeds"].extend(metrics["speeds"])
            metrics_by_type[vtype]["distances"].extend(metrics["distances"])
            metrics_by_type[vtype]["accelerations"].extend(metrics["accelerations"])
            metrics_by_type[vtype]["route_changes"].append(metrics["route_changes"])
            metrics_by_type[vtype]["average_speeds"].append(vehicle["average_speed"])
            metrics_by_type[vtype]["max_speeds"].append(vehicle["max_speed_reached"])
        
        return metrics_by_type

class BootstrapAnalyzer:
    """Analizador estadístico con bootstrapping"""
    
    def __init__(self, data, n_bootstrap=BOOTSTRAP_SAMPLES):
        self.data = data
        self.n_bootstrap = n_bootstrap
        self.confidence_level = CONFIDENCE_LEVEL
    
    def bootstrap_statistic(self, statistic_func, data_array):
        """Realiza bootstrapping para una estadística específica"""
        if len(data_array) == 0:
            return []
        
        bootstrap_stats = []
        data_array = np.array(data_array)
        
        for _ in range(self.n_bootstrap):
            # Muestreo con reemplazo
            bootstrap_sample = np.random.choice(data_array, size=len(data_array), replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        return np.array(bootstrap_stats)
    
    def confidence_interval(self, bootstrap_stats):
        """Calcula el intervalo de confianza"""
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return np.percentile(bootstrap_stats, [lower_percentile, upper_percentile])
    
    def analyze_metric(self, metric_name, vehicle_types=None):
        """Analiza una métrica específica con bootstrapping"""
        if vehicle_types is None:
            vehicle_types = list(self.data.keys())
        
        results = {}
        
        for vtype in vehicle_types:
            if vtype not in self.data or metric_name not in self.data[vtype]:
                continue
                
            data_array = self.data[vtype][metric_name]
            if len(data_array) == 0:
                continue
            
            # Estadísticas básicas
            original_mean = np.mean(data_array)
            original_std = np.std(data_array)
            original_median = np.median(data_array)
            
            # Bootstrap para la media
            bootstrap_means = self.bootstrap_statistic(np.mean, data_array)
            mean_ci = self.confidence_interval(bootstrap_means)
            
            # Bootstrap para la mediana
            bootstrap_medians = self.bootstrap_statistic(np.median, data_array)
            median_ci = self.confidence_interval(bootstrap_medians)
            
            # Bootstrap para la desviación estándar
            bootstrap_stds = self.bootstrap_statistic(np.std, data_array)
            std_ci = self.confidence_interval(bootstrap_stds)
            
            results[vtype] = {
                "original_stats": {
                    "mean": original_mean,
                    "std": original_std,
                    "median": original_median,
                    "min": np.min(data_array),
                    "max": np.max(data_array),
                    "n_samples": len(data_array)
                },
                "bootstrap_results": {
                    "mean_ci": mean_ci,
                    "median_ci": median_ci,
                    "std_ci": std_ci,
                    "bootstrap_means": bootstrap_means,
                    "bootstrap_medians": bootstrap_medians,
                    "bootstrap_stds": bootstrap_stds
                }
            }
        
        return results

class HypothesisTestManager:
    """Gestor de pruebas de hipótesis"""
    
    def __init__(self, data, alpha=0.05):
        self.data = data
        self.alpha = alpha
        self.test_results = {}
    
    def test_normality(self, metric_name, vehicle_types=None):
        """Pruebas de normalidad para las métricas"""
        if vehicle_types is None:
            vehicle_types = list(self.data.keys())
        
        results = {}
        
        for vtype in vehicle_types:
            if vtype not in self.data or metric_name not in self.data[vtype]:
                continue
                
            data_array = np.array(self.data[vtype][metric_name])
            if len(data_array) < 3:
                continue
            
            # Shapiro-Wilk test
            try:
                shapiro_stat, shapiro_p = shapiro(data_array)
            except:
                shapiro_stat, shapiro_p = np.nan, np.nan
            
            # Kolmogorov-Smirnov test contra distribución normal
            try:
                # Normalizar datos
                normalized_data = (data_array - np.mean(data_array)) / np.std(data_array)
                ks_stat, ks_p = kstest(normalized_data, 'norm')
            except:
                ks_stat, ks_p = np.nan, np.nan
            
            # Anderson-Darling test
            try:
                ad_result = anderson(data_array, dist='norm')
                ad_stat = ad_result.statistic
                ad_critical_values = ad_result.critical_values
                ad_significance_levels = ad_result.significance_level
            except:
                ad_stat = np.nan
                ad_critical_values = []
                ad_significance_levels = []
            
            results[vtype] = {
                "shapiro": {"statistic": shapiro_stat, "p_value": shapiro_p, "is_normal": shapiro_p > self.alpha},
                "ks": {"statistic": ks_stat, "p_value": ks_p, "is_normal": ks_p > self.alpha},
                "anderson": {
                    "statistic": ad_stat, 
                    "critical_values": ad_critical_values,
                    "significance_levels": ad_significance_levels
                }
            }
        
        return results
    
    def compare_vehicle_types(self, metric_name, type1, type2):
        """Compara dos tipos de vehículos para una métrica"""
        if (type1 not in self.data or type2 not in self.data or 
            metric_name not in self.data[type1] or metric_name not in self.data[type2]):
            return None
        
        data1 = np.array(self.data[type1][metric_name])
        data2 = np.array(self.data[type2][metric_name])
        
        if len(data1) < 3 or len(data2) < 3:
            return None
        
        # Prueba t de Student (asume normalidad)
        try:
            t_stat, t_p = stats.ttest_ind(data1, data2)
        except:
            t_stat, t_p = np.nan, np.nan
        
        # Prueba de Mann-Whitney U (no paramétrica)
        try:
            u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        except:
            u_stat, u_p = np.nan, np.nan
        
        # Prueba de Kolmogorov-Smirnov
        try:
            ks_stat, ks_p = stats.ks_2samp(data1, data2)
        except:
            ks_stat, ks_p = np.nan, np.nan
        
        return {
            "comparison": f"{type1} vs {type2}",
            "metric": metric_name,
            "descriptive": {
                f"mean_{type1}": np.mean(data1),
                f"mean_{type2}": np.mean(data2),
                f"std_{type1}": np.std(data1),
                f"std_{type2}": np.std(data2),
                "mean_difference": np.mean(data1) - np.mean(data2)
            },
            "tests": {
                "t_test": {"statistic": t_stat, "p_value": t_p, "significant": t_p < self.alpha},
                "mannwhitney": {"statistic": u_stat, "p_value": u_p, "significant": u_p < self.alpha},
                "ks_test": {"statistic": ks_stat, "p_value": ks_p, "significant": ks_p < self.alpha}
            }
        }
    
    def anova_vehicle_types(self, metric_name):
        """ANOVA de una vía para comparar todos los tipos de vehículos"""
        groups = []
        group_names = []
        
        for vtype in self.data.keys():
            if metric_name in self.data[vtype] and len(self.data[vtype][metric_name]) > 0:
                groups.append(self.data[vtype][metric_name])
                group_names.append(vtype)
        
        if len(groups) < 2:
            return None
        
        # ANOVA de una vía
        try:
            f_stat, f_p = stats.f_oneway(*groups)
        except:
            f_stat, f_p = np.nan, np.nan
        
        # Kruskal-Wallis (versión no paramétrica de ANOVA)
        try:
            kw_stat, kw_p = stats.kruskal(*groups)
        except:
            kw_stat, kw_p = np.nan, np.nan
        
        return {
            "metric": metric_name,
            "groups": group_names,
            "group_means": [np.mean(group) for group in groups],
            "tests": {
                "anova": {"statistic": f_stat, "p_value": f_p, "significant": f_p < self.alpha},
                "kruskal_wallis": {"statistic": kw_stat, "p_value": kw_p, "significant": kw_p < self.alpha}
            }
        }

class VehicleBehaviorAnalysisVisualizer:
    """Visualizador para análisis de comportamiento de vehículos"""
    
    def __init__(self, save_path="experiments/vehicle_behavior_results"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        # Configurar estilo
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        sns.set_style("whitegrid")
    
    def plot_bootstrap_distributions(self, bootstrap_results, metric_name):
        """Visualiza las distribuciones bootstrap"""
        n_types = len(bootstrap_results)
        fig, axes = plt.subplots(2, (n_types + 1) // 2, figsize=(15, 10))
        if n_types == 1:
            axes = [axes]
        elif n_types <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Distribuciones Bootstrap - {metric_name}', fontsize=16, fontweight='bold')
        
        for i, (vtype, results) in enumerate(bootstrap_results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            bootstrap_means = results["bootstrap_results"]["bootstrap_means"]
            original_mean = results["original_stats"]["mean"]
            mean_ci = results["bootstrap_results"]["mean_ci"]
            
            # Histograma de las medias bootstrap
            ax.hist(bootstrap_means, bins=30, alpha=0.7, color=VEHICLE_TYPES[vtype]["color"], 
                   density=True, label=f'Bootstrap means')
            
            # Línea vertical para la media original
            ax.axvline(original_mean, color='red', linestyle='--', linewidth=2, 
                      label=f'Media original: {original_mean:.3f}')
            
            # Intervalo de confianza
            ax.axvspan(mean_ci[0], mean_ci[1], alpha=0.3, color='gray', 
                      label=f'IC {int(CONFIDENCE_LEVEL*100)}%: [{mean_ci[0]:.3f}, {mean_ci[1]:.3f}]')
            
            ax.set_title(f'Tipo: {vtype}', fontweight='bold')
            ax.set_xlabel('Media Bootstrap')
            ax.set_ylabel('Densidad')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Ocultar axes no utilizados
        for i in range(len(bootstrap_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/bootstrap_distributions_{metric_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metric_comparison(self, data, metric_name):
        """Gráfico de comparación de métricas entre tipos de vehículos"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Análisis Comparativo - {metric_name}', fontsize=16, fontweight='bold')
        
        # Preparar datos para visualización
        plot_data = []
        for vtype, metrics in data.items():
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                for value in metrics[metric_name]:
                    plot_data.append({"tipo": vtype, "valor": value})
        
        df = pd.DataFrame(plot_data)
        
        if df.empty:
            print(f"No hay datos para visualizar en {metric_name}")
            plt.close()
            return
        
        # 1. Box plot
        sns.boxplot(data=df, x="tipo", y="valor", ax=axes[0,0])
        axes[0,0].set_title('Distribución por Tipo (Box Plot)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Violin plot
        sns.violinplot(data=df, x="tipo", y="valor", ax=axes[0,1])
        axes[0,1].set_title('Distribución por Tipo (Violin Plot)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Histograma superpuesto
        for vtype in df['tipo'].unique():
            subset = df[df['tipo'] == vtype]['valor']
            axes[1,0].hist(subset, alpha=0.6, label=vtype, bins=20, 
                          color=VEHICLE_TYPES.get(vtype, {"color": "gray"})["color"])
        axes[1,0].set_title('Histogramas Superpuestos')
        axes[1,0].set_xlabel(metric_name)
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].legend()
        
        # 4. Q-Q plots
        unique_types = df['tipo'].unique()
        if len(unique_types) >= 2:
            type1, type2 = unique_types[0], unique_types[1]
            data1 = df[df['tipo'] == type1]['valor'].values
            data2 = df[df['tipo'] == type2]['valor'].values
            
            # Q-Q plot entre dos tipos
            stats.probplot(data1, dist="norm", plot=axes[1,1])
            axes[1,1].get_lines()[0].set_markerfacecolor(VEHICLE_TYPES.get(type1, {"color": "blue"})["color"])
            axes[1,1].get_lines()[0].set_label(type1)
            
            stats.probplot(data2, dist="norm", plot=axes[1,1])
            axes[1,1].get_lines()[2].set_markerfacecolor(VEHICLE_TYPES.get(type2, {"color": "red"})["color"])
            axes[1,1].get_lines()[2].set_label(type2)
            
            axes[1,1].set_title(f'Q-Q Plot Normal: {type1} vs {type2}')
            axes[1,1].legend()
        else:
            axes[1,1].text(0.5, 0.5, 'Insuficientes tipos\npara Q-Q plot', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/metric_comparison_{metric_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hypothesis_test_results(self, test_results, metric_name):
        """Visualiza resultados de pruebas de hipótesis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Resultados de Pruebas de Hipótesis - {metric_name}', fontsize=16, fontweight='bold')
        
        # 1. Resultados de normalidad
        if 'normality' in test_results:
            normality_results = test_results['normality']
            types = list(normality_results.keys())
            shapiro_pvals = [normality_results[t]['shapiro']['p_value'] for t in types if not np.isnan(normality_results[t]['shapiro']['p_value'])]
            ks_pvals = [normality_results[t]['ks']['p_value'] for t in types if not np.isnan(normality_results[t]['ks']['p_value'])]
            
            if shapiro_pvals and ks_pvals:
                x_pos = np.arange(len(types))
                width = 0.35
                
                axes[0,0].bar(x_pos - width/2, shapiro_pvals, width, label='Shapiro-Wilk', alpha=0.7)
                axes[0,0].bar(x_pos + width/2, ks_pvals, width, label='Kolmogorov-Smirnov', alpha=0.7)
                axes[0,0].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
                axes[0,0].set_xlabel('Tipo de Vehículo')
                axes[0,0].set_ylabel('p-valor')
                axes[0,0].set_title('Pruebas de Normalidad')
                axes[0,0].set_xticks(x_pos)
                axes[0,0].set_xticklabels(types, rotation=45)
                axes[0,0].legend()
        
        # 2. Comparaciones por pares
        if 'pairwise' in test_results:
            pairwise_results = test_results['pairwise']
            comparisons = list(pairwise_results.keys())
            t_test_pvals = [pairwise_results[c]['tests']['t_test']['p_value'] for c in comparisons 
                           if not np.isnan(pairwise_results[c]['tests']['t_test']['p_value'])]
            mw_pvals = [pairwise_results[c]['tests']['mannwhitney']['p_value'] for c in comparisons 
                       if not np.isnan(pairwise_results[c]['tests']['mannwhitney']['p_value'])]
            
            if t_test_pvals and mw_pvals:
                x_pos = np.arange(len(comparisons))
                width = 0.35
                
                axes[0,1].bar(x_pos - width/2, t_test_pvals, width, label='t-test', alpha=0.7)
                axes[0,1].bar(x_pos + width/2, mw_pvals, width, label='Mann-Whitney U', alpha=0.7)
                axes[0,1].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
                axes[0,1].set_xlabel('Comparación')
                axes[0,1].set_ylabel('p-valor')
                axes[0,1].set_title('Comparaciones por Pares')
                axes[0,1].set_xticks(x_pos)
                axes[0,1].set_xticklabels([c.replace(' vs ', '\nvs\n') for c in comparisons], rotation=0, fontsize=8)
                axes[0,1].legend()
        
        # 3. ANOVA
        if 'anova' in test_results:
            anova_result = test_results['anova']
            if anova_result:
                tests = ['ANOVA', 'Kruskal-Wallis']
                p_values = [anova_result['tests']['anova']['p_value'], 
                           anova_result['tests']['kruskal_wallis']['p_value']]
                significant = [anova_result['tests']['anova']['significant'], 
                              anova_result['tests']['kruskal_wallis']['significant']]
                
                colors = ['green' if sig else 'red' for sig in significant]
                bars = axes[1,0].bar(tests, p_values, color=colors, alpha=0.7)
                axes[1,0].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
                axes[1,0].set_ylabel('p-valor')
                axes[1,0].set_title('ANOVA - Comparación Global')
                axes[1,0].legend()
                
                # Añadir valores de p en las barras
                for bar, pval in zip(bars, p_values):
                    height = bar.get_height()
                    axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                  f'p={pval:.4f}', ha='center', va='bottom')
        
        # 4. Resumen de significancia
        significance_summary = {}
        if 'pairwise' in test_results:
            for comparison, result in test_results['pairwise'].items():
                significance_summary[comparison] = {
                    't_test': result['tests']['t_test']['significant'],
                    'mann_whitney': result['tests']['mannwhitney']['significant']
                }
        
        if significance_summary:
            summary_data = []
            for comp, tests in significance_summary.items():
                for test_name, is_sig in tests.items():
                    summary_data.append({
                        'Comparación': comp,
                        'Test': test_name,
                        'Significativo': 'Sí' if is_sig else 'No'
                    })
            
            df_summary = pd.DataFrame(summary_data)
            pivot_summary = df_summary.pivot(index='Comparación', columns='Test', values='Significativo')
            
            # Convertir a valores numéricos para el heatmap
            pivot_numeric = pivot_summary.replace({'Sí': 1, 'No': 0})
            
            im = axes[1,1].imshow(pivot_numeric.values, cmap='RdYlGn', aspect='auto')
            axes[1,1].set_xticks(range(len(pivot_numeric.columns)))
            axes[1,1].set_yticks(range(len(pivot_numeric.index)))
            axes[1,1].set_xticklabels(pivot_numeric.columns)
            axes[1,1].set_yticklabels(pivot_numeric.index, fontsize=8)
            axes[1,1].set_title('Mapa de Significancia')
            
            # Añadir texto en celdas
            for i in range(len(pivot_numeric.index)):
                for j in range(len(pivot_numeric.columns)):
                    text = pivot_summary.iloc[i, j]
                    axes[1,1].text(j, i, text, ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/hypothesis_tests_{metric_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, all_results):
        """Crea un reporte visual comprehensivo"""
        # Métricas disponibles
        metrics = ["speeds", "distances", "accelerations", "average_speeds", "max_speeds"]
        
        # Crear un dashboard resumen
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('Dashboard de Análisis de Comportamiento Vehicular', fontsize=18, fontweight='bold')
        
        # 1. Comparación de velocidades promedio
        if "speeds" in all_results['data']:
            speed_means = {}
            for vtype, data in all_results['data'].items():
                if "speeds" in data and len(data["speeds"]) > 0:
                    speed_means[vtype] = np.mean(data["speeds"])
            
            if speed_means:
                types = list(speed_means.keys())
                means = list(speed_means.values())
                colors = [VEHICLE_TYPES.get(t, {"color": "gray"})["color"] for t in types]
                
                bars = axes[0,0].bar(types, means, color=colors, alpha=0.7)
                axes[0,0].set_title('Velocidad Promedio por Tipo de Vehículo')
                axes[0,0].set_ylabel('Velocidad')
                axes[0,0].tick_params(axis='x', rotation=45)
                
                # Añadir valores en las barras
                for bar, mean in zip(bars, means):
                    height = bar.get_height()
                    axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                  f'{mean:.2f}', ha='center', va='bottom')
        
        # 2. Distribución de distancias recorridas
        if "distances" in all_results['data']:
            distance_data = []
            for vtype, data in all_results['data'].items():
                if "distances" in data:
                    for dist in data["distances"]:
                        distance_data.append({"tipo": vtype, "distancia": dist})
            
            if distance_data:
                df_dist = pd.DataFrame(distance_data)
                sns.boxplot(data=df_dist, x="tipo", y="distancia", ax=axes[0,1])
                axes[0,1].set_title('Distribución de Distancias Recorridas')
                axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Análisis de aceleraciones
        if "accelerations" in all_results['data']:
            accel_means = {}
            accel_stds = {}
            for vtype, data in all_results['data'].items():
                if "accelerations" in data and len(data["accelerations"]) > 0:
                    accel_means[vtype] = np.mean(data["accelerations"])
                    accel_stds[vtype] = np.std(data["accelerations"])
            
            if accel_means:
                types = list(accel_means.keys())
                means = list(accel_means.values())
                stds = list(accel_stds.values())
                colors = [VEHICLE_TYPES.get(t, {"color": "gray"})["color"] for t in types]
                
                axes[1,0].errorbar(types, means, yerr=stds, fmt='o', capsize=5, capthick=2, 
                                  color='black', markersize=8)
                for i, (t, m, c) in enumerate(zip(types, means, colors)):
                    axes[1,0].scatter(i, m, color=c, s=100, alpha=0.7)
                
                axes[1,0].set_title('Aceleración Promedio ± DE por Tipo')
                axes[1,0].set_ylabel('Aceleración')
                axes[1,0].tick_params(axis='x', rotation=45)
                axes[1,0].grid(True, alpha=0.3)
        
        # 4. Intervalos de confianza bootstrap para velocidades
        if "speeds" in all_results.get('bootstrap', {}):
            bootstrap_speeds = all_results['bootstrap']['speeds']
            types = list(bootstrap_speeds.keys())
            means = [bootstrap_speeds[t]['original_stats']['mean'] for t in types]
            cis = [bootstrap_speeds[t]['bootstrap_results']['mean_ci'] for t in types]
            colors = [VEHICLE_TYPES.get(t, {"color": "gray"})["color"] for t in types]
            
            x_pos = np.arange(len(types))
            for i, (mean, ci, color) in enumerate(zip(means, cis, colors)):
                axes[1,1].errorbar(i, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], 
                                  fmt='o', capsize=5, capthick=2, color=color, markersize=8)
            
            axes[1,1].set_xticks(x_pos)
            axes[1,1].set_xticklabels(types, rotation=45)
            axes[1,1].set_title(f'Intervalos de Confianza Bootstrap {int(CONFIDENCE_LEVEL*100)}% - Velocidades')
            axes[1,1].set_ylabel('Velocidad')
            axes[1,1].grid(True, alpha=0.3)
        
        # 5. Resumen de pruebas de normalidad
        if "normality_tests" in all_results:
            normality_data = []
            for metric, tests in all_results['normality_tests'].items():
                for vtype, result in tests.items():
                    normality_data.append({
                        "Métrica": metric,
                        "Tipo": vtype,
                        "Shapiro_pval": result['shapiro']['p_value'],
                        "Normal": result['shapiro']['is_normal']
                    })
            
            if normality_data:
                df_norm = pd.DataFrame(normality_data)
                df_norm = df_norm[~df_norm['Shapiro_pval'].isna()]
                
                if not df_norm.empty:
                    pivot_norm = df_norm.pivot(index='Tipo', columns='Métrica', values='Normal')
                    pivot_norm_numeric = pivot_norm.astype(int)
                    
                    im = axes[2,0].imshow(pivot_norm_numeric.values, cmap='RdYlGn', aspect='auto')
                    axes[2,0].set_xticks(range(len(pivot_norm.columns)))
                    axes[2,0].set_yticks(range(len(pivot_norm.index)))
                    axes[2,0].set_xticklabels(pivot_norm.columns, rotation=45)
                    axes[2,0].set_yticklabels(pivot_norm.index)
                    axes[2,0].set_title('Pruebas de Normalidad (Verde=Normal, Rojo=No Normal)')
                    
                    # Añadir texto
                    for i in range(len(pivot_norm.index)):
                        for j in range(len(pivot_norm.columns)):
                            if not pd.isna(pivot_norm.iloc[i, j]):
                                text = "✓" if pivot_norm.iloc[i, j] else "✗"
                                axes[2,0].text(j, i, text, ha="center", va="center", 
                                             color="white", fontweight='bold', fontsize=14)
        
        # 6. Resumen estadístico general
        summary_stats = {}
        for vtype in VEHICLE_TYPES.keys():
            if vtype in all_results['data']:
                vtype_data = all_results['data'][vtype]
                summary_stats[vtype] = {
                    "n_speeds": len(vtype_data.get('speeds', [])),
                    "mean_speed": np.mean(vtype_data.get('speeds', [0])) if vtype_data.get('speeds') else 0,
                    "n_distances": len(vtype_data.get('distances', [])),
                    "total_distance": np.sum(vtype_data.get('distances', [0])) if vtype_data.get('distances') else 0
                }
        
        if summary_stats:
            summary_text = "Resumen Estadístico:\n\n"
            for vtype, stats in summary_stats.items():
                summary_text += f"{vtype.capitalize()}:\n"
                summary_text += f"  • Observaciones de velocidad: {stats['n_speeds']}\n"
                summary_text += f"  • Velocidad promedio: {stats['mean_speed']:.2f}\n"
                summary_text += f"  • Distancia total: {stats['total_distance']:.1f}m\n\n"
            
            axes[2,1].text(0.05, 0.95, summary_text, transform=axes[2,1].transAxes, 
                          fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[2,1].set_xlim(0, 1)
            axes[2,1].set_ylim(0, 1)
            axes[2,1].axis('off')
            axes[2,1].set_title('Resumen Estadístico')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/comprehensive_dashboard.png", dpi=300, bbox_inches='tight')
        plt.show()

def run_complete_analysis():
    """Función principal para ejecutar el análisis completo"""
    print("="*80)
    print("🚗 ANÁLISIS COMPLETO DE COMPORTAMIENTO VEHICULAR 🚗")
    print("="*80)
    
    # 1. Ejecutar simulación
    print("\n1️⃣ Ejecutando simulación...")
    simulator = VehicleBehaviorSimulator(num_vehicles=NUM_VEHICLES_TEST)
    simulator.run_simulation(steps=SIMULATION_STEPS)
    
    # 2. Extraer métricas
    print("\n2️⃣ Extrayendo métricas de comportamiento...")
    behavior_metrics = simulator.get_behavior_metrics()
    
    # 3. Análisis Bootstrap
    print("\n3️⃣ Realizando análisis Bootstrap...")
    bootstrap_analyzer = BootstrapAnalyzer(behavior_metrics)
    
    bootstrap_results = {}
    metrics_to_analyze = ["speeds", "distances", "accelerations", "average_speeds", "max_speeds"]
    
    for metric in metrics_to_analyze:
        print(f"   Analizando {metric}...")
        bootstrap_results[metric] = bootstrap_analyzer.analyze_metric(metric)
    
    # 4. Pruebas de hipótesis
    print("\n4️⃣ Ejecutando pruebas de hipótesis...")
    hypothesis_tester = HypothesisTestManager(behavior_metrics)
    
    # Pruebas de normalidad
    normality_tests = {}
    for metric in metrics_to_analyze:
        normality_tests[metric] = hypothesis_tester.test_normality(metric)
    
    # Comparaciones por pares
    vehicle_types = list(VEHICLE_TYPES.keys())
    pairwise_comparisons = {}
    
    for metric in ["speeds", "average_speeds"]:  # Métricas más importantes
        pairwise_comparisons[metric] = {}
        for i in range(len(vehicle_types)):
            for j in range(i+1, len(vehicle_types)):
                type1, type2 = vehicle_types[i], vehicle_types[j]
                comparison_key = f"{type1} vs {type2}"
                result = hypothesis_tester.compare_vehicle_types(metric, type1, type2)
                if result:
                    pairwise_comparisons[metric][comparison_key] = result
    
    # ANOVA
    anova_results = {}
    for metric in metrics_to_analyze:
        anova_results[metric] = hypothesis_tester.anova_vehicle_types(metric)
    
    # 5. Visualización
    print("\n5️⃣ Generando visualizaciones...")
    visualizer = VehicleBehaviorAnalysisVisualizer()
    
    # Visualizaciones bootstrap
    for metric in ["speeds", "average_speeds", "distances"]:
        if metric in bootstrap_results and bootstrap_results[metric]:
            print(f"   Generando gráficos bootstrap para {metric}...")
            visualizer.plot_bootstrap_distributions(bootstrap_results[metric], metric)
    
    # Comparaciones de métricas
    for metric in ["speeds", "distances", "accelerations"]:
        print(f"   Generando comparaciones para {metric}...")
        visualizer.plot_metric_comparison(behavior_metrics, metric)
    
    # Resultados de pruebas de hipótesis
    for metric in ["speeds", "average_speeds"]:
        test_results = {
            'normality': normality_tests.get(metric, {}),
            'pairwise': pairwise_comparisons.get(metric, {}),
            'anova': anova_results.get(metric)
        }
        print(f"   Generando gráficos de pruebas de hipótesis para {metric}...")
        visualizer.plot_hypothesis_test_results(test_results, metric)
    
    # 6. Reporte comprehensivo
    print("\n6️⃣ Creando reporte comprehensivo...")
    all_results = {
        'data': behavior_metrics,
        'bootstrap': bootstrap_results,
        'normality_tests': normality_tests,
        'pairwise_comparisons': pairwise_comparisons,
        'anova_results': anova_results
    }
    
    visualizer.create_comprehensive_report(all_results)
    
    # 7. Guardar resultados
    print("\n7️⃣ Guardando resultados...")
    results_file = f"{visualizer.save_path}/analysis_results.json"
    
    # Convertir numpy arrays a listas para JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    all_results_json = convert_numpy(all_results)
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results_json, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Resultados guardados en: {results_file}")
    except Exception as e:
        print(f"   ⚠️ Error guardando resultados: {e}")
    
    # 8. Resumen de hallazgos
    print("\n" + "="*80)
    print("📊 RESUMEN DE HALLAZGOS")
    print("="*80)
    
    # Resumen de velocidades promedio
    print("\n🚀 Velocidades Promedio por Tipo de Vehículo:")
    for vtype in vehicle_types:
        if vtype in behavior_metrics and behavior_metrics[vtype]['average_speeds']:
            avg_speed = np.mean(behavior_metrics[vtype]['average_speeds'])
            print(f"   {vtype.capitalize()}: {avg_speed:.2f} unidades")
    
    # Resumen de normalidad
    print("\n📈 Pruebas de Normalidad (Shapiro-Wilk, α=0.05):")
    for metric in ["speeds", "average_speeds"]:
        if metric in normality_tests:
            print(f"\n   {metric.capitalize()}:")
            for vtype, test_result in normality_tests[metric].items():
                is_normal = test_result['shapiro']['is_normal']
                p_val = test_result['shapiro']['p_value']
                status = "✓ Normal" if is_normal else "✗ No Normal"
                print(f"     {vtype.capitalize()}: {status} (p={p_val:.4f})")
    
    # Resumen de comparaciones significativas
    print("\n🔍 Comparaciones Significativas (α=0.05):")
    for metric in ["speeds", "average_speeds"]:
        if metric in pairwise_comparisons:
            print(f"\n   {metric.capitalize()}:")
            for comparison, result in pairwise_comparisons[metric].items():
                t_test_sig = result['tests']['t_test']['significant']
                mw_sig = result['tests']['mannwhitney']['significant']
                if t_test_sig or mw_sig:
                    tests_sig = []
                    if t_test_sig:
                        tests_sig.append("t-test")
                    if mw_sig:
                        tests_sig.append("Mann-Whitney")
                    print(f"     {comparison}: Significativo en {', '.join(tests_sig)}")
    
    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("📁 Todos los archivos se guardaron en: experiments/vehicle_behavior_results/")
    print("="*80)

if __name__ == "__main__":
    # Configurar seeds para reproducibilidad
    np.random.seed(42)
    random.seed(42)
    
    # Ejecutar análisis completo
    run_complete_analysis()
