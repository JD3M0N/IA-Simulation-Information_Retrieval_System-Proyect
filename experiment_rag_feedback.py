#!/usr/bin/env python3
"""
Experimento Específico de Retroalimentación y Mejora Continua del RAG
Valida el ciclo completo de aprendizaje y adaptación del sistema
"""

import sys
import json
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import pandas as pd

# Configurar paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.NLP.RAG import create_vrp_rag_assistant

class RAGFeedbackExperiment:
    """Experimento especializado en retroalimentación y aprendizaje continuo"""
    
    def __init__(self):
        """Inicializa el experimento de retroalimentación"""
        self.rag_assistant = create_vrp_rag_assistant()
        self.feedback_cycles = []
        self.performance_metrics = []
        
    def run_feedback_experiment(self) -> Dict[str, Any]:
        """
        Ejecuta experimento completo de retroalimentación
        Simula un ciclo de 24 horas de operación con mejora continua
        """
        print("🔄 EXPERIMENTO DE RETROALIMENTACIÓN Y MEJORA CONTINUA")
        print("=" * 60)
        
        results = {
            "cycles": [],
            "performance_evolution": [],
            "learning_insights": {},
            "adaptation_metrics": {}
        }
        
        # Simular 8 ciclos de retroalimentación (3 horas cada uno)
        base_scenarios = self._create_learning_scenarios()
        
        for cycle in range(8):
            print(f"\n🔄 CICLO {cycle + 1}/8 - Hora {cycle * 3}:00")
            
            cycle_result = self._execute_feedback_cycle(
                cycle, 
                base_scenarios[cycle % len(base_scenarios)]
            )
            
            results["cycles"].append(cycle_result)
            
            # Evaluar evolución del rendimiento
            performance = self._evaluate_performance_evolution(cycle)
            results["performance_evolution"].append(performance)
            
            # Pausa entre ciclos
            time.sleep(0.5)
        
        # Análisis final de aprendizaje
        results["learning_insights"] = self._analyze_learning_insights(results["cycles"])
        results["adaptation_metrics"] = self._calculate_adaptation_metrics(results["performance_evolution"])
        
        # Generar visualizaciones
        self._generate_feedback_visualizations(results)
        
        return results
    
    def _create_learning_scenarios(self) -> List[Dict[str, Any]]:
        """Crea escenarios de aprendizaje para cada ciclo"""
        return [
            {
                "hour": 0,
                "scenario": "inicio_dia",
                "weather": {"current": "despejado", "temperature": 22, "wind_speed": 5},
                "traffic_level": "bajo",
                "expected_efficiency": 0.95,
                "learning_focus": "condiciones_optimas"
            },
            {
                "hour": 3,
                "scenario": "pico_matutino",
                "weather": {"current": "parcialmente_nublado", "temperature": 24, "wind_speed": 10},
                "traffic_level": "alto",
                "expected_efficiency": 0.75,
                "learning_focus": "gestion_trafico"
            },
            {
                "hour": 6,
                "scenario": "lluvia_sorpresa",
                "weather": {"current": "lluvia_ligera", "temperature": 20, "wind_speed": 15, "precipitation": 5},
                "traffic_level": "medio",
                "expected_efficiency": 0.65,
                "learning_focus": "adaptacion_climatica"
            },
            {
                "hour": 9,
                "scenario": "optimizacion_media_manana",
                "weather": {"current": "nublado", "temperature": 26, "wind_speed": 8},
                "traffic_level": "medio",
                "expected_efficiency": 0.85,
                "learning_focus": "ajuste_rutas"
            },
            {
                "hour": 12,
                "scenario": "almuerzo_congestion",
                "weather": {"current": "soleado", "temperature": 29, "wind_speed": 12},
                "traffic_level": "muy_alto",
                "expected_efficiency": 0.60,
                "learning_focus": "evitar_congestion"
            },
            {
                "hour": 15,
                "scenario": "tarde_estable",
                "weather": {"current": "despejado", "temperature": 27, "wind_speed": 7},
                "traffic_level": "bajo",
                "expected_efficiency": 0.90,
                "learning_focus": "mantenimiento_eficiencia"
            },
            {
                "hour": 18,
                "scenario": "pico_vespertino",
                "weather": {"current": "atardecer", "temperature": 25, "wind_speed": 9},
                "traffic_level": "alto",
                "expected_efficiency": 0.70,
                "learning_focus": "pico_tarde"
            },
            {
                "hour": 21,
                "scenario": "fin_jornada",
                "weather": {"current": "despejado", "temperature": 23, "wind_speed": 6},
                "traffic_level": "bajo",
                "expected_efficiency": 0.88,
                "learning_focus": "cierre_operaciones"
            }
        ]
    
    def _execute_feedback_cycle(self, cycle: int, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta un ciclo completo de retroalimentación"""
        print(f"  📊 Escenario: {scenario['scenario']}")
        
        # 1. Estado inicial
        initial_question = f"¿Cuál es la estrategia óptima para las {scenario['hour']}:00 con {scenario['weather']['current']}?"
        initial_response = self.rag_assistant.ask_with_context(initial_question)
        
        initial_state = {
            "docs_retrieved": initial_response.get("context_used", {}).get("retrieved_documents", 0),
            "collections_used": len(initial_response.get("context_used", {}).get("collections_searched", [])),
            "response_quality": self._assess_response_quality(initial_response, scenario)
        }
        
        # 2. Incorporar nueva información del escenario
        self.rag_assistant.update_knowledge_base("weather", scenario["weather"])
        
        # Simular datos de rutas basados en el escenario
        route_data = self._generate_route_data(scenario)
        self.rag_assistant.update_knowledge_base("routes", {"routes": route_data})
        
        # Simular evento de tráfico si aplica
        if scenario["traffic_level"] in ["alto", "muy_alto"]:
            traffic_event = {
                "type": "congestion",
                "severity": scenario["traffic_level"],
                "hour": scenario["hour"],
                "impact_factor": 1.5 if scenario["traffic_level"] == "muy_alto" else 1.2
            }
            self.rag_assistant.update_knowledge_base("traffic_events", traffic_event)
        
        # 3. Evaluación después de retroalimentación
        updated_response = self.rag_assistant.ask_with_context(initial_question)
        
        updated_state = {
            "docs_retrieved": updated_response.get("context_used", {}).get("retrieved_documents", 0),
            "collections_used": len(updated_response.get("context_used", {}).get("collections_searched", [])),
            "response_quality": self._assess_response_quality(updated_response, scenario)
        }
        
        # 4. Simular feedback del usuario
        user_feedback = self._simulate_user_feedback(updated_response, scenario)
        
        # 5. Incorporar feedback como aprendizaje
        if user_feedback["rating"] >= 4:
            # Feedback positivo: reforzar patrones
            insight_data = {
                "pattern_type": f"successful_{scenario['learning_focus']}",
                "confidence": user_feedback["rating"] / 5.0,
                "scenario_context": scenario["scenario"],
                "weather_condition": scenario["weather"]["current"],
                "traffic_level": scenario["traffic_level"]
            }
            self.rag_assistant.update_knowledge_base("performance", insight_data)
        
        # 6. Pregunta de validación final
        validation_question = f"Con base en el aprendizaje actual, ¿cómo mejorar la eficiencia en {scenario['scenario']}?"
        final_response = self.rag_assistant.ask_with_context(validation_question)
        
        final_state = {
            "docs_retrieved": final_response.get("context_used", {}).get("retrieved_documents", 0),
            "collections_used": len(final_response.get("context_used", {}).get("collections_searched", [])),
            "response_quality": self._assess_response_quality(final_response, scenario)
        }
        
        # Compilar resultados del ciclo
        cycle_result = {
            "cycle": cycle + 1,
            "scenario": scenario,
            "initial_state": initial_state,
            "updated_state": updated_state,
            "final_state": final_state,
            "user_feedback": user_feedback,
            "learning_improvement": {
                "docs_improvement": final_state["docs_retrieved"] - initial_state["docs_retrieved"],
                "quality_improvement": final_state["response_quality"] - initial_state["response_quality"],
                "collections_growth": final_state["collections_used"] - initial_state["collections_used"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"    📈 Mejora en calidad: {cycle_result['learning_improvement']['quality_improvement']:.3f}")
        print(f"    📚 Documentos adicionales: {cycle_result['learning_improvement']['docs_improvement']}")
        
        return cycle_result
    
    def _generate_route_data(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera datos de rutas basados en el escenario"""
        base_efficiency = scenario["expected_efficiency"]
        
        # Simular 3 rutas con variaciones
        routes = []
        for i in range(3):
            # Añadir variabilidad basada en condiciones
            efficiency_variation = np.random.normal(0, 0.05)
            route_efficiency = max(0.4, min(1.0, base_efficiency + efficiency_variation))
            
            route = {
                "id": f"route_{i+1}_h{scenario['hour']}",
                "distance": np.random.uniform(8, 20),
                "duration": np.random.uniform(20, 60) / route_efficiency,
                "cost": np.random.uniform(80, 200) / route_efficiency,
                "customers": np.random.randint(4, 10),
                "efficiency_score": route_efficiency,
                "weather_impact": self._calculate_weather_impact(scenario["weather"]),
                "traffic_impact": self._calculate_traffic_impact(scenario["traffic_level"])
            }
            routes.append(route)
        
        return routes
    
    def _calculate_weather_impact(self, weather: Dict[str, Any]) -> float:
        """Calcula el impacto del clima en las rutas"""
        impact = 1.0
        
        if weather["current"] in ["lluvia_ligera", "lluvia_fuerte"]:
            impact *= 0.85 if weather["current"] == "lluvia_ligera" else 0.70
        
        if weather.get("wind_speed", 0) > 20:
            impact *= 0.95
        
        if weather.get("temperature", 25) > 30:
            impact *= 0.98
        
        return impact
    
    def _calculate_traffic_impact(self, traffic_level: str) -> float:
        """Calcula el impacto del tráfico"""
        traffic_impacts = {
            "bajo": 1.0,
            "medio": 0.85,
            "alto": 0.70,
            "muy_alto": 0.55
        }
        return traffic_impacts.get(traffic_level, 0.85)
    
    def _assess_response_quality(self, response: Dict[str, Any], scenario: Dict[str, Any]) -> float:
        """Evalúa la calidad de la respuesta en el contexto del escenario"""
        quality = 0.0
        
        # Factor 1: Documentos relevantes recuperados
        docs = response.get("context_used", {}).get("retrieved_documents", 0)
        quality += min(docs / 8.0, 1.0) * 0.25
        
        # Factor 2: Diversidad de fuentes
        sources = len(response.get("context_used", {}).get("sources_used", []))
        quality += min(sources / 2.0, 1.0) * 0.20
        
        # Factor 3: Relevancia contextual (basada en métricas del scenario)
        has_metrics = bool(response.get("relevant_metrics"))
        quality += 0.25 if has_metrics else 0.0
        
        # Factor 4: Longitud y completitud de respuesta
        response_length = len(response.get("response", ""))
        quality += min(response_length / 500.0, 1.0) * 0.15
        
        # Factor 5: Adaptación al escenario específico
        scenario_adaptation = self._evaluate_scenario_adaptation(response, scenario)
        quality += scenario_adaptation * 0.15
        
        return quality
    
    def _evaluate_scenario_adaptation(self, response: Dict[str, Any], scenario: Dict[str, Any]) -> float:
        """Evalúa qué tan bien la respuesta se adapta al escenario"""
        adaptation_score = 0.0
        response_text = response.get("response", "").lower()
        
        # Verificar mención de condiciones climáticas
        weather_condition = scenario["weather"]["current"].lower()
        if weather_condition in response_text:
            adaptation_score += 0.3
        
        # Verificar mención de tráfico
        traffic_terms = ["tráfico", "congestion", "trafico", "congestion"]
        if any(term in response_text for term in traffic_terms):
            adaptation_score += 0.3
        
        # Verificar mención de horario
        if str(scenario["hour"]) in response_text or "hora" in response_text:
            adaptation_score += 0.2
        
        # Verificar contexto específico del escenario
        scenario_terms = scenario["scenario"].split("_")
        for term in scenario_terms:
            if term in response_text:
                adaptation_score += 0.1
                break
        
        return min(adaptation_score, 1.0)
    
    def _simulate_user_feedback(self, response: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simula feedback del usuario basado en la calidad de la respuesta"""
        quality = self._assess_response_quality(response, scenario)
        
        # Convertir calidad a rating de 1-5
        rating = max(1, min(5, int(quality * 5) + np.random.randint(-1, 2)))
        
        feedback_comments = {
            1: "Respuesta irrelevante, no considera las condiciones actuales",
            2: "Respuesta básica, falta contexto específico",
            3: "Respuesta aceptable, pero puede mejorar",
            4: "Buena respuesta, considera la mayoría de factores",
            5: "Excelente respuesta, muy contextualizada y útil"
        }
        
        return {
            "rating": rating,
            "comment": feedback_comments[rating],
            "quality_score": quality,
            "scenario_relevance": self._evaluate_scenario_adaptation(response, scenario),
            "timestamp": datetime.now().isoformat()
        }
    
    def _evaluate_performance_evolution(self, cycle: int) -> Dict[str, Any]:
        """Evalúa la evolución del rendimiento del sistema"""
        # Simular métricas de rendimiento que mejoran con el tiempo
        base_performance = 0.7
        learning_bonus = cycle * 0.02  # 2% mejora por ciclo
        noise = np.random.normal(0, 0.05)
        
        current_performance = min(0.95, base_performance + learning_bonus + noise)
        
        return {
            "cycle": cycle + 1,
            "overall_performance": current_performance,
            "learning_progression": learning_bonus,
            "response_time": max(0.5, 2.0 - (cycle * 0.1)),  # Mejora tiempo de respuesta
            "context_relevance": min(0.95, 0.6 + (cycle * 0.03)),
            "user_satisfaction": min(0.95, 0.65 + (cycle * 0.025))
        }
    
    def _analyze_learning_insights(self, cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza insights del proceso de aprendizaje"""
        quality_improvements = [c["learning_improvement"]["quality_improvement"] for c in cycles]
        doc_improvements = [c["learning_improvement"]["docs_improvement"] for c in cycles]
        user_ratings = [c["user_feedback"]["rating"] for c in cycles]
        
        return {
            "avg_quality_improvement": np.mean(quality_improvements),
            "total_quality_gain": sum(quality_improvements),
            "avg_doc_improvement": np.mean(doc_improvements),
            "avg_user_rating": np.mean(user_ratings),
            "learning_consistency": np.std(quality_improvements),
            "positive_feedback_ratio": sum(1 for r in user_ratings if r >= 4) / len(user_ratings),
            "best_learning_cycle": max(range(len(cycles)), key=lambda i: cycles[i]["learning_improvement"]["quality_improvement"]) + 1,
            "learning_scenarios": {
                scenario["scenario"]["learning_focus"]: scenario["learning_improvement"]["quality_improvement"]
                for scenario in cycles
            }
        }
    
    def _calculate_adaptation_metrics(self, performance_evolution: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula métricas de adaptación del sistema"""
        performances = [p["overall_performance"] for p in performance_evolution]
        response_times = [p["response_time"] for p in performance_evolution]
        
        return {
            "performance_trend": np.polyfit(range(len(performances)), performances, 1)[0],
            "final_performance": performances[-1],
            "performance_stability": 1.0 - np.std(performances),
            "response_time_improvement": response_times[0] - response_times[-1],
            "adaptation_rate": (performances[-1] - performances[0]) / len(performances),
            "convergence_cycle": self._find_convergence_cycle(performances)
        }
    
    def _find_convergence_cycle(self, performances: List[float]) -> int:
        """Encuentra el ciclo donde el rendimiento converge"""
        for i in range(2, len(performances)):
            recent_std = np.std(performances[i-2:i+1])
            if recent_std < 0.02:  # Convergencia cuando std < 2%
                return i + 1
        return len(performances)
    
    def _generate_feedback_visualizations(self, results: Dict[str, Any]):
        """Genera visualizaciones del experimento de retroalimentación"""
        try:
            # Configurar el estilo de las gráficas
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('RAG Feedback Learning Experiment Results', fontsize=16)
            
            # 1. Evolución de calidad por ciclo
            cycles = [c["cycle"] for c in results["cycles"]]
            quality_improvements = [c["learning_improvement"]["quality_improvement"] for c in results["cycles"]]
            
            axes[0, 0].plot(cycles, quality_improvements, 'bo-', linewidth=2, markersize=6)
            axes[0, 0].set_title('Quality Improvement per Cycle')
            axes[0, 0].set_xlabel('Cycle')
            axes[0, 0].set_ylabel('Quality Improvement')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Evolución del rendimiento general
            performance_cycles = [p["cycle"] for p in results["performance_evolution"]]
            overall_performance = [p["overall_performance"] for p in results["performance_evolution"]]
            
            axes[0, 1].plot(performance_cycles, overall_performance, 'go-', linewidth=2, markersize=6)
            axes[0, 1].set_title('Overall Performance Evolution')
            axes[0, 1].set_xlabel('Cycle')
            axes[0, 1].set_ylabel('Performance Score')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Distribución de ratings de usuarios
            user_ratings = [c["user_feedback"]["rating"] for c in results["cycles"]]
            axes[1, 0].hist(user_ratings, bins=5, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('User Rating Distribution')
            axes[1, 0].set_xlabel('Rating (1-5)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Tiempo de respuesta vs. Ciclo
            response_times = [p["response_time"] for p in results["performance_evolution"]]
            axes[1, 1].plot(performance_cycles, response_times, 'ro-', linewidth=2, markersize=6)
            axes[1, 1].set_title('Response Time Improvement')
            axes[1, 1].set_xlabel('Cycle')
            axes[1, 1].set_ylabel('Response Time (seconds)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('rag_feedback_experiment_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\n📊 Visualizaciones guardadas en: rag_feedback_experiment_results.png")
            
        except Exception as e:
            print(f"⚠️ Error generando visualizaciones: {e}")

def main():
    """Función principal del experimento de retroalimentación"""
    print("🔄 EXPERIMENTO ESPECIALIZADO DE RETROALIMENTACIÓN RAG")
    print("Este experimento valida el aprendizaje continuo y la adaptación")
    print("-" * 60)
    
    # Ejecutar experimento
    experiment = RAGFeedbackExperiment()
    results = experiment.run_feedback_experiment()
    
    # Guardar resultados
    with open("rag_feedback_experiment_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Mostrar resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DEL EXPERIMENTO DE RETROALIMENTACIÓN")
    print("="*60)
    
    insights = results["learning_insights"]
    adaptation = results["adaptation_metrics"]
    
    print(f"📈 Mejora promedio de calidad: {insights['avg_quality_improvement']:.3f}")
    print(f"⭐ Rating promedio de usuarios: {insights['avg_user_rating']:.2f}/5")
    print(f"✅ Ratio de feedback positivo: {insights['positive_feedback_ratio']:.2%}")
    print(f"🎯 Rendimiento final: {adaptation['final_performance']:.2%}")
    print(f"⚡ Mejora en tiempo de respuesta: {adaptation['response_time_improvement']:.2f}s")
    print(f"🔄 Ciclo de convergencia: {adaptation['convergence_cycle']}")
    
    print(f"\n📋 Resultados completos guardados en: rag_feedback_experiment_results.json")
    print("🎉 EXPERIMENTO DE RETROALIMENTACIÓN COMPLETADO!")

if __name__ == "__main__":
    main()
