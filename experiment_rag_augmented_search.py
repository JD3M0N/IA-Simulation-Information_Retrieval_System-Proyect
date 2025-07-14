#!/usr/bin/env python3
"""
Experimento de Búsqueda Aumentada y Respuesta Contextualizada
Valida la capacidad del RAG para generar respuestas mejoradas mediante búsqueda híbrida
"""

import sys
import json
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Configurar paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.NLP.RAG import create_vrp_rag_assistant
from src.SRI.VectorDatabase import VRPVectorDatabase
from src.SRI.VRPInformationRetrieval import VRPInformationRetrievalSystem

class AugmentedSearchExperiment:
    """Experimento para validar búsqueda aumentada y respuestas contextualizadas"""
    
    def __init__(self):
        """Inicializa el experimento de búsqueda aumentada"""
        self.rag_assistant = create_vrp_rag_assistant()
        self.vector_db = VRPVectorDatabase("augmented_search_cache")
        self.ir_system = VRPInformationRetrievalSystem("augmented_search_cache")
        
        # Preparar datos de referencia
        self._populate_reference_data()
        
    def _populate_reference_data(self):
        """Pobla la base de datos con información de referencia rica"""
        print("📚 Poblando base de datos con información de referencia...")
        
        # Datos meteorológicos históricos
        weather_data = [
            {
                "content": "El análisis histórico de La Habana muestra que las lluvias intensas (>10mm/h) reducen la velocidad promedio de vehículos en 35% y aumentan el tiempo de entrega en 45%. La visibilidad se reduce considerablemente en el Malecón debido a la brisa marina.",
                "metadata": {"type": "weather_analysis", "location": "habana", "impact_factor": "high"}
            },
            {
                "content": "Los vientos superiores a 25 km/h en La Habana afectan principalmente a vehículos de carga ligera. Las rutas costeras (Malecón, Primera) experimentan mayor impacto. Recomendación: usar vehículos más pesados en estas condiciones.",
                "metadata": {"type": "wind_analysis", "vehicle_type": "light_cargo", "routes": "coastal"}
            },
            {
                "content": "Temperaturas superiores a 32°C en La Habana aumentan el consumo de combustible en 8% y reducen la eficiencia del conductor en 12%. Horarios óptimos: 6:00-9:00 AM y 5:00-7:00 PM para evitar pico térmico.",
                "metadata": {"type": "temperature_analysis", "efficiency_impact": "medium"}
            }
        ]
        
        for weather in weather_data:
            self.vector_db.add_document("weather_data", weather["content"], weather["metadata"])
        
        # Análisis de rutas específicos
        route_analyses = [
            {
                "content": "Análisis de eficiencia de rutas en Centro Habana: Las calles estrechas (Neptuno, San Lázaro) limitan vehículos >3.5 toneladas. Ruta óptima para distribución: Zanja-Belascoaín-Carlos III. Tiempo promedio: 45min, 12 paradas máximo.",
                "metadata": {"area": "centro_habana", "vehicle_limit": "3.5t", "optimal_stops": "12"}
            },
            {
                "content": "Optimización Vedado-Miramar: Autopista ideal para cargas pesadas. Alternativa por Línea para distribución local. Puente del Almendares: cuello de botella 7-9 AM y 5-7 PM. Eficiencia: 85% fuera de pico, 65% en pico.",
                "metadata": {"area": "vedado_miramar", "bottleneck": "almendares_bridge", "efficiency_peak": "65%"}
            },
            {
                "content": "Distribución en Habana Vieja: Acceso vehicular restringido zona patrimonial. Usar vehículos eléctricos pequeños. Horarios permitidos: 6-10 AM. Puntos de carga: Plaza de Armas, Plaza Vieja. Capacidad máxima: 500kg por viaje.",
                "metadata": {"area": "habana_vieja", "vehicle_type": "electric_small", "max_capacity": "500kg"}
            }
        ]
        
        for route in route_analyses:
            self.vector_db.add_document("route_analysis", route["content"], route["metadata"])
        
        # Eventos de tráfico y patrones
        traffic_patterns = [
            {
                "content": "Patrón de tráfico La Habana: Picos principales 7:30-9:00 AM (intensidad 85%) y 5:00-6:30 PM (intensidad 90%). Sábados: tráfico reducido 40%. Domingos: óptimo para entregas pesadas. Evitar: 23 y 12, Línea en horas pico.",
                "metadata": {"pattern_type": "daily", "peak_intensity": "90%", "optimal_day": "sunday"}
            },
            {
                "content": "Eventos especiales en La Habana impactan rutas: Carnaval (julio/agosto): cerrar Malecón. Feria del Libro (febrero): evitar Fortaleza. Maratón Habana (noviembre): cerrar circuito costero. Planificar rutas alternativas 48h antes.",
                "metadata": {"event_type": "special", "planning_horizon": "48h", "impact": "route_closure"}
            }
        ]
        
        for traffic in traffic_patterns:
            self.vector_db.add_document("traffic_events", traffic["content"], traffic["metadata"])
        
        print("✅ Base de datos poblada con información de referencia")
    
    def run_augmented_search_experiment(self) -> Dict[str, Any]:
        """Ejecuta experimento completo de búsqueda aumentada"""
        print("🔍 EXPERIMENTO DE BÚSQUEDA AUMENTADA Y RESPUESTAS CONTEXTUALIZADAS")
        print("=" * 70)
        
        results = {
            "search_quality_tests": [],
            "context_augmentation_tests": [],
            "response_enhancement_tests": [],
            "comparative_analysis": {},
            "performance_metrics": {}
        }
        
        # 1. Pruebas de calidad de búsqueda
        print("\n1️⃣ PRUEBAS DE CALIDAD DE BÚSQUEDA")
        results["search_quality_tests"] = self._test_search_quality()
        
        # 2. Pruebas de aumento de contexto
        print("\n2️⃣ PRUEBAS DE AUMENTO DE CONTEXTO")
        results["context_augmentation_tests"] = self._test_context_augmentation()
        
        # 3. Pruebas de mejora de respuestas
        print("\n3️⃣ PRUEBAS DE MEJORA DE RESPUESTAS")
        results["response_enhancement_tests"] = self._test_response_enhancement()
        
        # 4. Análisis comparativo
        print("\n4️⃣ ANÁLISIS COMPARATIVO")
        results["comparative_analysis"] = self._comparative_analysis()
        
        # 5. Métricas de rendimiento
        print("\n5️⃣ MÉTRICAS DE RENDIMIENTO")
        results["performance_metrics"] = self._performance_metrics()
        
        # Generar reporte
        self._generate_augmented_search_report(results)
        
        return results
    
    def _test_search_quality(self) -> List[Dict[str, Any]]:
        """Prueba la calidad de búsqueda híbrida vs. individual"""
        print("🔍 Evaluando calidad de búsqueda...")
        
        test_queries = [
            {
                "query": "¿Cómo afecta la lluvia intensa a las entregas en La Habana?",
                "expected_topics": ["weather", "delivery_impact", "habana"],
                "complexity": "medium"
            },
            {
                "query": "Optimización de rutas para vehículos pesados en Centro Habana",
                "expected_topics": ["heavy_vehicles", "centro_habana", "optimization"],
                "complexity": "high"
            },
            {
                "query": "Mejores horarios para evitar tráfico en el Vedado",
                "expected_topics": ["traffic_patterns", "vedado", "timing"],
                "complexity": "medium"
            },
            {
                "query": "Restricciones vehiculares en Habana Vieja para distribución",
                "expected_topics": ["vehicle_restrictions", "habana_vieja", "distribution"],
                "complexity": "high"
            },
            {
                "query": "Impacto del viento fuerte en rutas costeras",
                "expected_topics": ["wind_impact", "coastal_routes", "weather"],
                "complexity": "medium"
            }
        ]
        
        quality_results = []
        
        for test_query in test_queries:
            print(f"  🔍 Probando: '{test_query['query'][:40]}...'")
            
            # Búsqueda solo vectorial
            vector_only = self.vector_db.search(test_query["query"], top_k=5)
            
            # Búsqueda solo LSI
            lsi_only = self.ir_system.search(test_query["query"], top_k=5, use_hybrid=False)
            
            # Búsqueda híbrida
            hybrid_search = self.ir_system.search(test_query["query"], top_k=5, use_hybrid=True)
            
            # Evaluar relevancia de cada método
            vector_relevance = self._evaluate_search_relevance(vector_only, test_query["expected_topics"])
            lsi_relevance = self._evaluate_search_relevance(lsi_only, test_query["expected_topics"])
            hybrid_relevance = self._evaluate_search_relevance(hybrid_search, test_query["expected_topics"])
            
            quality_result = {
                "query": test_query["query"],
                "complexity": test_query["complexity"],
                "vector_only": {
                    "results_count": len(vector_only),
                    "relevance_score": vector_relevance,
                    "avg_similarity": np.mean([r.get('similarity', 0) for r in vector_only]) if vector_only else 0
                },
                "lsi_only": {
                    "results_count": len(lsi_only),
                    "relevance_score": lsi_relevance,
                    "avg_score": np.mean([r.get('score', 0) for r in lsi_only]) if lsi_only else 0
                },
                "hybrid": {
                    "results_count": len(hybrid_search),
                    "relevance_score": hybrid_relevance,
                    "avg_score": np.mean([r.get('score', 0) for r in hybrid_search]) if hybrid_search else 0,
                    "sources_diversity": len(set(r.get('source', 'unknown') for r in hybrid_search))
                },
                "improvement_over_vector": hybrid_relevance - vector_relevance,
                "improvement_over_lsi": hybrid_relevance - lsi_relevance
            }
            
            quality_results.append(quality_result)
            
            print(f"    📊 Vector: {vector_relevance:.3f}, LSI: {lsi_relevance:.3f}, Híbrido: {hybrid_relevance:.3f}")
        
        return quality_results
    
    def _test_context_augmentation(self) -> List[Dict[str, Any]]:
        """Prueba el aumento de contexto en tiempo real"""
        print("🔄 Evaluando aumento de contexto en tiempo real...")
        
        augmentation_results = []
        
        # Escenarios de contexto progresivo
        scenarios = [
            {
                "question": "¿Cuál es la mejor estrategia para entregas en Centro Habana?",
                "context_layers": [
                    {"weather": {"current": "soleado", "temperature": 28, "wind_speed": 8}},
                    {"routes": [{"area": "centro_habana", "efficiency": 0.85, "vehicle_type": "medium"}]},
                    {"traffic_events": {"location": "centro_habana", "congestion_level": "medium", "hour": 14}}
                ]
            },
            {
                "question": "¿Cómo optimizar entregas cuando llueve en La Habana?",
                "context_layers": [
                    {"weather": {"current": "lluvia_intensa", "temperature": 22, "precipitation": 15, "visibility": 3}},
                    {"traffic_events": {"weather_impact": "high", "speed_reduction": 0.35}},
                    {"routes": [{"weather_adapted": True, "efficiency_reduction": 0.45}]}
                ]
            }
        ]
        
        for scenario in scenarios:
            print(f"  🔄 Escenario: {scenario['question'][:30]}...")
            
            context_progression = []
            
            # Estado inicial (sin contexto adicional)
            initial_response = self.rag_assistant.ask_with_context(scenario["question"])
            context_progression.append({
                "stage": "initial",
                "docs_retrieved": initial_response.get("context_used", {}).get("retrieved_documents", 0),
                "collections_used": len(initial_response.get("context_used", {}).get("collections_searched", [])),
                "response_quality": self._assess_response_comprehensiveness(initial_response)
            })
            
            # Añadir capas de contexto progresivamente
            for i, context_layer in enumerate(scenario["context_layers"]):
                # Añadir nueva capa de contexto
                for data_type, data in context_layer.items():
                    self.rag_assistant.update_knowledge_base(data_type, data)
                
                # Evaluar respuesta con contexto aumentado
                augmented_response = self.rag_assistant.ask_with_context(scenario["question"])
                
                context_progression.append({
                    "stage": f"layer_{i+1}",
                    "context_added": list(context_layer.keys()),
                    "docs_retrieved": augmented_response.get("context_used", {}).get("retrieved_documents", 0),
                    "collections_used": len(augmented_response.get("context_used", {}).get("collections_searched", [])),
                    "response_quality": self._assess_response_comprehensiveness(augmented_response),
                    "sources_used": augmented_response.get("context_used", {}).get("sources_used", [])
                })
            
            # Analizar progresión del contexto
            quality_progression = [stage["response_quality"] for stage in context_progression]
            docs_progression = [stage["docs_retrieved"] for stage in context_progression]
            
            augmentation_result = {
                "scenario": scenario["question"],
                "context_progression": context_progression,
                "quality_improvement": quality_progression[-1] - quality_progression[0],
                "docs_growth": docs_progression[-1] - docs_progression[0],
                "context_effectiveness": np.mean(np.diff(quality_progression)),
                "final_quality": quality_progression[-1]
            }
            
            augmentation_results.append(augmentation_result)
            
            print(f"    📈 Mejora de calidad: {augmentation_result['quality_improvement']:.3f}")
            print(f"    📚 Crecimiento de docs: {augmentation_result['docs_growth']}")
        
        return augmentation_results
    
    def _test_response_enhancement(self) -> List[Dict[str, Any]]:
        """Prueba la mejora de respuestas mediante contexto recuperado"""
        print("🚀 Evaluando mejora de respuestas con contexto...")
        
        enhancement_results = []
        
        # Configurar escenarios complejos
        complex_scenarios = [
            {
                "question": "Diseña una estrategia completa de distribución para días lluviosos en La Habana",
                "required_context": {
                    "weather": {"current": "lluvia_fuerte", "precipitation": 20, "visibility": 2, "duration": 180},
                    "routes": [
                        {"area": "vedado", "weather_impact": 0.65, "alternative_routes": ["linea", "23"]},
                        {"area": "centro_habana", "weather_impact": 0.55, "vehicle_restrictions": True}
                    ],
                    "traffic_events": {"rain_related_accidents": 3, "speed_reduction": 0.4}
                },
                "expected_elements": ["weather_adaptation", "route_modification", "vehicle_selection", "timing_optimization"]
            },
            {
                "question": "Optimiza la eficiencia energética de la flota considerando tráfico y topografía de La Habana",
                "required_context": {
                    "routes": [
                        {"area": "loma_colinas", "elevation_change": 45, "fuel_impact": 1.25},
                        {"area": "malecon_flat", "elevation_change": 2, "fuel_impact": 0.95}
                    ],
                    "traffic_events": {"congestion_zones": ["23_y_12", "linea_paseo"], "idle_time": 25},
                    "performance": {"fleet_efficiency": 0.78, "fuel_consumption": "12.5L/100km"}
                },
                "expected_elements": ["fuel_optimization", "route_efficiency", "traffic_avoidance", "topographic_consideration"]
            }
        ]
        
        for scenario in complex_scenarios:
            print(f"  🚀 Escenario complejo: {scenario['question'][:40]}...")
            
            # Establecer contexto rico
            for data_type, data in scenario["required_context"].items():
                self.rag_assistant.update_knowledge_base(data_type, data)
            
            # Obtener respuesta mejorada
            enhanced_response = self.rag_assistant.ask_with_context(scenario["question"])
            
            # Evaluar elementos esperados en la respuesta
            response_text = enhanced_response.get("response", "").lower()
            elements_found = []
            for element in scenario["expected_elements"]:
                element_keywords = element.split("_")
                if any(keyword in response_text for keyword in element_keywords):
                    elements_found.append(element)
            
            enhancement_result = {
                "scenario": scenario["question"],
                "context_richness": len(scenario["required_context"]),
                "docs_retrieved": enhanced_response.get("context_used", {}).get("retrieved_documents", 0),
                "collections_involved": enhanced_response.get("context_used", {}).get("collections_searched", []),
                "sources_used": enhanced_response.get("context_used", {}).get("sources_used", []),
                "expected_elements": scenario["expected_elements"],
                "elements_found": elements_found,
                "completeness_score": len(elements_found) / len(scenario["expected_elements"]),
                "response_length": len(enhanced_response.get("response", "")),
                "context_integration_quality": self._assess_context_integration(enhanced_response),
                "actionability_score": self._assess_response_actionability(enhanced_response)
            }
            
            enhancement_results.append(enhancement_result)
            
            print(f"    ✅ Elementos encontrados: {len(elements_found)}/{len(scenario['expected_elements'])}")
            print(f"    🎯 Completitud: {enhancement_result['completeness_score']:.2%}")
            print(f"    🔧 Accionabilidad: {enhancement_result['actionability_score']:.3f}")
        
        return enhancement_results
    
    def _comparative_analysis(self) -> Dict[str, Any]:
        """Realiza análisis comparativo entre diferentes enfoques"""
        print("📊 Realizando análisis comparativo...")
        
        comparative_queries = [
            "¿Cómo planificar rutas eficientes en La Habana?",
            "¿Qué hacer cuando llueve durante las entregas?",
            "¿Cómo evitar el tráfico en horas pico?",
            "¿Cuál es la mejor estrategia para vehículos pesados?"
        ]
        
        comparison_results = {
            "response_quality": {"simple": [], "context_aware": [], "fully_augmented": []},
            "response_times": {"simple": [], "context_aware": [], "fully_augmented": []},
            "context_utilization": {"simple": [], "context_aware": [], "fully_augmented": []}
        }
        
        for query in comparative_queries:
            print(f"  📊 Comparando: '{query[:30]}...'")
            
            # 1. Respuesta simple (sin contexto adicional)
            start_time = time.time()
            simple_response = self.rag_assistant.ask_with_context(query)
            simple_time = time.time() - start_time
            
            # 2. Respuesta con contexto básico
            self.rag_assistant.update_knowledge_base("weather", {"current": "variable", "temperature": 26})
            start_time = time.time()
            context_response = self.rag_assistant.ask_with_context(query)
            context_time = time.time() - start_time
            
            # 3. Respuesta completamente aumentada
            self.rag_assistant.update_knowledge_base("routes", {"current_efficiency": 0.82, "active_routes": 5})
            self.rag_assistant.update_knowledge_base("traffic_events", {"current_incidents": 2, "congestion_level": "medium"})
            start_time = time.time()
            augmented_response = self.rag_assistant.ask_with_context(query)
            augmented_time = time.time() - start_time
            
            # Evaluar calidad
            simple_quality = self._assess_response_comprehensiveness(simple_response)
            context_quality = self._assess_response_comprehensiveness(context_response)
            augmented_quality = self._assess_response_comprehensiveness(augmented_response)
            
            # Evaluar utilización de contexto
            simple_context = simple_response.get("context_used", {}).get("retrieved_documents", 0)
            context_context = context_response.get("context_used", {}).get("retrieved_documents", 0)
            augmented_context = augmented_response.get("context_used", {}).get("retrieved_documents", 0)
            
            # Almacenar resultados
            comparison_results["response_quality"]["simple"].append(simple_quality)
            comparison_results["response_quality"]["context_aware"].append(context_quality)
            comparison_results["response_quality"]["fully_augmented"].append(augmented_quality)
            
            comparison_results["response_times"]["simple"].append(simple_time)
            comparison_results["response_times"]["context_aware"].append(context_time)
            comparison_results["response_times"]["fully_augmented"].append(augmented_time)
            
            comparison_results["context_utilization"]["simple"].append(simple_context)
            comparison_results["context_utilization"]["context_aware"].append(context_context)
            comparison_results["context_utilization"]["fully_augmented"].append(augmented_context)
        
        # Calcular estadísticas comparativas
        comparison_stats = {}
        for metric in ["response_quality", "response_times", "context_utilization"]:
            comparison_stats[metric] = {}
            for approach in ["simple", "context_aware", "fully_augmented"]:
                values = comparison_results[metric][approach]
                comparison_stats[metric][approach] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        # Calcular mejoras
        quality_improvement_context = ((comparison_stats["response_quality"]["context_aware"]["mean"] - 
                                      comparison_stats["response_quality"]["simple"]["mean"]) / 
                                     comparison_stats["response_quality"]["simple"]["mean"]) * 100
        
        quality_improvement_augmented = ((comparison_stats["response_quality"]["fully_augmented"]["mean"] - 
                                        comparison_stats["response_quality"]["simple"]["mean"]) / 
                                       comparison_stats["response_quality"]["simple"]["mean"]) * 100
        
        return {
            "detailed_results": comparison_results,
            "statistical_summary": comparison_stats,
            "improvements": {
                "context_aware_quality": quality_improvement_context,
                "fully_augmented_quality": quality_improvement_augmented,
                "context_utilization_growth": comparison_stats["context_utilization"]["fully_augmented"]["mean"] - 
                                            comparison_stats["context_utilization"]["simple"]["mean"]
            }
        }
    
    def _performance_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de rendimiento del sistema aumentado"""
        print("⚡ Calculando métricas de rendimiento...")
        
        # Test de escalabilidad
        scalability_queries = [f"Query de prueba número {i} sobre optimización de rutas" for i in range(20)]
        
        latencies = []
        throughputs = []
        memory_usage = []
        
        # Medir latencia individual
        for query in scalability_queries[:10]:
            start_time = time.time()
            response = self.rag_assistant.ask_with_context(query)
            end_time = time.time()
            latencies.append(end_time - start_time)
        
        # Medir throughput en lote
        batch_start = time.time()
        batch_responses = []
        for query in scalability_queries[10:15]:
            response = self.rag_assistant.ask_with_context(query)
            batch_responses.append(response)
        batch_end = time.time()
        
        batch_throughput = len(batch_responses) / (batch_end - batch_start)
        
        # Evaluar uso de recursos
        total_documents = 0
        collection_sizes = {}
        for collection_name, collection in self.vector_db.collections.items():
            count = collection.count()
            collection_sizes[collection_name] = count
            total_documents += count
        
        return {
            "latency_analysis": {
                "avg_latency": np.mean(latencies),
                "p50_latency": np.percentile(latencies, 50),
                "p95_latency": np.percentile(latencies, 95),
                "p99_latency": np.percentile(latencies, 99),
                "latency_std": np.std(latencies)
            },
            "throughput_analysis": {
                "queries_per_second": batch_throughput,
                "batch_processing_time": batch_end - batch_start,
                "avg_batch_quality": np.mean([self._assess_response_comprehensiveness(r) for r in batch_responses])
            },
            "resource_utilization": {
                "total_documents": total_documents,
                "collection_distribution": collection_sizes,
                "largest_collection": max(collection_sizes.values()) if collection_sizes else 0,
                "avg_collection_size": np.mean(list(collection_sizes.values())) if collection_sizes else 0
            },
            "scalability_indicators": {
                "latency_scalability": np.corrcoef(range(len(latencies)), latencies)[0, 1],  # Correlación con orden
                "memory_efficiency": total_documents / (np.mean(latencies) * 1000),  # Docs por ms
                "context_efficiency": np.mean([len(r.get("context_used", {}).get("sources_used", [])) for r in batch_responses])
            }
        }
    
    def _evaluate_search_relevance(self, results: List[Dict], expected_topics: List[str]) -> float:
        """Evalúa la relevancia de los resultados de búsqueda"""
        if not results:
            return 0.0
        
        relevance_score = 0.0
        
        for result in results:
            content = result.get('document', result.get('content', '')).lower()
            metadata = result.get('metadata', {})
            
            # Evaluar coincidencia con temas esperados
            topic_matches = sum(1 for topic in expected_topics if topic.lower() in content)
            relevance_score += topic_matches / len(expected_topics)
            
            # Bonus por metadata relevante
            if any(topic.lower() in str(metadata).lower() for topic in expected_topics):
                relevance_score += 0.1
        
        return relevance_score / len(results)
    
    def _assess_response_comprehensiveness(self, response: Dict[str, Any]) -> float:
        """Evalúa la exhaustividad de una respuesta"""
        score = 0.0
        
        # Factor 1: Documentos recuperados
        docs = response.get("context_used", {}).get("retrieved_documents", 0)
        score += min(docs / 10.0, 1.0) * 0.25
        
        # Factor 2: Diversidad de colecciones
        collections = len(response.get("context_used", {}).get("collections_searched", []))
        score += min(collections / 6.0, 1.0) * 0.25
        
        # Factor 3: Longitud de respuesta
        response_length = len(response.get("response", ""))
        score += min(response_length / 800.0, 1.0) * 0.25
        
        # Factor 4: Métricas incluidas
        has_metrics = bool(response.get("relevant_metrics"))
        score += 0.25 if has_metrics else 0.0
        
        return score
    
    def _assess_context_integration(self, response: Dict[str, Any]) -> float:
        """Evalúa qué tan bien se integra el contexto en la respuesta"""
        integration_score = 0.0
        
        response_text = response.get("response", "").lower()
        context_used = response.get("context_used", {})
        
        # Verificar uso de fuentes múltiples
        sources = context_used.get("sources_used", [])
        if len(sources) > 1:
            integration_score += 0.3
        
        # Verificar uso de colecciones múltiples
        collections = context_used.get("collections_searched", [])
        if len(collections) > 2:
            integration_score += 0.3
        
        # Verificar menciones específicas del contexto
        context_indicators = ["basado en", "según", "considerando", "análisis", "datos"]
        context_mentions = sum(1 for indicator in context_indicators if indicator in response_text)
        integration_score += min(context_mentions / len(context_indicators), 1.0) * 0.4
        
        return integration_score
    
    def _assess_response_actionability(self, response: Dict[str, Any]) -> float:
        """Evalúa qué tan accionable es una respuesta"""
        response_text = response.get("response", "").lower()
        
        # Indicadores de accionabilidad
        action_indicators = [
            "recomendación", "sugerencia", "debe", "puede", "considere",
            "implemente", "use", "evite", "optimice", "planifique"
        ]
        
        action_count = sum(1 for indicator in action_indicators if indicator in response_text)
        actionability = min(action_count / 5.0, 1.0)  # Normalizar a 5 indicadores
        
        return actionability
    
    def _generate_augmented_search_report(self, results: Dict[str, Any]):
        """Genera reporte del experimento de búsqueda aumentada"""
        report_path = "augmented_search_experiment_report.json"
        
        # Agregar resumen ejecutivo
        results["executive_summary"] = {
            "experiment_date": datetime.now().isoformat(),
            "total_test_categories": 5,
            "search_quality_improvement": self._calculate_search_improvement(results["search_quality_tests"]),
            "context_augmentation_effectiveness": self._calculate_context_effectiveness(results["context_augmentation_tests"]),
            "response_enhancement_success": self._calculate_enhancement_success(results["response_enhancement_tests"]),
            "overall_performance": self._calculate_overall_performance(results)
        }
        
        # Guardar reporte
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Reporte de búsqueda aumentada guardado en: {report_path}")
        self._print_augmented_summary(results)
    
    def _calculate_search_improvement(self, search_tests: List[Dict]) -> float:
        """Calcula mejora promedio en búsqueda híbrida"""
        if not search_tests:
            return 0.0
        
        improvements = [test["improvement_over_vector"] + test["improvement_over_lsi"] for test in search_tests]
        return np.mean(improvements)
    
    def _calculate_context_effectiveness(self, augmentation_tests: List[Dict]) -> float:
        """Calcula efectividad de aumento de contexto"""
        if not augmentation_tests:
            return 0.0
        
        effectiveness = [test["context_effectiveness"] for test in augmentation_tests]
        return np.mean(effectiveness)
    
    def _calculate_enhancement_success(self, enhancement_tests: List[Dict]) -> float:
        """Calcula éxito de mejora de respuestas"""
        if not enhancement_tests:
            return 0.0
        
        completeness_scores = [test["completeness_score"] for test in enhancement_tests]
        return np.mean(completeness_scores)
    
    def _calculate_overall_performance(self, results: Dict[str, Any]) -> float:
        """Calcula rendimiento general del experimento"""
        scores = []
        
        if results["search_quality_tests"]:
            avg_hybrid_relevance = np.mean([test["hybrid"]["relevance_score"] for test in results["search_quality_tests"]])
            scores.append(avg_hybrid_relevance)
        
        if results["context_augmentation_tests"]:
            avg_final_quality = np.mean([test["final_quality"] for test in results["context_augmentation_tests"]])
            scores.append(avg_final_quality)
        
        if results["response_enhancement_tests"]:
            avg_completeness = np.mean([test["completeness_score"] for test in results["response_enhancement_tests"]])
            scores.append(avg_completeness)
        
        return np.mean(scores) if scores else 0.0
    
    def _print_augmented_summary(self, results: Dict[str, Any]):
        """Imprime resumen del experimento de búsqueda aumentada"""
        print("\n" + "="*70)
        print("📊 RESUMEN DEL EXPERIMENTO DE BÚSQUEDA AUMENTADA")
        print("="*70)
        
        summary = results["executive_summary"]
        
        print(f"🎯 Rendimiento general: {summary['overall_performance']:.2%}")
        print(f"🔍 Mejora en búsqueda: {summary['search_quality_improvement']:.3f}")
        print(f"🔄 Efectividad de contexto: {summary['context_augmentation_effectiveness']:.3f}")
        print(f"🚀 Éxito en mejora de respuestas: {summary['response_enhancement_success']:.2%}")
        
        if "performance_metrics" in results:
            perf = results["performance_metrics"]
            print(f"⚡ Latencia promedio: {perf['latency_analysis']['avg_latency']:.3f}s")
            print(f"📊 Throughput: {perf['throughput_analysis']['queries_per_second']:.2f} QPS")
            print(f"📚 Documentos totales: {perf['resource_utilization']['total_documents']}")
        
        if "comparative_analysis" in results:
            comp = results["comparative_analysis"]["improvements"]
            print(f"📈 Mejora con contexto: {comp['context_aware_quality']:.1f}%")
            print(f"🚀 Mejora completamente aumentada: {comp['fully_augmented_quality']:.1f}%")

def main():
    """Función principal del experimento de búsqueda aumentada"""
    print("🔍 EXPERIMENTO DE BÚSQUEDA AUMENTADA Y RESPUESTA CONTEXTUALIZADA")
    print("Este experimento valida la calidad de búsqueda híbrida y mejora de respuestas")
    print("-" * 70)
    
    # Ejecutar experimento
    experiment = AugmentedSearchExperiment()
    results = experiment.run_augmented_search_experiment()
    
    print("\n🎉 EXPERIMENTO DE BÚSQUEDA AUMENTADA COMPLETADO!")
    print("Revisa 'augmented_search_experiment_report.json' para detalles completos.")

if __name__ == "__main__":
    main()
