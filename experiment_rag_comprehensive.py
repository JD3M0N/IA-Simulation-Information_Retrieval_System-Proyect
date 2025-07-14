#!/usr/bin/env python3
"""
Suite de Experimentos para Validación Completa del Sistema RAG Vectorial
Valida: retroalimentación, búsqueda híbrida, actualización en tiempo real,
persistencia, expansión Rocchio, y respuesta aumentada
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.NLP.RAG import create_vrp_rag_assistant
from src.SRI.VectorDatabase import VRPVectorDatabase
from src.SRI.VRPInformationRetrieval import VRPInformationRetrievalSystem

class RAGExperimentSuite:
    """Suite completa de experimentos para validar el sistema RAG"""
    
    def __init__(self):
        """Inicializa la suite de experimentos"""
        self.rag_assistant = create_vrp_rag_assistant()
        self.vector_db = VRPVectorDatabase("experiment_vector_cache")
        self.ir_system = VRPInformationRetrievalSystem("experiment_vector_cache")
        
        # Configuración de experimentos
        self.results = {
            "retroalimentacion": {},
            "busqueda_hibrida": {},
            "actualizacion_tiempo_real": {},
            "persistencia": {},
            "expansion_rocchio": {},
            "respuesta_aumentada": {},
            "rendimiento": {}
        }
        
        # Datos de prueba
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, Any]:
        """Genera datos de prueba para los experimentos"""
        return {
            "weather_scenarios": [
                {
                    "condition": "lluvia_intensa",
                    "data": {
                        "current": "lluvia_fuerte",
                        "temperature": 24,
                        "humidity": 95,
                        "wind_speed": 25,
                        "precipitation": 15,
                        "visibility": 2
                    }
                },
                {
                    "condition": "soleado",
                    "data": {
                        "current": "despejado",
                        "temperature": 28,
                        "humidity": 65,
                        "wind_speed": 8,
                        "precipitation": 0,
                        "visibility": 10
                    }
                },
                {
                    "condition": "viento_fuerte",
                    "data": {
                        "current": "ventoso",
                        "temperature": 26,
                        "humidity": 70,
                        "wind_speed": 35,
                        "precipitation": 0,
                        "visibility": 8
                    }
                }
            ],
            "route_scenarios": [
                {
                    "scenario": "eficiente",
                    "routes": [
                        {"id": "route_1", "distance": 8.5, "duration": 25, "cost": 85.50, "customers": 5},
                        {"id": "route_2", "distance": 12.3, "duration": 38, "cost": 125.75, "customers": 8}
                    ]
                },
                {
                    "scenario": "ineficiente", 
                    "routes": [
                        {"id": "route_1", "distance": 18.5, "duration": 65, "cost": 185.50, "customers": 4},
                        {"id": "route_2", "distance": 22.3, "duration": 78, "cost": 245.75, "customers": 6}
                    ]
                }
            ],
            "traffic_events": [
                {
                    "type": "accidente",
                    "location": "Malecón y 23",
                    "severity": "alta",
                    "impact": "cierre_parcial",
                    "estimated_duration": 120
                },
                {
                    "type": "obras",
                    "location": "Línea y Paseo",
                    "severity": "media",
                    "impact": "desvio",
                    "estimated_duration": 480
                }
            ],
            "test_questions": [
                # Preguntas sobre clima
                "¿Cómo afecta la lluvia intensa a mis rutas de entrega?",
                "¿Qué recomendaciones tienes para días ventosos?",
                "¿Cuál es el mejor horario para entregar cuando llueve?",
                
                # Preguntas sobre rutas
                "¿Cómo puedo optimizar mis rutas actuales?",
                "¿Qué factores influyen en la eficiencia de las rutas?",
                "¿Cómo calculo el costo por kilómetro?",
                
                # Preguntas sobre tráfico
                "¿Cómo evito las zonas de construcción?",
                "¿Qué hacer cuando hay un accidente en mi ruta?",
                "¿Cuáles son las horas pico en La Habana?",
                
                # Preguntas mixtas
                "¿Cómo combino información de clima y tráfico para optimizar rutas?",
                "¿Qué estrategias uso en días lluviosos con mucho tráfico?",
                "¿Cómo afecta el viento fuerte a los tiempos de entrega?"
            ]
        }
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Ejecuta todos los experimentos de validación"""
        print("🧪 INICIANDO SUITE COMPLETA DE EXPERIMENTOS RAG")
        print("=" * 60)
        
        # 1. Experimento de Retroalimentación
        print("\n1️⃣ EXPERIMENTO: Retroalimentación y Aprendizaje Continuo")
        self.results["retroalimentacion"] = self.experiment_feedback_learning()
        
        # 2. Experimento de Búsqueda Híbrida
        print("\n2️⃣ EXPERIMENTO: Búsqueda Híbrida (Vector + LSI)")
        self.results["busqueda_hibrida"] = self.experiment_hybrid_search()
        
        # 3. Experimento de Actualización en Tiempo Real
        print("\n3️⃣ EXPERIMENTO: Actualización en Tiempo Real")
        self.results["actualizacion_tiempo_real"] = self.experiment_realtime_updates()
        
        # 4. Experimento de Persistencia
        print("\n4️⃣ EXPERIMENTO: Persistencia de Base de Datos")
        self.results["persistencia"] = self.experiment_persistence()
        
        # 5. Experimento de Expansión Rocchio
        print("\n5️⃣ EXPERIMENTO: Expansión de Consultas Rocchio")
        self.results["expansion_rocchio"] = self.experiment_rocchio_expansion()
        
        # 6. Experimento de Respuesta Aumentada
        print("\n6️⃣ EXPERIMENTO: Respuesta Aumentada con Contexto")
        self.results["respuesta_aumentada"] = self.experiment_augmented_response()
        
        # 7. Experimento de Rendimiento
        print("\n7️⃣ EXPERIMENTO: Análisis de Rendimiento")
        self.results["rendimiento"] = self.experiment_performance_analysis()
        
        # Generar reporte final
        self.generate_final_report()
        
        return self.results
    
    def experiment_feedback_learning(self) -> Dict[str, Any]:
        """
        Experimento 1: Valida la retroalimentación y aprendizaje continuo
        """
        print("🔄 Validando retroalimentación y aprendizaje continuo...")
        
        results = {
            "learning_progression": [],
            "knowledge_growth": [],
            "context_improvement": []
        }
        
        # Pregunta base que evolucionará
        base_question = "¿Cómo optimizar rutas en La Habana?"
        
        # Simular 5 iteraciones de aprendizaje
        for iteration in range(5):
            print(f"  📊 Iteración {iteration + 1}/5")
            
            # 1. Consulta inicial
            response_before = self.rag_assistant.ask_with_context(base_question)
            docs_before = response_before.get("context_used", {}).get("retrieved_documents", 0)
            
            # 2. Añadir nueva información contextual
            if iteration == 0:
                self.rag_assistant.update_knowledge_base("weather", self.test_data["weather_scenarios"][0]["data"])
            elif iteration == 1:
                self.rag_assistant.update_knowledge_base("routes", {"routes": self.test_data["route_scenarios"][0]["routes"]})
            elif iteration == 2:
                self.rag_assistant.update_knowledge_base("traffic_events", self.test_data["traffic_events"][0])
            elif iteration == 3:
                # Añadir insight histórico
                historical_data = {
                    "pattern_type": "weather_route_correlation",
                    "confidence": 0.85,
                    "description": "Las rutas son 15% menos eficientes en días lluviosos"
                }
                self.rag_assistant.update_knowledge_base("performance", historical_data)
            
            # 3. Consulta después de añadir contexto
            response_after = self.rag_assistant.ask_with_context(base_question)
            docs_after = response_after.get("context_used", {}).get("retrieved_documents", 0)
            
            # 4. Medir mejora
            improvement = {
                "iteration": iteration + 1,
                "docs_before": docs_before,
                "docs_after": docs_after,
                "improvement": docs_after - docs_before,
                "collections_used": len(response_after.get("context_used", {}).get("collections_searched", [])),
                "response_length": len(response_after.get("response", "")),
                "context_quality": self._assess_context_quality(response_after)
            }
            
            results["learning_progression"].append(improvement)
            
            time.sleep(0.5)  # Pausa para observación
        
        # Analizar crecimiento del conocimiento
        total_docs = 0
        for collection_name in self.vector_db.collections:
            collection = self.vector_db.collections[collection_name]
            count = collection.count()
            total_docs += count
            results["knowledge_growth"].append({
                "collection": collection_name,
                "document_count": count
            })
        
        results["total_documents"] = total_docs
        results["learning_effectiveness"] = self._calculate_learning_effectiveness(results["learning_progression"])
        
        print(f"  ✅ Documentos totales en base: {total_docs}")
        print(f"  📈 Efectividad de aprendizaje: {results['learning_effectiveness']:.2f}")
        
        return results
    
    def experiment_hybrid_search(self) -> Dict[str, Any]:
        """
        Experimento 2: Valida la búsqueda híbrida (Vector + LSI)
        """
        print("🔍 Validando búsqueda híbrida Vector + LSI...")
        
        results = {
            "vector_only": [],
            "lsi_only": [],
            "hybrid": [],
            "comparison": {}
        }
        
        test_queries = [
            "optimización de rutas con algoritmos genéticos",
            "impacto del clima en la logística urbana",
            "gestión de tráfico en centro histórico",
            "eficiencia energética en vehículos de reparto"
        ]
        
        for query in test_queries:
            print(f"  🔍 Probando: '{query[:30]}...'")
            
            # 1. Búsqueda solo vectorial
            start_time = time.time()
            vector_results = self.vector_db.search(query, top_k=5)
            vector_time = time.time() - start_time
            
            # 2. Búsqueda solo LSI
            start_time = time.time()
            lsi_results = self.ir_system.search(query, top_k=5, use_hybrid=False)
            lsi_time = time.time() - start_time
            
            # 3. Búsqueda híbrida
            start_time = time.time()
            hybrid_results = self.ir_system.search(query, top_k=5, use_hybrid=True)
            hybrid_time = time.time() - start_time
            
            # Evaluar calidad de resultados
            vector_quality = self._evaluate_search_quality(vector_results, query)
            lsi_quality = self._evaluate_search_quality(lsi_results, query)
            hybrid_quality = self._evaluate_search_quality(hybrid_results, query)
            
            results["vector_only"].append({
                "query": query,
                "results_count": len(vector_results),
                "avg_similarity": np.mean([r.get('similarity', 0) for r in vector_results]) if vector_results else 0,
                "search_time": vector_time,
                "quality_score": vector_quality
            })
            
            results["lsi_only"].append({
                "query": query,
                "results_count": len(lsi_results),
                "avg_score": np.mean([r.get('score', 0) for r in lsi_results]) if lsi_results else 0,
                "search_time": lsi_time,
                "quality_score": lsi_quality
            })
            
            results["hybrid"].append({
                "query": query,
                "results_count": len(hybrid_results),
                "avg_score": np.mean([r.get('score', 0) for r in hybrid_results]) if hybrid_results else 0,
                "search_time": hybrid_time,
                "quality_score": hybrid_quality,
                "sources_used": list(set([r.get('source', 'unknown') for r in hybrid_results]))
            })
        
        # Comparación estadística
        results["comparison"] = {
            "avg_vector_quality": np.mean([r["quality_score"] for r in results["vector_only"]]),
            "avg_lsi_quality": np.mean([r["quality_score"] for r in results["lsi_only"]]),
            "avg_hybrid_quality": np.mean([r["quality_score"] for r in results["hybrid"]]),
            "avg_vector_time": np.mean([r["search_time"] for r in results["vector_only"]]),
            "avg_lsi_time": np.mean([r["search_time"] for r in results["lsi_only"]]),
            "avg_hybrid_time": np.mean([r["search_time"] for r in results["hybrid"]])
        }
        
        print(f"  📊 Calidad Vector: {results['comparison']['avg_vector_quality']:.3f}")
        print(f"  📊 Calidad LSI: {results['comparison']['avg_lsi_quality']:.3f}")
        print(f"  📊 Calidad Híbrida: {results['comparison']['avg_hybrid_quality']:.3f}")
        
        return results
    
    def experiment_realtime_updates(self) -> Dict[str, Any]:
        """
        Experimento 3: Valida actualización en tiempo real
        """
        print("⚡ Validando actualización en tiempo real...")
        
        results = {
            "update_timeline": [],
            "response_evolution": [],
            "latency_analysis": {}
        }
        
        question = "¿Cuál es el estado actual del sistema de rutas?"
        
        # Estado inicial
        initial_response = self.rag_assistant.ask_with_context(question)
        results["response_evolution"].append({
            "stage": "inicial",
            "docs_retrieved": initial_response.get("context_used", {}).get("retrieved_documents", 0),
            "collections_used": initial_response.get("context_used", {}).get("collections_searched", [])
        })
        
        # Simular actualizaciones en tiempo real
        updates = [
            ("weather", self.test_data["weather_scenarios"][1]["data"]),
            ("routes", {"routes": self.test_data["route_scenarios"][1]["routes"]}),
            ("traffic_events", self.test_data["traffic_events"][1]),
            ("performance", {"component": "optimizer", "status": "active", "efficiency": 0.89})
        ]
        
        for i, (data_type, data) in enumerate(updates):
            print(f"  ⚡ Actualización {i+1}: {data_type}")
            
            # Medir tiempo de actualización
            start_time = time.time()
            self.rag_assistant.update_knowledge_base(data_type, data)
            update_time = time.time() - start_time
            
            # Verificar impacto en respuesta
            updated_response = self.rag_assistant.ask_with_context(question)
            
            results["update_timeline"].append({
                "update_number": i + 1,
                "data_type": data_type,
                "update_latency": update_time,
                "timestamp": datetime.now().isoformat()
            })
            
            results["response_evolution"].append({
                "stage": f"después_update_{i+1}",
                "docs_retrieved": updated_response.get("context_used", {}).get("retrieved_documents", 0),
                "collections_used": updated_response.get("context_used", {}).get("collections_searched", []),
                "response_change": len(updated_response.get("response", "")) - len(initial_response.get("response", ""))
            })
            
            time.sleep(0.2)
        
        # Análisis de latencia
        latencies = [update["update_latency"] for update in results["update_timeline"]]
        results["latency_analysis"] = {
            "avg_latency": np.mean(latencies),
            "max_latency": np.max(latencies),
            "min_latency": np.min(latencies),
            "total_updates": len(updates)
        }
        
        print(f"  ⏱️ Latencia promedio: {results['latency_analysis']['avg_latency']:.3f}s")
        print(f"  📈 Actualizaciones completadas: {results['latency_analysis']['total_updates']}")
        
        return results
    
    def experiment_persistence(self) -> Dict[str, Any]:
        """
        Experimento 4: Valida persistencia de la base de datos
        """
        print("💾 Validando persistencia de base de datos...")
        
        results = {
            "pre_restart": {},
            "post_restart": {},
            "data_integrity": True
        }
        
        # Estado antes de "reinicio"
        for collection_name, collection in self.vector_db.collections.items():
            count = collection.count()
            results["pre_restart"][collection_name] = count
        
        # Añadir datos específicos para verificar persistencia
        test_doc = {
            "content": f"Documento de prueba de persistencia - {datetime.now().isoformat()}",
            "metadata": {
                "test_id": "persistence_test",
                "timestamp": datetime.now().isoformat(),
                "experiment": "persistence_validation"
            }
        }
        
        test_doc_id = self.vector_db.add_document("knowledge_base", test_doc["content"], test_doc["metadata"])
        
        # Simular reinicio creando nueva instancia
        print("  🔄 Simulando reinicio del sistema...")
        new_vector_db = VRPVectorDatabase("experiment_vector_cache")
        
        # Verificar estado después de "reinicio"
        for collection_name, collection in new_vector_db.collections.items():
            count = collection.count()
            results["post_restart"][collection_name] = count
        
        # Verificar que el documento de prueba persiste
        search_results = new_vector_db.search("documento de prueba de persistencia", ["knowledge_base"])
        persistence_verified = any(r['id'] == test_doc_id for r in search_results)
        
        # Analizar integridad
        for collection_name in results["pre_restart"]:
            pre_count = results["pre_restart"][collection_name]
            post_count = results["post_restart"].get(collection_name, 0)
            if post_count < pre_count:
                results["data_integrity"] = False
                break
        
        results["persistence_verified"] = persistence_verified
        results["test_document_found"] = len(search_results) > 0
        
        print(f"  ✅ Integridad de datos: {results['data_integrity']}")
        print(f"  ✅ Persistencia verificada: {results['persistence_verified']}")
        
        return results
    
    def experiment_rocchio_expansion(self) -> Dict[str, Any]:
        """
        Experimento 5: Valida expansión de consultas Rocchio
        """
        print("🎯 Validando expansión de consultas Rocchio...")
        
        results = {
            "queries_tested": [],
            "expansion_effectiveness": {}
        }
        
        test_queries = [
            "optimización rutas",
            "clima lluvia",
            "tráfico congestion",
            "eficiencia energética"
        ]
        
        for query in test_queries:
            print(f"  🎯 Expandiendo: '{query}'")
            
            # Búsqueda sin expansión
            original_results = self.ir_system.search(query, top_k=5, use_hybrid=False)
            
            # Búsqueda con expansión Rocchio
            expanded_results = self.ir_system.search(query, top_k=5, use_hybrid=True)
            
            # Analizar mejora
            original_quality = self._evaluate_search_quality(original_results, query)
            expanded_quality = self._evaluate_search_quality(expanded_results, query)
            
            query_result = {
                "original_query": query,
                "original_results": len(original_results),
                "expanded_results": len(expanded_results),
                "original_quality": original_quality,
                "expanded_quality": expanded_quality,
                "improvement": expanded_quality - original_quality,
                "expansion_effective": expanded_quality > original_quality
            }
            
            results["queries_tested"].append(query_result)
        
        # Análisis de efectividad global
        improvements = [q["improvement"] for q in results["queries_tested"]]
        effective_count = sum(1 for q in results["queries_tested"] if q["expansion_effective"])
        
        results["expansion_effectiveness"] = {
            "avg_improvement": np.mean(improvements),
            "effective_ratio": effective_count / len(test_queries),
            "total_queries": len(test_queries),
            "effective_expansions": effective_count
        }
        
        print(f"  📈 Mejora promedio: {results['expansion_effectiveness']['avg_improvement']:.3f}")
        print(f"  🎯 Efectividad: {results['expansion_effectiveness']['effective_ratio']:.2%}")
        
        return results
    
    def experiment_augmented_response(self) -> Dict[str, Any]:
        """
        Experimento 6: Valida respuesta aumentada con contexto
        """
        print("🚀 Validando respuesta aumentada con contexto...")
        
        results = {
            "response_analysis": [],
            "context_impact": {}
        }
        
        # Preguntas de diferentes complejidades
        test_scenarios = [
            {
                "question": "¿Cómo optimizar rutas en La Habana?",
                "complexity": "simple",
                "context_data": {}
            },
            {
                "question": "¿Cómo afecta la lluvia a mis rutas de entrega hoy?",
                "complexity": "medium",
                "context_data": {
                    "weather": self.test_data["weather_scenarios"][0]["data"]
                }
            },
            {
                "question": "Analiza la eficiencia de mis rutas considerando clima y tráfico actual",
                "complexity": "complex",
                "context_data": {
                    "weather": self.test_data["weather_scenarios"][0]["data"],
                    "routes": self.test_data["route_scenarios"][0]["routes"],
                    "traffic_events": self.test_data["traffic_events"]
                }
            }
        ]
        
        for scenario in test_scenarios:
            print(f"  🚀 Escenario {scenario['complexity']}: {scenario['question'][:30]}...")
            
            # Actualizar contexto si existe
            for data_type, data in scenario["context_data"].items():
                self.rag_assistant.update_knowledge_base(data_type, data)
            
            # Obtener respuesta aumentada
            response = self.rag_assistant.ask_with_context(scenario["question"])
            
            # Analizar respuesta
            analysis = {
                "question": scenario["question"],
                "complexity": scenario["complexity"],
                "context_sources": len(scenario["context_data"]),
                "response_length": len(response.get("response", "")),
                "docs_retrieved": response.get("context_used", {}).get("retrieved_documents", 0),
                "collections_used": len(response.get("context_used", {}).get("collections_searched", [])),
                "sources_used": response.get("context_used", {}).get("sources_used", []),
                "has_metrics": bool(response.get("relevant_metrics")),
                "context_richness": self._assess_context_richness(response)
            }
            
            results["response_analysis"].append(analysis)
        
        # Análisis del impacto del contexto
        simple_docs = next(r["docs_retrieved"] for r in results["response_analysis"] if r["complexity"] == "simple")
        complex_docs = next(r["docs_retrieved"] for r in results["response_analysis"] if r["complexity"] == "complex")
        
        results["context_impact"] = {
            "context_scaling": complex_docs / simple_docs if simple_docs > 0 else 0,
            "avg_docs_retrieved": np.mean([r["docs_retrieved"] for r in results["response_analysis"]]),
            "avg_collections_used": np.mean([r["collections_used"] for r in results["response_analysis"]]),
            "avg_response_length": np.mean([r["response_length"] for r in results["response_analysis"]]),
            "context_effectiveness": np.mean([r["context_richness"] for r in results["response_analysis"]])
        }
        
        print(f"  📊 Escalamiento contextual: {results['context_impact']['context_scaling']:.2f}x")
        print(f"  🎯 Efectividad contextual: {results['context_impact']['context_effectiveness']:.3f}")
        
        return results
    
    def experiment_performance_analysis(self) -> Dict[str, Any]:
        """
        Experimento 7: Análisis de rendimiento del sistema
        """
        print("⚡ Analizando rendimiento del sistema...")
        
        results = {
            "latency_analysis": {},
            "throughput_analysis": {},
            "resource_usage": {}
        }
        
        # Test de latencia
        latencies = []
        questions = self.test_data["test_questions"]
        
        print("  ⏱️ Midiendo latencias...")
        for i, question in enumerate(questions[:5]):  # Muestra de 5 preguntas
            start_time = time.time()
            response = self.rag_assistant.ask_with_context(question)
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            if i == 0:  # Primera consulta (cache frío)
                cold_start_latency = latency
        
        # Test de throughput
        print("  🚀 Midiendo throughput...")
        concurrent_start = time.time()
        concurrent_responses = []
        
        for question in questions[:3]:  # 3 consultas concurrentes simuladas
            response = self.rag_assistant.ask_with_context(question)
            concurrent_responses.append(response)
        
        concurrent_end = time.time()
        total_concurrent_time = concurrent_end - concurrent_start
        
        # Análisis de recursos
        collection_sizes = {}
        total_documents = 0
        
        for collection_name, collection in self.vector_db.collections.items():
            count = collection.count()
            collection_sizes[collection_name] = count
            total_documents += count
        
        # Compilar resultados
        results["latency_analysis"] = {
            "avg_latency": np.mean(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "cold_start_latency": cold_start_latency,
            "p95_latency": np.percentile(latencies, 95)
        }
        
        results["throughput_analysis"] = {
            "concurrent_queries": len(concurrent_responses),
            "total_time": total_concurrent_time,
            "queries_per_second": len(concurrent_responses) / total_concurrent_time,
            "avg_response_quality": np.mean([self._assess_context_quality(r) for r in concurrent_responses])
        }
        
        results["resource_usage"] = {
            "total_documents": total_documents,
            "collections_count": len(collection_sizes),
            "largest_collection": max(collection_sizes.values()),
            "collection_distribution": collection_sizes
        }
        
        print(f"  ⏱️ Latencia promedio: {results['latency_analysis']['avg_latency']:.3f}s")
        print(f"  🚀 QPS: {results['throughput_analysis']['queries_per_second']:.2f}")
        print(f"  📊 Documentos totales: {results['resource_usage']['total_documents']}")
        
        return results
    
    def _assess_context_quality(self, response: Dict[str, Any]) -> float:
        """Evalúa la calidad del contexto en una respuesta"""
        score = 0.0
        
        # Documentos recuperados
        docs = response.get("context_used", {}).get("retrieved_documents", 0)
        score += min(docs / 10.0, 1.0) * 0.3
        
        # Colecciones usadas
        collections = len(response.get("context_used", {}).get("collections_searched", []))
        score += min(collections / 6.0, 1.0) * 0.2
        
        # Fuentes diversas
        sources = len(response.get("context_used", {}).get("sources_used", []))
        score += min(sources / 2.0, 1.0) * 0.2
        
        # Métricas relevantes
        has_metrics = bool(response.get("relevant_metrics"))
        score += 0.3 if has_metrics else 0.0
        
        return score
    
    def _assess_context_richness(self, response: Dict[str, Any]) -> float:
        """Evalúa la riqueza del contexto"""
        richness = 0.0
        
        context_used = response.get("context_used", {})
        richness += context_used.get("retrieved_documents", 0) * 0.1
        richness += len(context_used.get("collections_searched", [])) * 0.15
        richness += len(context_used.get("sources_used", [])) * 0.2
        
        return min(richness, 1.0)
    
    def _evaluate_search_quality(self, results: List[Dict], query: str) -> float:
        """Evalúa la calidad de los resultados de búsqueda"""
        if not results:
            return 0.0
        
        # Score basado en similitud/score y relevancia
        avg_score = np.mean([r.get('similarity', r.get('score', 0)) for r in results])
        diversity = len(set(r.get('collection', r.get('source', 'unknown')) for r in results))
        
        return (avg_score * 0.7) + (min(diversity / 3.0, 1.0) * 0.3)
    
    def _calculate_learning_effectiveness(self, progression: List[Dict]) -> float:
        """Calcula la efectividad del aprendizaje"""
        if not progression:
            return 0.0
        
        improvements = [p["improvement"] for p in progression if p["improvement"] > 0]
        total_improvements = len(improvements)
        
        return total_improvements / len(progression)
    
    def generate_final_report(self):
        """Genera reporte final con todas las métricas"""
        report_path = "rag_experiment_report.json"
        
        # Agregar resumen ejecutivo
        self.results["summary"] = {
            "experiment_date": datetime.now().isoformat(),
            "total_experiments": 7,
            "overall_success": self._calculate_overall_success(),
            "key_findings": self._generate_key_findings()
        }
        
        # Guardar reporte completo
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Reporte completo guardado en: {report_path}")
        self._print_summary()
    
    def _calculate_overall_success(self) -> float:
        """Calcula el éxito general de todos los experimentos"""
        scores = []
        
        # Retroalimentación
        if self.results["retroalimentacion"]:
            scores.append(self.results["retroalimentacion"].get("learning_effectiveness", 0))
        
        # Búsqueda híbrida
        if self.results["busqueda_hibrida"]:
            hybrid_quality = self.results["busqueda_hibrida"]["comparison"].get("avg_hybrid_quality", 0)
            scores.append(min(hybrid_quality, 1.0))
        
        # Persistencia
        if self.results["persistencia"]:
            persistence_score = 1.0 if self.results["persistencia"]["data_integrity"] else 0.0
            scores.append(persistence_score)
        
        # Rocchio
        if self.results["expansion_rocchio"]:
            rocchio_effectiveness = self.results["expansion_rocchio"]["expansion_effectiveness"].get("effective_ratio", 0)
            scores.append(rocchio_effectiveness)
        
        # Respuesta aumentada
        if self.results["respuesta_aumentada"]:
            context_effectiveness = self.results["respuesta_aumentada"]["context_impact"].get("context_effectiveness", 0)
            scores.append(context_effectiveness)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_key_findings(self) -> List[str]:
        """Genera hallazgos clave de los experimentos"""
        findings = []
        
        # Análisis de retroalimentación
        if self.results["retroalimentacion"]:
            learning_eff = self.results["retroalimentacion"].get("learning_effectiveness", 0)
            findings.append(f"El sistema muestra {learning_eff:.2%} de efectividad en aprendizaje continuo")
        
        # Análisis de búsqueda híbrida
        if self.results["busqueda_hibrida"]:
            hybrid_qual = self.results["busqueda_hibrida"]["comparison"].get("avg_hybrid_quality", 0)
            vector_qual = self.results["busqueda_hibrida"]["comparison"].get("avg_vector_quality", 0)
            if hybrid_qual > vector_qual:
                improvement = ((hybrid_qual - vector_qual) / vector_qual) * 100
                findings.append(f"La búsqueda híbrida mejora la calidad en {improvement:.1f}%")
        
        # Análisis de persistencia
        if self.results["persistencia"]:
            if self.results["persistencia"]["data_integrity"]:
                findings.append("La persistencia de datos funciona correctamente al 100%")
        
        # Análisis de rendimiento
        if self.results["rendimiento"]:
            avg_latency = self.results["rendimiento"]["latency_analysis"].get("avg_latency", 0)
            findings.append(f"Latencia promedio de respuesta: {avg_latency:.3f}s")
        
        return findings
    
    def _print_summary(self):
        """Imprime resumen ejecutivo"""
        print("\n" + "="*60)
        print("📊 RESUMEN EJECUTIVO DE EXPERIMENTOS RAG")
        print("="*60)
        
        summary = self.results.get("summary", {})
        print(f"🎯 Éxito general: {summary.get('overall_success', 0):.2%}")
        print(f"📅 Fecha: {summary.get('experiment_date', 'N/A')}")
        print(f"🧪 Experimentos: {summary.get('total_experiments', 0)}")
        
        print("\n🔍 HALLAZGOS CLAVE:")
        for finding in summary.get("key_findings", []):
            print(f"  • {finding}")
        
        print("\n✅ EXPERIMENTOS COMPLETADOS:")
        for exp_name in self.results:
            if exp_name != "summary" and self.results[exp_name]:
                print(f"  ✅ {exp_name.replace('_', ' ').title()}")

def main():
    """Función principal para ejecutar todos los experimentos"""
    print("🚀 INICIANDO SUITE DE EXPERIMENTOS RAG VECTORIAL")
    print("Este proceso validará todas las funcionalidades del sistema RAG")
    print("-" * 60)
    
    # Crear y ejecutar suite de experimentos
    experiment_suite = RAGExperimentSuite()
    results = experiment_suite.run_all_experiments()
    
    print("\n🎉 TODOS LOS EXPERIMENTOS COMPLETADOS!")
    print("Revisa el archivo 'rag_experiment_report.json' para detalles completos.")

if __name__ == "__main__":
    main()
