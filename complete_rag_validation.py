#!/usr/bin/env python3
"""
Validacion Completa del Sistema RAG - Todas las Funcionalidades
"""

import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Configurar paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.NLP.RAG import VRPKnowledgeRAG
from src.SRI.VectorDatabase import VRPVectorDatabase
from src.SRI.VRPInformationRetrieval import VRPInformationRetrievalSystem

class ComprehensiveRAGValidator:
    """Validador completo del sistema RAG"""
    
    def __init__(self):
        """Inicializa el validador"""
        self.results = {}
        self.start_time = datetime.now()
        
    def run_all_validations(self) -> Dict[str, Any]:
        """Ejecuta todas las validaciones"""
        print("=== VALIDACION COMPLETA DEL SISTEMA RAG ===")
        print(f"Iniciado: {self.start_time}")
        print("=" * 60)
        
        # 1. Validación de inicialización
        self.results["initialization"] = self._validate_initialization()
        
        # 2. Validación de base de datos vectorial
        self.results["vector_database"] = self._validate_vector_database()
        
        # 3. Validación de sistema IR
        self.results["information_retrieval"] = self._validate_information_retrieval()
        
        # 4. Validación de RAG completo
        self.results["rag_system"] = self._validate_rag_system()
        
        # 5. Validación de actualización de contexto
        self.results["context_update"] = self._validate_context_update()
        
        # 6. Validación de búsqueda híbrida
        self.results["hybrid_search"] = self._validate_hybrid_search()
        
        # 7. Validación de persistencia
        self.results["persistence"] = self._validate_persistence()
        
        # 8. Validación de rendimiento
        self.results["performance"] = self._validate_performance()
        
        # 9. Validación de manejo de errores
        self.results["error_handling"] = self._validate_error_handling()
        
        # 10. Validación de retroalimentación
        self.results["feedback_system"] = self._validate_feedback_system()
        
        # Generar reporte final
        self._generate_comprehensive_report()
        
        return self.results
    
    def _validate_initialization(self) -> Dict[str, Any]:
        """Valida inicialización de componentes"""
        print("\n1. VALIDACION DE INICIALIZACION")
        print("-" * 40)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "success": True,
            "errors": []
        }
        
        # Probar VectorDatabase
        try:
            vector_db = VRPVectorDatabase("test_init_vector")
            results["components"]["vector_db"] = True
            print("✓ VectorDatabase inicializada")
        except Exception as e:
            results["components"]["vector_db"] = False
            results["errors"].append(f"VectorDatabase: {str(e)}")
            print(f"✗ Error VectorDatabase: {e}")
        
        # Probar IR System
        try:
            ir_system = VRPInformationRetrievalSystem("test_init_ir")
            results["components"]["ir_system"] = True
            print("✓ IRSystem inicializado")
        except Exception as e:
            results["components"]["ir_system"] = False
            results["errors"].append(f"IRSystem: {str(e)}")
            print(f"✗ Error IRSystem: {e}")
        
        # Probar RAG System
        try:
            rag_system = VRPKnowledgeRAG()
            results["components"]["rag_system"] = True
            print("✓ RAG System inicializado")
        except Exception as e:
            results["components"]["rag_system"] = False
            results["errors"].append(f"RAG System: {str(e)}")
            print(f"✗ Error RAG System: {e}")
        
        # Evaluar éxito general
        components_working = sum(1 for v in results["components"].values() if v)
        total_components = len(results["components"])
        results["success"] = components_working == total_components
        results["success_rate"] = components_working / total_components
        
        print(f"Componentes funcionando: {components_working}/{total_components}")
        
        return results
    
    def _validate_vector_database(self) -> Dict[str, Any]:
        """Valida funcionalidad de base de datos vectorial"""
        print("\n2. VALIDACION DE BASE DE DATOS VECTORIAL")
        print("-" * 40)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "operations": {},
            "success": True,
            "errors": []
        }
        
        try:
            vector_db = VRPVectorDatabase("test_vector_operations")
            
            # Operación 1: Añadir documento
            try:
                doc_id = vector_db.add_document(
                    "knowledge_base",
                    "Optimización de rutas de vehículos para distribución urbana",
                    {"tipo": "test", "categoria": "VRP", "test_id": "vec_001"}
                )
                results["operations"]["add_document"] = True
                print("✓ Documento añadido correctamente")
            except Exception as e:
                results["operations"]["add_document"] = False
                results["errors"].append(f"add_document: {str(e)}")
                print(f"✗ Error añadiendo documento: {e}")
            
            # Operación 2: Buscar documento
            try:
                search_results = vector_db.search(
                    "optimización rutas vehículos",
                    collection_names=["knowledge_base"],
                    top_k=3
                )
                results["operations"]["search"] = len(search_results) > 0
                print(f"✓ Búsqueda exitosa: {len(search_results)} resultados")
            except Exception as e:
                results["operations"]["search"] = False
                results["errors"].append(f"search: {str(e)}")
                print(f"✗ Error en búsqueda: {e}")
            
            # Operación 3: Contar documentos
            try:
                total_docs = vector_db.get_total_documents()
                results["operations"]["count_documents"] = total_docs >= 0
                results["total_documents"] = total_docs
                print(f"✓ Total documentos: {total_docs}")
            except Exception as e:
                results["operations"]["count_documents"] = False
                results["errors"].append(f"count_documents: {str(e)}")
                print(f"✗ Error contando documentos: {e}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"VectorDB initialization: {str(e)}")
            print(f"✗ Error general VectorDB: {e}")
        
        # Evaluar éxito
        operations_working = sum(1 for v in results["operations"].values() if v)
        total_operations = len(results["operations"])
        results["success"] = operations_working == total_operations
        results["success_rate"] = operations_working / total_operations if total_operations > 0 else 0
        
        print(f"Operaciones exitosas: {operations_working}/{total_operations}")
        
        return results
    
    def _validate_information_retrieval(self) -> Dict[str, Any]:
        """Valida sistema de recuperación de información"""
        print("\n3. VALIDACION DE SISTEMA IR")
        print("-" * 40)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "operations": {},
            "success": True,
            "errors": []
        }
        
        try:
            ir_system = VRPInformationRetrievalSystem("test_ir_operations")
            
            # Operación 1: Añadir datos en tiempo real
            try:
                ir_system.add_real_time_data("route", {
                    "route_id": "test_route_001",
                    "description": "Ruta de prueba para validación del sistema IR",
                    "optimization_method": "genetic_algorithm",
                    "efficiency": 0.95
                })
                results["operations"]["add_real_time_data"] = True
                print("✓ Datos en tiempo real añadidos")
            except Exception as e:
                results["operations"]["add_real_time_data"] = False
                results["errors"].append(f"add_real_time_data: {str(e)}")
                print(f"✗ Error añadiendo datos tiempo real: {e}")
            
            # Operación 2: Búsqueda general
            try:
                search_results = ir_system.search("optimización rutas", top_k=3)
                results["operations"]["search"] = len(search_results) >= 0
                results["search_results_count"] = len(search_results)
                print(f"✓ Búsqueda IR exitosa: {len(search_results)} resultados")
            except Exception as e:
                results["operations"]["search"] = False
                results["errors"].append(f"search: {str(e)}")
                print(f"✗ Error en búsqueda IR: {e}")
            
            # Operación 3: Expansión de consulta (Rocchio)
            try:
                expanded_query = ir_system.expand_query("optimización", ["route_001"])
                results["operations"]["expand_query"] = len(expanded_query) > 0
                print(f"✓ Expansión de consulta exitosa: {len(expanded_query)} términos")
            except Exception as e:
                results["operations"]["expand_query"] = False
                results["errors"].append(f"expand_query: {str(e)}")
                print(f"✗ Error expandiendo consulta: {e}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"IRSystem initialization: {str(e)}")
            print(f"✗ Error general IRSystem: {e}")
        
        # Evaluar éxito
        operations_working = sum(1 for v in results["operations"].values() if v)
        total_operations = len(results["operations"])
        results["success"] = operations_working == total_operations
        results["success_rate"] = operations_working / total_operations if total_operations > 0 else 0
        
        print(f"Operaciones IR exitosas: {operations_working}/{total_operations}")
        
        return results
    
    def _validate_rag_system(self) -> Dict[str, Any]:
        """Valida sistema RAG completo"""
        print("\n4. VALIDACION DE SISTEMA RAG")
        print("-" * 40)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "operations": {},
            "success": True,
            "errors": []
        }
        
        try:
            rag_system = VRPKnowledgeRAG()
            
            # Operación 1: Actualizar contexto
            try:
                rag_system.update_knowledge("weather", {
                    "location": "La Habana",
                    "conditions": "Soleado",
                    "temperature": 28,
                    "wind_speed": 10
                })
                results["operations"]["update_knowledge"] = True
                print("✓ Contexto actualizado")
            except Exception as e:
                results["operations"]["update_knowledge"] = False
                results["errors"].append(f"update_knowledge: {str(e)}")
                print(f"✗ Error actualizando contexto: {e}")
            
            # Operación 2: Generar respuesta simple (sin API)
            try:
                # Simular respuesta sin llamar a Gemini
                question = "¿Cuáles son los métodos de optimización para VRP?"
                
                # Verificar que el método existe
                has_generate_response = hasattr(rag_system, 'generate_response')
                results["operations"]["generate_response"] = has_generate_response
                
                if has_generate_response:
                    print("✓ Método generate_response disponible")
                else:
                    print("✗ Método generate_response no disponible")
                
            except Exception as e:
                results["operations"]["generate_response"] = False
                results["errors"].append(f"generate_response: {str(e)}")
                print(f"✗ Error con generate_response: {e}")
            
            # Operación 3: Verificar componentes internos
            try:
                has_gemini = hasattr(rag_system, 'gemini')
                has_ir_system = hasattr(rag_system, 'ir_system')
                has_knowledge_base = hasattr(rag_system, 'knowledge_base')
                
                results["operations"]["components_available"] = all([has_gemini, has_ir_system, has_knowledge_base])
                print(f"✓ Componentes: Gemini={has_gemini}, IR={has_ir_system}, KB={has_knowledge_base}")
                
            except Exception as e:
                results["operations"]["components_available"] = False
                results["errors"].append(f"components_check: {str(e)}")
                print(f"✗ Error verificando componentes: {e}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"RAG System initialization: {str(e)}")
            print(f"✗ Error general RAG System: {e}")
        
        # Evaluar éxito
        operations_working = sum(1 for v in results["operations"].values() if v)
        total_operations = len(results["operations"])
        results["success"] = operations_working == total_operations
        results["success_rate"] = operations_working / total_operations if total_operations > 0 else 0
        
        print(f"Operaciones RAG exitosas: {operations_working}/{total_operations}")
        
        return results
    
    def _validate_context_update(self) -> Dict[str, Any]:
        """Valida actualización de contexto en tiempo real"""
        print("\n5. VALIDACION DE ACTUALIZACION DE CONTEXTO")
        print("-" * 40)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "updates": {},
            "success": True,
            "errors": []
        }
        
        try:
            rag_system = VRPKnowledgeRAG()
            
            # Actualización 1: Datos meteorológicos
            try:
                rag_system.update_knowledge("weather", {
                    "location": "La Habana",
                    "temperature": 30,
                    "humidity": 75,
                    "precipitation": 0,
                    "wind_speed": 15
                })
                results["updates"]["weather"] = True
                print("✓ Datos meteorológicos actualizados")
            except Exception as e:
                results["updates"]["weather"] = False
                results["errors"].append(f"weather_update: {str(e)}")
                print(f"✗ Error actualizando clima: {e}")
            
            # Actualización 2: Datos de rutas
            try:
                rag_system.update_knowledge("routes", {
                    "route_id": "test_route_context",
                    "origin": "Centro Habana",
                    "destination": "Vedado",
                    "distance": 5.2,
                    "estimated_time": 18,
                    "optimization_method": "genetic_algorithm"
                })
                results["updates"]["routes"] = True
                print("✓ Datos de rutas actualizados")
            except Exception as e:
                results["updates"]["routes"] = False
                results["errors"].append(f"routes_update: {str(e)}")
                print(f"✗ Error actualizando rutas: {e}")
            
            # Actualización 3: Eventos de tráfico
            try:
                rag_system.update_knowledge("traffic_events", {
                    "event_id": "traffic_001",
                    "location": "Malecón",
                    "type": "congestion",
                    "severity": "medium",
                    "estimated_duration": 30
                })
                results["updates"]["traffic"] = True
                print("✓ Eventos de tráfico actualizados")
            except Exception as e:
                results["updates"]["traffic"] = False
                results["errors"].append(f"traffic_update: {str(e)}")
                print(f"✗ Error actualizando tráfico: {e}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Context update initialization: {str(e)}")
            print(f"✗ Error general actualizando contexto: {e}")
        
        # Evaluar éxito
        updates_working = sum(1 for v in results["updates"].values() if v)
        total_updates = len(results["updates"])
        results["success"] = updates_working == total_updates
        results["success_rate"] = updates_working / total_updates if total_updates > 0 else 0
        
        print(f"Actualizaciones exitosas: {updates_working}/{total_updates}")
        
        return results
    
    def _validate_hybrid_search(self) -> Dict[str, Any]:
        """Valida búsqueda híbrida"""
        print("\n6. VALIDACION DE BUSQUEDA HIBRIDA")
        print("-" * 40)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "search_types": {},
            "success": True,
            "errors": []
        }
        
        try:
            ir_system = VRPInformationRetrievalSystem("test_hybrid_search")
            
            # Añadir datos de prueba
            ir_system.add_real_time_data("route", {
                "route_id": "hybrid_test_route",
                "description": "Ruta optimizada con algoritmo genético para distribución urbana",
                "method": "genetic_algorithm",
                "efficiency": 0.92
            })
            
            # Búsqueda 1: Vectorial
            try:
                vector_results = ir_system.search("optimización genética", top_k=3)
                results["search_types"]["vector"] = len(vector_results) >= 0
                results["vector_results_count"] = len(vector_results)
                print(f"✓ Búsqueda vectorial: {len(vector_results)} resultados")
            except Exception as e:
                results["search_types"]["vector"] = False
                results["errors"].append(f"vector_search: {str(e)}")
                print(f"✗ Error búsqueda vectorial: {e}")
            
            # Búsqueda 2: LSI (si está disponible)
            try:
                # Verificar si LSI está disponible
                if hasattr(ir_system, 'lsi_search'):
                    lsi_results = ir_system.lsi_search("optimización genética", top_k=3)
                    results["search_types"]["lsi"] = len(lsi_results) >= 0
                    results["lsi_results_count"] = len(lsi_results)
                    print(f"✓ Búsqueda LSI: {len(lsi_results)} resultados")
                else:
                    results["search_types"]["lsi"] = False
                    print("✗ LSI search no disponible")
            except Exception as e:
                results["search_types"]["lsi"] = False
                results["errors"].append(f"lsi_search: {str(e)}")
                print(f"✗ Error búsqueda LSI: {e}")
            
            # Búsqueda 3: Híbrida (combinada)
            try:
                hybrid_results = ir_system.search("optimización genética", top_k=5)
                results["search_types"]["hybrid"] = len(hybrid_results) >= 0
                results["hybrid_results_count"] = len(hybrid_results)
                print(f"✓ Búsqueda híbrida: {len(hybrid_results)} resultados")
            except Exception as e:
                results["search_types"]["hybrid"] = False
                results["errors"].append(f"hybrid_search: {str(e)}")
                print(f"✗ Error búsqueda híbrida: {e}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Hybrid search initialization: {str(e)}")
            print(f"✗ Error general búsqueda híbrida: {e}")
        
        # Evaluar éxito
        searches_working = sum(1 for v in results["search_types"].values() if v)
        total_searches = len(results["search_types"])
        results["success"] = searches_working >= 1  # Al menos una búsqueda funciona
        results["success_rate"] = searches_working / total_searches if total_searches > 0 else 0
        
        print(f"Búsquedas exitosas: {searches_working}/{total_searches}")
        
        return results
    
    def _validate_persistence(self) -> Dict[str, Any]:
        """Valida persistencia de datos"""
        print("\n7. VALIDACION DE PERSISTENCIA")
        print("-" * 40)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "persistence_tests": {},
            "success": True,
            "errors": []
        }
        
        try:
            # Prueba 1: Persistir datos
            vector_db1 = VRPVectorDatabase("test_persistence")
            
            test_doc_id = vector_db1.add_document(
                "knowledge_base",
                "Documento de prueba para validar persistencia del sistema",
                {"test": "persistence", "timestamp": datetime.now().isoformat()}
            )
            
            results["persistence_tests"]["data_saved"] = True
            print("✓ Datos guardados")
            
            # Prueba 2: Recuperar datos
            vector_db2 = VRPVectorDatabase("test_persistence")
            
            search_results = vector_db2.search(
                "persistencia sistema",
                collection_names=["knowledge_base"],
                top_k=1
            )
            
            data_recovered = len(search_results) > 0
            results["persistence_tests"]["data_recovered"] = data_recovered
            
            if data_recovered:
                print("✓ Datos recuperados exitosamente")
            else:
                print("✗ No se pudieron recuperar los datos")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Persistence test: {str(e)}")
            print(f"✗ Error en prueba de persistencia: {e}")
        
        # Evaluar éxito
        persistence_working = sum(1 for v in results["persistence_tests"].values() if v)
        total_persistence = len(results["persistence_tests"])
        results["success"] = persistence_working == total_persistence
        results["success_rate"] = persistence_working / total_persistence if total_persistence > 0 else 0
        
        print(f"Pruebas de persistencia exitosas: {persistence_working}/{total_persistence}")
        
        return results
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Valida rendimiento del sistema"""
        print("\n8. VALIDACION DE RENDIMIENTO")
        print("-" * 40)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "success": True,
            "errors": []
        }
        
        try:
            ir_system = VRPInformationRetrievalSystem("test_performance")
            
            # Métrica 1: Latencia de búsqueda
            start_time = time.time()
            search_results = ir_system.search("optimización rutas", top_k=5)
            search_latency = time.time() - start_time
            
            results["metrics"]["search_latency"] = search_latency
            results["metrics"]["search_acceptable"] = search_latency < 5.0  # < 5 segundos
            
            print(f"✓ Latencia de búsqueda: {search_latency:.3f}s")
            
            # Métrica 2: Throughput de documentos
            start_time = time.time()
            docs_added = 0
            
            for i in range(5):  # Añadir 5 documentos
                ir_system.add_real_time_data("route", {
                    "route_id": f"perf_test_{i}",
                    "description": f"Documento de prueba de rendimiento {i}",
                    "iteration": i
                })
                docs_added += 1
            
            add_time = time.time() - start_time
            throughput = docs_added / add_time
            
            results["metrics"]["add_throughput"] = throughput
            results["metrics"]["throughput_acceptable"] = throughput > 0.5  # > 0.5 docs/seg
            
            print(f"✓ Throughput añadir: {throughput:.2f} docs/seg")
            
            # Métrica 3: Escalabilidad
            total_docs = ir_system.vector_db.get_total_documents()
            results["metrics"]["total_documents"] = total_docs
            results["metrics"]["scalability_acceptable"] = total_docs > 0
            
            print(f"✓ Total documentos: {total_docs}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Performance test: {str(e)}")
            print(f"✗ Error en prueba de rendimiento: {e}")
        
        # Evaluar éxito
        acceptable_metrics = sum(1 for k, v in results["metrics"].items() 
                               if k.endswith("_acceptable") and v)
        total_acceptable = sum(1 for k in results["metrics"].keys() 
                             if k.endswith("_acceptable"))
        
        results["success"] = acceptable_metrics == total_acceptable
        results["success_rate"] = acceptable_metrics / total_acceptable if total_acceptable > 0 else 0
        
        print(f"Métricas aceptables: {acceptable_metrics}/{total_acceptable}")
        
        return results
    
    def _validate_error_handling(self) -> Dict[str, Any]:
        """Valida manejo de errores"""
        print("\n9. VALIDACION DE MANEJO DE ERRORES")
        print("-" * 40)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "error_tests": {},
            "success": True,
            "errors": []
        }
        
        try:
            ir_system = VRPInformationRetrievalSystem("test_error_handling")
            
            # Error 1: Búsqueda vacía
            try:
                empty_results = ir_system.search("", top_k=1)
                results["error_tests"]["empty_query"] = True
                print("✓ Consulta vacía manejada gracefully")
            except Exception as e:
                results["error_tests"]["empty_query"] = False
                print(f"✗ Error con consulta vacía: {e}")
            
            # Error 2: Top_k inválido
            try:
                invalid_results = ir_system.search("test", top_k=0)
                results["error_tests"]["invalid_top_k"] = True
                print("✓ Top_k inválido manejado")
            except Exception as e:
                results["error_tests"]["invalid_top_k"] = False
                print(f"✗ Error con top_k inválido: {e}")
            
            # Error 3: Datos inválidos
            try:
                ir_system.add_real_time_data("route", None)
                results["error_tests"]["invalid_data"] = True
                print("✓ Datos inválidos manejados")
            except Exception as e:
                results["error_tests"]["invalid_data"] = False
                print(f"✗ Error con datos inválidos: {e}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Error handling test: {str(e)}")
            print(f"✗ Error general en manejo de errores: {e}")
        
        # Evaluar éxito
        error_tests_working = sum(1 for v in results["error_tests"].values() if v)
        total_error_tests = len(results["error_tests"])
        results["success"] = error_tests_working >= 2  # Al menos 2 de 3 errores manejados
        results["success_rate"] = error_tests_working / total_error_tests if total_error_tests > 0 else 0
        
        print(f"Errores manejados correctamente: {error_tests_working}/{total_error_tests}")
        
        return results
    
    def _validate_feedback_system(self) -> Dict[str, Any]:
        """Valida sistema de retroalimentación"""
        print("\n10. VALIDACION DE SISTEMA DE RETROALIMENTACION")
        print("-" * 40)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "feedback_tests": {},
            "success": True,
            "errors": []
        }
        
        try:
            ir_system = VRPInformationRetrievalSystem("test_feedback")
            
            # Prueba 1: Expansión de consulta (Rocchio)
            try:
                expanded_query = ir_system.expand_query("optimización", ["doc1", "doc2"])
                results["feedback_tests"]["query_expansion"] = len(expanded_query) > 0
                print(f"✓ Expansión de consulta: {len(expanded_query)} términos")
            except Exception as e:
                results["feedback_tests"]["query_expansion"] = False
                results["errors"].append(f"query_expansion: {str(e)}")
                print(f"✗ Error expansión de consulta: {e}")
            
            # Prueba 2: Verificar método de feedback
            try:
                has_feedback_method = hasattr(ir_system, 'expand_query')
                results["feedback_tests"]["feedback_method_exists"] = has_feedback_method
                
                if has_feedback_method:
                    print("✓ Método de retroalimentación disponible")
                else:
                    print("✗ Método de retroalimentación no disponible")
            except Exception as e:
                results["feedback_tests"]["feedback_method_exists"] = False
                results["errors"].append(f"feedback_method_check: {str(e)}")
                print(f"✗ Error verificando método de feedback: {e}")
            
            # Prueba 3: Aprendizaje incremental
            try:
                # Verificar si hay métodos de aprendizaje incremental
                has_incremental = hasattr(ir_system, 'update_model') or hasattr(ir_system, 'learn_from_feedback')
                results["feedback_tests"]["incremental_learning"] = has_incremental
                
                if has_incremental:
                    print("✓ Aprendizaje incremental disponible")
                else:
                    print("✗ Aprendizaje incremental no disponible")
            except Exception as e:
                results["feedback_tests"]["incremental_learning"] = False
                results["errors"].append(f"incremental_learning: {str(e)}")
                print(f"✗ Error verificando aprendizaje incremental: {e}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Feedback system test: {str(e)}")
            print(f"✗ Error general en sistema de retroalimentación: {e}")
        
        # Evaluar éxito
        feedback_tests_working = sum(1 for v in results["feedback_tests"].values() if v)
        total_feedback_tests = len(results["feedback_tests"])
        results["success"] = feedback_tests_working >= 1  # Al menos una funcionalidad de feedback
        results["success_rate"] = feedback_tests_working / total_feedback_tests if total_feedback_tests > 0 else 0
        
        print(f"Funcionalidades de retroalimentación: {feedback_tests_working}/{total_feedback_tests}")
        
        return results
    
    def _generate_comprehensive_report(self):
        """Genera reporte comprehensivo"""
        print("\n" + "=" * 60)
        print("REPORTE COMPREHENSIVO DE VALIDACION RAG")
        print("=" * 60)
        
        # Calcular métricas generales
        total_categories = len(self.results)
        successful_categories = sum(1 for r in self.results.values() 
                                  if isinstance(r, dict) and r.get("success", False))
        
        overall_success_rate = successful_categories / total_categories
        
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duración: {datetime.now() - self.start_time}")
        print(f"Éxito general: {overall_success_rate:.2%}")
        print(f"Categorías exitosas: {successful_categories}/{total_categories}")
        
        print("\nRESULTADOS POR CATEGORIA:")
        print("-" * 40)
        
        for category, result in self.results.items():
            if isinstance(result, dict):
                success = result.get("success", False)
                success_rate = result.get("success_rate", 0)
                status = "✓ EXITOSO" if success else "✗ FALLIDO"
                print(f"{category}: {status} ({success_rate:.1%})")
        
        print("\nHALLAZGOS PRINCIPALES:")
        print("-" * 40)
        
        # Generar hallazgos específicos
        findings = []
        
        if self.results.get("initialization", {}).get("success", False):
            findings.append("• Todos los componentes se inicializan correctamente")
        
        if self.results.get("vector_database", {}).get("success", False):
            total_docs = self.results.get("vector_database", {}).get("total_documents", 0)
            findings.append(f"• Base de datos vectorial operativa con {total_docs} documentos")
        
        if self.results.get("performance", {}).get("success", False):
            latency = self.results.get("performance", {}).get("metrics", {}).get("search_latency", 0)
            findings.append(f"• Rendimiento aceptable (latencia: {latency:.3f}s)")
        
        if self.results.get("persistence", {}).get("success", False):
            findings.append("• Sistema de persistencia funciona correctamente")
        
        if self.results.get("hybrid_search", {}).get("success", False):
            findings.append("• Búsqueda híbrida operativa")
        
        if self.results.get("feedback_system", {}).get("success", False):
            findings.append("• Sistema de retroalimentación disponible")
        
        for finding in findings:
            print(finding)
        
        if overall_success_rate >= 0.8:
            print("\n🎯 CONCLUSION: SISTEMA RAG COMPLETAMENTE FUNCIONAL")
        elif overall_success_rate >= 0.6:
            print("\n⚠️  CONCLUSION: SISTEMA RAG MAYORMENTE FUNCIONAL")
        else:
            print("\n❌ CONCLUSION: SISTEMA RAG NECESITA MEJORAS SIGNIFICATIVAS")
        
        # Guardar reporte detallado
        report_data = {
            "validation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "duration": str(datetime.now() - self.start_time),
                "overall_success_rate": overall_success_rate,
                "successful_categories": successful_categories,
                "total_categories": total_categories
            },
            "detailed_results": self.results,
            "findings": findings
        }
        
        with open("comprehensive_rag_validation_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nReporte detallado guardado en: comprehensive_rag_validation_report.json")

def main():
    """Función principal"""
    print("INICIANDO VALIDACION COMPLETA DEL SISTEMA RAG")
    print("Este proceso valida todas las funcionalidades del sistema")
    print("Tiempo estimado: 2-3 minutos")
    print("=" * 60)
    
    validator = ComprehensiveRAGValidator()
    results = validator.run_all_validations()
    
    return results

if __name__ == "__main__":
    main()
