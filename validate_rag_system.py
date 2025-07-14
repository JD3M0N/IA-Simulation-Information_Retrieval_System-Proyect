#!/usr/bin/env python3
"""
Validacion Comprehensiva del Sistema RAG - VRP Information Retrieval
Valida todas las funcionalidades principales sin caracteres unicode problematicos
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

from src.NLP.RAG import create_vrp_rag_assistant

class RAGSystemValidator:
    """Validador comprehensivo para funcionalidades RAG"""
    
    def __init__(self):
        """Inicializa el validador"""
        print("Inicializando validador del sistema RAG...")
        self.rag_assistant = create_vrp_rag_assistant()
        self.test_results = {}
        
    def run_complete_validation(self) -> Dict[str, Any]:
        """Ejecuta validacion completa de funcionalidades RAG"""
        print("\nVALIDACION COMPREHENSIVA DEL SISTEMA RAG")
        print("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "initialization": self._test_initialization(),
            "basic_qa": self._test_basic_qa(),
            "context_update": self._test_context_update(),
            "knowledge_growth": self._test_knowledge_growth(),
            "response_quality": self._test_response_quality(),
            "error_handling": self._test_error_handling(),
            "performance": self._test_performance()
        }
        
        # Generar reporte
        self._generate_validation_report(results)
        
        return results
    
    def _test_initialization(self) -> Dict[str, Any]:
        """Prueba la inicializacion del sistema"""
        print("\n1. PRUEBA DE INICIALIZACION")
        print("-" * 40)
        
        try:
            # Verificar componentes principales
            has_gemini = hasattr(self.rag_assistant, 'gemini_client')
            has_ir_system = hasattr(self.rag_assistant, 'ir_system') 
            has_knowledge_base = hasattr(self.rag_assistant, 'knowledge_base')
            
            print(f"  [OK] Gemini: {has_gemini}")
            print(f"  [OK] Sistema IR: {has_ir_system}")
            print(f"  [OK] Base de conocimientos: {has_knowledge_base}")
            
            return {
                "success": True,
                "components": {
                    "gemini_available": has_gemini,
                    "ir_system_available": has_ir_system,
                    "knowledge_base_available": has_knowledge_base
                }
            }
        except Exception as e:
            print(f"  [ERROR] Error de inicializacion: {e}")
            return {"success": False, "error": str(e)}
    
    def _test_basic_qa(self) -> Dict[str, Any]:
        """Prueba funcionalidad basica de Q&A"""
        print("\n2. PRUEBA DE Q&A BASICO")
        print("-" * 40)
        
        test_questions = [
            "Como optimizar rutas de vehiculos?",
            "Que factores afectan el trafico urbano?",
            "Cuales son las mejores practicas para VRP?",
            "Como manejar restricciones de tiempo en rutas?"
        ]
        
        results = []
        total_success = 0
        
        for question in test_questions:
            print(f"  [Q] Pregunta: {question[:40]}...")
            
            try:
                start_time = time.time()
                response = self.rag_assistant.generate_response(question)
                response_time = time.time() - start_time
                
                success = bool(response and len(response) > 50)
                docs_retrieved = response.get('sources_count', 0) if isinstance(response, dict) else 0
                
                result = {
                    "question": question,
                    "success": success,
                    "response_time": response_time,
                    "docs_retrieved": docs_retrieved,
                    "response_length": len(str(response))
                }
                
                results.append(result)
                if success:
                    total_success += 1
                    
                print(f"    [OK] Exito: {success}, Docs: {docs_retrieved}, Tiempo: {response_time:.2f}s")
                
            except Exception as e:
                print(f"    [ERROR] Error: {e}")
                results.append({"question": question, "success": False, "error": str(e)})
        
        return {
            "success_rate": total_success / len(test_questions),
            "total_questions": len(test_questions),
            "successful_answers": total_success,
            "individual_results": results
        }
    
    def _test_context_update(self) -> Dict[str, Any]:
        """Prueba actualizacion de contexto"""
        print("\n3. PRUEBA DE ACTUALIZACION DE CONTEXTO")
        print("-" * 40)
        
        try:
            # Obtener estado inicial
            initial_docs = self._count_total_documents()
            print(f"  [INFO] Estado inicial - Docs: {initial_docs}")
            
            # Datos de prueba para actualizar
            test_updates = [
                {
                    "type": "route_analysis",
                    "data": {
                        "route_id": "test_route_validation",
                        "optimization_type": "validation_test",
                        "results": "Prueba de validacion del sistema RAG",
                        "performance_metrics": {"efficiency": 0.95, "time_saved": "15%"}
                    }
                },
                {
                    "type": "weather_data", 
                    "data": {
                        "location": "test_location_validation",
                        "conditions": "Condiciones de prueba para validacion",
                        "impact_analysis": "Impacto minimo en rutas de validacion"
                    }
                },
                {
                    "type": "traffic_events",
                    "data": {
                        "event_id": "validation_event",
                        "description": "Evento de trafico para pruebas de validacion",
                        "route_impact": "Impacto controlado para testing"
                    }
                }
            ]
            
            results = []
            
            for update_data in test_updates:
                data_type = update_data["type"]
                
                try:
                    # Actualizar contexto
                    self.rag_assistant.update_context(data_type, update_data["data"])
                    
                    # Verificar cambios
                    updated_docs = self._count_total_documents()
                    improvement = updated_docs - initial_docs
                    
                    result = {
                        "data_type": data_type,
                        "success": True,
                        "improvement": improvement,
                        "docs_before": initial_docs,
                        "docs_after": updated_docs
                    }
                    
                    results.append(result)
                    print(f"    [OK] Docs: {initial_docs} -> {updated_docs} (+{improvement})")
                    initial_docs = updated_docs
                    
                except Exception as e:
                    print(f"    [ERROR] Error actualizando {data_type}: {e}")
                    results.append({"data_type": data_type, "success": False, "error": str(e)})
            
            success_rate = sum(1 for r in results if r.get("success", False)) / len(results)
            
            return {
                "success_rate": success_rate,
                "total_updates": len(test_updates),
                "successful_updates": sum(1 for r in results if r.get("success", False)),
                "individual_results": results
            }
            
        except Exception as e:
            print(f"  [ERROR] Error en prueba de contexto: {e}")
            return {"success": False, "error": str(e)}
    
    def _test_knowledge_growth(self) -> Dict[str, Any]:
        """Prueba crecimiento de base de conocimientos"""
        print("\n4. PRUEBA DE CRECIMIENTO DE CONOCIMIENTOS")
        print("-" * 40)
        
        try:
            initial_docs = self._count_total_documents()
            print(f"  [INFO] Documentos iniciales: {initial_docs}")
            
            # Simular multiples actualizaciones
            knowledge_updates = [
                {"tipo": "nueva_tecnica", "contenido": "Tecnica avanzada de optimizacion de rutas"},
                {"tipo": "caso_estudio", "contenido": "Estudio de caso real de implementacion VRP"},
                {"tipo": "mejores_practicas", "contenido": "Practicas recomendadas para sistemas de distribucion"},
                {"tipo": "metricas_rendimiento", "contenido": "Nuevas metricas para evaluar eficiencia de rutas"},
                {"tipo": "algoritmos_innovadores", "contenido": "Algoritmos emergentes en optimizacion logistica"}
            ]
            
            results = []
            
            for i, update in enumerate(knowledge_updates):
                try:
                    before_docs = self._count_total_documents()
                    
                    # Agregar conocimiento
                    self.rag_assistant.knowledge_base.add_document(
                        f"knowledge_growth_test_{i}",
                        update["contenido"],
                        {"categoria": update["tipo"], "test_iteration": i}
                    )
                    
                    after_docs = self._count_total_documents()
                    
                    result = {
                        "iteration": i + 1,
                        "knowledge_type": update["tipo"],
                        "success": True,
                        "docs_added": after_docs - before_docs,
                        "total_docs": after_docs
                    }
                    
                    results.append(result)
                    print(f"    [INFO] Conocimiento: {before_docs} -> {after_docs}")
                    
                except Exception as e:
                    print(f"    [ERROR] Error en actualizacion {i+1}: {e}")
                    results.append({"iteration": i + 1, "success": False, "error": str(e)})
            
            final_docs = self._count_total_documents()
            total_growth = final_docs - initial_docs
            
            return {
                "initial_documents": initial_docs,
                "final_documents": final_docs,
                "total_growth": total_growth,
                "growth_rate": total_growth / initial_docs if initial_docs > 0 else 0,
                "successful_updates": sum(1 for r in results if r.get("success", False)),
                "individual_results": results
            }
            
        except Exception as e:
            print(f"  [ERROR] Error en prueba de crecimiento: {e}")
            return {"success": False, "error": str(e)}
    
    def _test_response_quality(self) -> Dict[str, Any]:
        """Prueba calidad de respuestas"""
        print("\n5. PRUEBA DE CALIDAD DE RESPUESTAS")
        print("-" * 40)
        
        quality_tests = [
            {
                "question": "Explica los principales algoritmos para VRP",
                "expected_keywords": ["algoritmo", "optimizacion", "ruta", "vehiculo", "distancia"],
                "min_length": 100
            },
            {
                "question": "Como afecta el clima a las rutas de entrega?",
                "expected_keywords": ["clima", "tiempo", "ruta", "entrega", "impacto"],
                "min_length": 80
            },
            {
                "question": "Cuales son las mejores practicas para reducir costos de transporte?",
                "expected_keywords": ["costo", "transporte", "reducir", "optimizar", "eficiencia"],
                "min_length": 120
            }
        ]
        
        results = []
        total_quality_score = 0
        
        for test in quality_tests:
            print(f"  [TEST] Evaluando: {test['question'][:40]}...")
            
            try:
                # Generar respuesta
                response = self.rag_assistant.generate_response(test["question"])
                response_text = str(response)
                
                # Evaluar calidad
                length_score = min(len(response_text) / test["min_length"], 1.0)
                
                # Verificar palabras clave
                keyword_hits = sum(1 for kw in test["expected_keywords"] 
                                 if kw.lower() in response_text.lower())
                keyword_score = keyword_hits / len(test["expected_keywords"])
                
                # Puntaje combinado
                quality_score = (length_score * 0.4 + keyword_score * 0.6)
                total_quality_score += quality_score
                
                # Completitud (longitud vs minimo esperado)
                completeness = len(response_text) / test["min_length"]
                
                result = {
                    "question": test["question"],
                    "success": True,
                    "quality_score": quality_score,
                    "length_score": length_score,
                    "keyword_score": keyword_score,
                    "completeness": completeness,
                    "response_length": len(response_text),
                    "keyword_hits": keyword_hits,
                    "total_keywords": len(test["expected_keywords"])
                }
                
                results.append(result)
                print(f"    [SCORE] Calidad: {quality_score:.3f}, Completitud: {completeness:.2%}")
                
            except Exception as e:
                print(f"    [ERROR] Error evaluando calidad: {e}")
                results.append({"question": test["question"], "success": False, "error": str(e)})
        
        avg_quality = total_quality_score / len(quality_tests) if quality_tests else 0
        
        return {
            "average_quality_score": avg_quality,
            "total_tests": len(quality_tests),
            "successful_evaluations": sum(1 for r in results if r.get("success", False)),
            "individual_results": results
        }
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Prueba manejo de errores"""
        print("\n6. PRUEBA DE MANEJO DE ERRORES")
        print("-" * 40)
        
        error_tests = [
            {"input": "", "description": "Entrada vacia"},
            {"input": "?" * 1000, "description": "Entrada extremadamente larga"},
            {"input": None, "description": "Entrada nula"},
            {"input": "🚀🔍📊", "description": "Caracteres especiales"},
            {"input": "query sin sentido xyzabc123", "description": "Query sin contexto relevante"}
        ]
        
        results = []
        graceful_failures = 0
        
        for test in error_tests:
            print(f"  [TEST] {test['description']}")
            
            try:
                response = self.rag_assistant.generate_response(test["input"])
                success = response is not None
                has_error_message = "error" in str(response).lower() if response else False
                
                result = {
                    "test": test["description"],
                    "input": str(test["input"])[:50] + "..." if len(str(test["input"])) > 50 else str(test["input"]),
                    "success": success,
                    "graceful_failure": success or has_error_message,
                    "response_provided": bool(response)
                }
                
                results.append(result)
                if success or has_error_message:
                    graceful_failures += 1
                    
                print(f"    [OK] Manejado gracefully: {success or has_error_message}")
                
            except Exception as e:
                print(f"    [ERROR] Error no manejado: {e}")
                results.append({
                    "test": test["description"],
                    "success": False,
                    "error": str(e),
                    "graceful_failure": False
                })
        
        return {
            "graceful_failure_rate": graceful_failures / len(error_tests),
            "total_tests": len(error_tests),
            "graceful_failures": graceful_failures,
            "individual_results": results
        }
    
    def _test_performance(self) -> Dict[str, Any]:
        """Prueba rendimiento del sistema"""
        print("\n7. PRUEBA DE RENDIMIENTO")
        print("-" * 40)
        
        try:
            # Prueba de latencia
            latency_tests = [
                "Optimizacion de rutas para vehiculos",
                "Analisis de trafico urbano",
                "Planificacion logistica eficiente"
            ]
            
            latencies = []
            for query in latency_tests:
                start_time = time.time()
                self.rag_assistant.generate_response(query)
                latency = time.time() - start_time
                latencies.append(latency)
            
            avg_latency = np.mean(latencies)
            max_latency = np.max(latencies)
            min_latency = np.min(latencies)
            
            print(f"  [PERF] Latencia promedio: {avg_latency:.2f}s")
            print(f"  [PERF] Latencia maxima: {max_latency:.2f}s")
            print(f"  [PERF] Latencia minima: {min_latency:.2f}s")
            
            # Prueba de throughput
            print("  [TEST] Midiendo throughput...")
            start_time = time.time()
            queries_processed = 0
            
            for i in range(5):  # Procesar 5 queries rapidas
                self.rag_assistant.generate_response(f"Query de throughput {i}")
                queries_processed += 1
            
            total_time = time.time() - start_time
            throughput = queries_processed / total_time
            
            print(f"  [PERF] Throughput: {throughput:.2f} queries/segundo")
            
            # Prueba de escalabilidad (documentos)
            initial_docs = self._count_total_documents()
            
            performance_acceptable = (
                avg_latency < 10.0 and  # Menos de 10 segundos promedio
                throughput > 0.1 and   # Al menos 0.1 queries por segundo
                initial_docs > 0       # Base de datos no vacia
            )
            
            return {
                "success": True,
                "performance_acceptable": performance_acceptable,
                "latency": {
                    "average": avg_latency,
                    "maximum": max_latency,
                    "minimum": min_latency,
                    "samples": latencies
                },
                "throughput": throughput,
                "total_documents": initial_docs,
                "queries_processed": queries_processed,
                "total_test_time": total_time
            }
            
        except Exception as e:
            print(f"  [ERROR] Error en prueba de rendimiento: {e}")
            return {"success": False, "error": str(e)}
    
    def _count_total_documents(self) -> int:
        """Cuenta total de documentos en la base de conocimientos"""
        try:
            if hasattr(self.rag_assistant, 'knowledge_base'):
                return self.rag_assistant.knowledge_base.get_total_documents()
            else:
                return 0
        except:
            return 0
    
    def _generate_validation_report(self, results: Dict[str, Any]):
        """Genera reporte de validacion"""
        print("\n" + "=" * 60)
        print("RESUMEN DE VALIDACION RAG")
        print("=" * 60)
        
        # Calcular metricas generales
        summary = self._calculate_summary_metrics(results)
        
        print(f"[RESUMEN] Exito general: {summary['overall_success_rate']:.2%}")
        print(f"[RESUMEN] Pruebas exitosas: {summary['successful_tests']}/{summary['total_tests']}")
        print(f"[RESUMEN] Tiempo total: {summary.get('total_time', 'N/A')}")
        
        print("\n[HALLAZGOS] HALLAZGOS CLAVE:")
        for finding in summary.get('key_findings', []):
            print(f"  - {finding}")
        
        # Guardar reporte detallado
        report_file = "rag_validation_report.json"
        full_report = {
            "validation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "validator_version": "1.0",
                "system_platform": sys.platform
            },
            "summary": summary,
            "detailed_results": results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n[REPORTE] Reporte detallado guardado en: {report_file}")
    
    def _calculate_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula metricas resumen de la validacion"""
        
        successful_tests = 0
        total_tests = 0
        key_findings = []
        
        # Analizar cada categoria de prueba
        for test_name, test_result in results.items():
            if test_name == "timestamp":
                continue
                
            total_tests += 1
            
            if isinstance(test_result, dict):
                # Determinar si la prueba fue exitosa
                success = test_result.get('success', False)
                
                if 'success_rate' in test_result:
                    success = test_result['success_rate'] > 0.7  # 70% threshold
                elif 'performance_acceptable' in test_result:
                    success = test_result['performance_acceptable']
                
                if success:
                    successful_tests += 1
                
                # Generar hallazgos especificos
                if test_name == "basic_qa" and 'success_rate' in test_result:
                    rate = test_result['success_rate']
                    key_findings.append(f"Q&A exitoso en {rate:.1%} de consultas")
                
                elif test_name == "knowledge_growth" and 'total_growth' in test_result:
                    growth = test_result['total_growth']
                    key_findings.append(f"Base de conocimientos crecio en {growth} documentos")
                
                elif test_name == "performance" and 'latency' in test_result:
                    avg_latency = test_result['latency']['average']
                    key_findings.append(f"Latencia promedio: {avg_latency:.2f}s")
        
        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        return {
            "overall_success_rate": overall_success_rate,
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "key_findings": key_findings,
            "validation_status": "EXITOSA" if overall_success_rate >= 0.7 else "NECESITA_MEJORAS"
        }

def main():
    """Funcion principal de validacion"""
    print("VALIDADOR COMPREHENSIVO DEL SISTEMA RAG")
    print("Este proceso valida todas las funcionalidades principales")
    print("-" * 60)
    
    try:
        # Crear y ejecutar validador
        validator = RAGSystemValidator()
        results = validator.run_complete_validation()
        
        print("\n[FINALIZADO] Validacion completada exitosamente!")
        return results
        
    except Exception as e:
        print(f"\n[ERROR CRITICO] Error durante validacion: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
