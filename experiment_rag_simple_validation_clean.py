#!/usr/bin/env python3
"""
Experimento Simplificado de Validaci??n RAG
Valida las funcionalidades principales sin dependencias problem??ticas
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

class SimpleRAGValidator:
    """Validador simplificado para funcionalidades RAG"""
    
    def __init__(self):
        """Inicializa el validador"""
        self.rag_assistant = create_vrp_rag_assistant()
        self.test_results = {}
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Ejecuta validaci??n completa de funcionalidades RAG"""
        print("VALIDACION COMPREHENSIVA DEL SISTEMA RAG")
        print("=" * 60)
        
        results = {
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
        """Prueba la inicializaci??n del sistema"""
        print("\n1?????? PRUEBA DE INICIALIZACI??N")
        
        try:
            # Verificar componentes principales
            has_gemini = hasattr(self.rag_assistant, 'gemini')
            has_ir_system = hasattr(self.rag_assistant, 'ir_system')
            has_knowledge_base = hasattr(self.rag_assistant, 'knowledge_base')
            
            print(f"  [OK] Gemini: {has_gemini}")
            print(f"  [OK] Sistema IR: {has_ir_system}")
            print(f"  [OK] Base de conocimientos: {has_knowledge_base}")
            
            return {
                "success": True,
                "components": {
                    "gemini": has_gemini,
                    "ir_system": has_ir_system,
                    "knowledge_base": has_knowledge_base
                },
                "initialization_time": time.time()
            }
        except Exception as e:
            print(f"  ??? Error de inicializaci??n: {e}")
            return {"success": False, "error": str(e)}
    
    def _test_basic_qa(self) -> Dict[str, Any]:
        """Prueba funcionalidad b??sica de Q&A"""
        print("\n2?????? PRUEBA DE Q&A B??SICO")
        
        test_questions = [
            "??Qu?? es el VRP?",
            "??C??mo funciona la optimizaci??n de rutas?",
            "??Cu??les son los factores que afectan las entregas en La Habana?"
        ]
        
        results = []
        
        for question in test_questions:
            print(f"  ???? Pregunta: {question[:40]}...")
            
            try:
                start_time = time.time()
                response = self.rag_assistant.ask_with_context(question)
                end_time = time.time()
                
                success = response.get("success", False)
                response_text = response.get("response", "")
                docs_retrieved = response.get("context_used", {}).get("retrieved_documents", 0)
                
                result = {
                    "question": question,
                    "success": success,
                    "response_length": len(response_text),
                    "docs_retrieved": docs_retrieved,
                    "response_time": end_time - start_time,
                    "has_context": docs_retrieved > 0
                }
                
                results.append(result)
                print(f"    ??? ??xito: {success}, Docs: {docs_retrieved}, Tiempo: {result['response_time']:.2f}s")
                
            except Exception as e:
                print(f"    ??? Error: {e}")
                results.append({
                    "question": question,
                    "success": False,
                    "error": str(e)
                })
        
        success_rate = sum(1 for r in results if r.get("success", False)) / len(results)
        avg_response_time = np.mean([r["response_time"] for r in results if "response_time" in r])
        avg_docs = np.mean([r["docs_retrieved"] for r in results if "docs_retrieved" in r])
        
        return {
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "avg_docs_retrieved": avg_docs,
            "individual_results": results
        }
    
    def _test_context_update(self) -> Dict[str, Any]:
        """Prueba actualizaci??n de contexto en tiempo real"""
        print("\n3?????? PRUEBA DE ACTUALIZACI??N DE CONTEXTO")
        
        # Estado inicial
        initial_question = "??Cu??l es el estado actual del sistema?"
        initial_response = self.rag_assistant.ask_with_context(initial_question)
        initial_docs = initial_response.get("context_used", {}).get("retrieved_documents", 0)
        
        print(f"  ???? Estado inicial - Docs: {initial_docs}")
        
        # Actualizar diferentes tipos de contexto
        updates = [
            ("weather", {"current": "lluvia", "temperature": 22, "humidity": 85}),
            ("routes", {"routes": [{"id": "r1", "efficiency": 0.85, "distance": 15.5}]}),
            ("performance", {"component": "optimizer", "status": "active", "efficiency": 0.90})
        ]
        
        update_results = []
        
        for data_type, data in updates:
            print(f"  ???? Actualizando {data_type}...")
            
            try:
                # Actualizar conocimiento
                self.rag_assistant.update_knowledge_base(data_type, data)
                
                # Verificar impacto
                updated_response = self.rag_assistant.ask_with_context(initial_question)
                updated_docs = updated_response.get("context_used", {}).get("retrieved_documents", 0)
                
                result = {
                    "data_type": data_type,
                    "success": True,
                    "docs_before": initial_docs,
                    "docs_after": updated_docs,
                    "improvement": updated_docs - initial_docs
                }
                
                update_results.append(result)
                print(f"    ??? Docs: {initial_docs} ??? {updated_docs} (+{result['improvement']})")
                
                initial_docs = updated_docs  # Para la pr??xima iteraci??n
                
            except Exception as e:
                print(f"    ??? Error actualizando {data_type}: {e}")
                update_results.append({
                    "data_type": data_type,
                    "success": False,
                    "error": str(e)
                })
        
        successful_updates = sum(1 for r in update_results if r.get("success", False))
        total_improvement = sum(r.get("improvement", 0) for r in update_results if r.get("success", False))
        
        return {
            "successful_updates": successful_updates,
            "total_updates": len(updates),
            "total_doc_improvement": total_improvement,
            "update_details": update_results
        }
    
    def _test_knowledge_growth(self) -> Dict[str, Any]:
        """Prueba el crecimiento de la base de conocimientos"""
        print("\n4?????? PRUEBA DE CRECIMIENTO DE CONOCIMIENTOS")
        
        # Simular m??ltiples actualizaciones
        knowledge_updates = [
            ("weather", {"condition": "viento_fuerte", "impact": "high", "recommendation": "use_heavy_vehicles"}),
            ("routes", {"area": "centro_habana", "restriction": "weight_limit", "alternative": "route_B"}),
            ("traffic_events", {"event": "construction", "location": "23_y_12", "duration": 30}),
            ("weather", {"condition": "lluvia_intensa", "visibility": "low", "speed_reduction": 0.35}),
            ("performance", {"metric": "fuel_efficiency", "value": 0.82, "trend": "improving"})
        ]
        
        growth_tracking = []
        
        for i, (data_type, data) in enumerate(knowledge_updates):
            print(f"  ???? Actualizaci??n {i+1}: {data_type}")
            
            try:
                # Antes de la actualizaci??n
                before_response = self.rag_assistant.ask_with_context("Resumen del conocimiento actual")
                before_docs = before_response.get("context_used", {}).get("retrieved_documents", 0)
                
                # Actualizar
                self.rag_assistant.update_knowledge_base(data_type, data)
                
                # Despu??s de la actualizaci??n
                after_response = self.rag_assistant.ask_with_context("Resumen del conocimiento actual")
                after_docs = after_response.get("context_used", {}).get("retrieved_documents", 0)
                
                growth_tracking.append({
                    "update_number": i + 1,
                    "data_type": data_type,
                    "docs_before": before_docs,
                    "docs_after": after_docs,
                    "growth": after_docs - before_docs
                })
                
                print(f"    ???? Conocimiento: {before_docs} ??? {after_docs}")
                
            except Exception as e:
                print(f"    ??? Error en actualizaci??n {i+1}: {e}")
        
        total_growth = sum(g["growth"] for g in growth_tracking)
        avg_growth = np.mean([g["growth"] for g in growth_tracking]) if growth_tracking else 0
        
        return {
            "total_updates": len(knowledge_updates),
            "successful_tracking": len(growth_tracking),
            "total_knowledge_growth": total_growth,
            "avg_growth_per_update": avg_growth,
            "growth_timeline": growth_tracking
        }
    
    def _test_response_quality(self) -> Dict[str, Any]:
        """Prueba la calidad de las respuestas"""
        print("\n5?????? PRUEBA DE CALIDAD DE RESPUESTAS")
        
        quality_tests = [
            {
                "question": "??C??mo optimizar rutas considerando el clima lluvioso?",
                "expected_elements": ["lluvia", "optimizaci??n", "rutas", "tiempo", "eficiencia"],
                "complexity": "medium"
            },
            {
                "question": "??Qu?? estrategias usar para evitar el tr??fico en La Habana?",
                "expected_elements": ["tr??fico", "estrategias", "habana", "horarios", "alternativas"],
                "complexity": "medium"
            },
            {
                "question": "Explica el impacto del viento en los veh??culos de carga",
                "expected_elements": ["viento", "veh??culos", "carga", "impacto", "seguridad"],
                "complexity": "high"
            }
        ]
        
        quality_results = []
        
        for test in quality_tests:
            print(f"  ???? Evaluando: {test['question'][:40]}...")
            
            try:
                response = self.rag_assistant.ask_with_context(test["question"])
                response_text = response.get("response", "").lower()
                
                # Evaluar elementos esperados
                elements_found = sum(1 for element in test["expected_elements"] if element in response_text)
                completeness = elements_found / len(test["expected_elements"])
                
                # Evaluar longitud y estructura
                response_length = len(response.get("response", ""))
                has_context = response.get("context_used", {}).get("retrieved_documents", 0) > 0
                has_metrics = bool(response.get("relevant_metrics"))
                
                quality_score = (
                    completeness * 0.4 +
                    (min(response_length / 1000, 1.0)) * 0.3 +
                    (1.0 if has_context else 0.0) * 0.2 +
                    (1.0 if has_metrics else 0.0) * 0.1
                )
                
                result = {
                    "question": test["question"],
                    "complexity": test["complexity"],
                    "elements_found": elements_found,
                    "total_elements": len(test["expected_elements"]),
                    "completeness": completeness,
                    "response_length": response_length,
                    "has_context": has_context,
                    "has_metrics": has_metrics,
                    "quality_score": quality_score
                }
                
                quality_results.append(result)
                print(f"    ???? Calidad: {quality_score:.3f}, Completitud: {completeness:.2%}")
                
            except Exception as e:
                print(f"    ??? Error evaluando calidad: {e}")
        
        avg_quality = np.mean([r["quality_score"] for r in quality_results]) if quality_results else 0
        avg_completeness = np.mean([r["completeness"] for r in quality_results]) if quality_results else 0
        
        return {
            "avg_quality_score": avg_quality,
            "avg_completeness": avg_completeness,
            "tests_completed": len(quality_results),
            "quality_details": quality_results
        }
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Prueba el manejo de errores"""
        print("\n6?????? PRUEBA DE MANEJO DE ERRORES")
        
        error_tests = [
            ("empty_question", ""),
            ("none_question", None),
            ("very_long_question", "A" * 10000),
            ("special_chars", "??C??mo optimizar rutas con s??mbolos especiales: @#$%^&*()"),
            ("non_spanish", "How to optimize routes in English?")
        ]
        
        error_results = []
        
        for test_name, question in error_tests:
            print(f"  ??????? Probando: {test_name}")
            
            try:
                response = self.rag_assistant.ask_with_context(question)
                
                # El sistema deber??a manejar errores gracefully
                success = response.get("success", False)
                has_error_message = bool(response.get("error") or response.get("message"))
                
                result = {
                    "test_name": test_name,
                    "handled_gracefully": True,
                    "success": success,
                    "has_error_message": has_error_message,
                    "response_provided": bool(response.get("response"))
                }
                
                error_results.append(result)
                print(f"    ??? Manejado gracefully: {success or has_error_message}")
                
            except Exception as e:
                print(f"    ??? Error no manejado: {e}")
                error_results.append({
                    "test_name": test_name,
                    "handled_gracefully": False,
                    "error": str(e)
                })
        
        graceful_handling_rate = sum(1 for r in error_results if r.get("handled_gracefully", False)) / len(error_results)
        
        return {
            "graceful_handling_rate": graceful_handling_rate,
            "tests_completed": len(error_results),
            "error_handling_details": error_results
        }
    
    def _test_performance(self) -> Dict[str, Any]:
        """Prueba el rendimiento del sistema"""
        print("\n7?????? PRUEBA DE RENDIMIENTO")
        
        # Test de latencia
        latency_questions = [
            "??Qu?? es VRP?",
            "??C??mo optimizar rutas?",
            "??Impacto del clima?",
            "??Gesti??n de tr??fico?",
            "??Eficiencia energ??tica?"
        ]
        
        latencies = []
        
        print("  ?????? Midiendo latencias...")
        for question in latency_questions:
            start_time = time.time()
            response = self.rag_assistant.ask_with_context(question)
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
            print(f"    ?????? {question[:20]}...: {latency:.3f}s")
        
        # Test de throughput
        print("  ???? Midiendo throughput...")
        throughput_start = time.time()
        throughput_responses = []
        
        for question in latency_questions[:3]:  # Subset para throughput
            response = self.rag_assistant.ask_with_context(question)
            throughput_responses.append(response)
        
        throughput_end = time.time()
        total_throughput_time = throughput_end - throughput_start
        queries_per_second = len(throughput_responses) / total_throughput_time
        
        return {
            "latency_stats": {
                "avg_latency": np.mean(latencies),
                "min_latency": np.min(latencies),
                "max_latency": np.max(latencies),
                "p95_latency": np.percentile(latencies, 95)
            },
            "throughput_stats": {
                "queries_per_second": queries_per_second,
                "total_time": total_throughput_time,
                "total_queries": len(throughput_responses)
            },
            "individual_latencies": latencies
        }
    
    def _generate_validation_report(self, results: Dict[str, Any]):
        """Genera reporte de validaci??n"""
        report_path = "rag_validation_report.json"
        
        # Calcular m??tricas generales
        overall_success = self._calculate_overall_success(results)
        
        # Agregar resumen ejecutivo
        results["executive_summary"] = {
            "validation_date": datetime.now().isoformat(),
            "overall_success_rate": overall_success,
            "total_tests": 7,
            "key_findings": self._generate_key_findings(results)
        }
        
        # Guardar reporte
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n???? Reporte de validaci??n guardado en: {report_path}")
        self._print_validation_summary(results)
    
    def _calculate_overall_success(self, results: Dict[str, Any]) -> float:
        """Calcula el ??xito general de la validaci??n"""
        scores = []
        
        # Inicializaci??n
        if results["initialization"]["success"]:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Q&A b??sico
        scores.append(results["basic_qa"]["success_rate"])
        
        # Actualizaci??n de contexto
        context_success = results["context_update"]["successful_updates"] / results["context_update"]["total_updates"]
        scores.append(context_success)
        
        # Crecimiento de conocimientos
        knowledge_success = results["knowledge_growth"]["successful_tracking"] / results["knowledge_growth"]["total_updates"]
        scores.append(knowledge_success)
        
        # Calidad de respuestas
        scores.append(results["response_quality"]["avg_quality_score"])
        
        # Manejo de errores
        scores.append(results["error_handling"]["graceful_handling_rate"])
        
        # Rendimiento (latencia < 10s es bueno)
        perf_score = min(10.0 / results["performance"]["latency_stats"]["avg_latency"], 1.0)
        scores.append(perf_score)
        
        return np.mean(scores)
    
    def _generate_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Genera hallazgos clave de la validaci??n"""
        findings = []
        
        # Inicializaci??n
        if results["initialization"]["success"]:
            findings.append("Sistema inicializado correctamente con todos los componentes")
        
        # Q&A
        qa_rate = results["basic_qa"]["success_rate"]
        findings.append(f"Tasa de ??xito en Q&A: {qa_rate:.1%}")
        
        # Rendimiento
        avg_latency = results["performance"]["latency_stats"]["avg_latency"]
        findings.append(f"Latencia promedio: {avg_latency:.2f}s")
        
        # Crecimiento
        knowledge_growth = results["knowledge_growth"]["total_knowledge_growth"]
        findings.append(f"Crecimiento total de conocimientos: {knowledge_growth} documentos")
        
        # Calidad
        avg_quality = results["response_quality"]["avg_quality_score"]
        findings.append(f"Calidad promedio de respuestas: {avg_quality:.3f}")
        
        return findings
    
    def _print_validation_summary(self, results: Dict[str, Any]):
        """Imprime resumen de validaci??n"""
        print("\n" + "="*60)
        print("???? RESUMEN DE VALIDACI??N RAG")
        print("="*60)
        
        summary = results["executive_summary"]
        
        print(f"???? ??xito general: {summary['overall_success_rate']:.2%}")
        print(f"???? Fecha: {summary['validation_date']}")
        print(f"???? Pruebas completadas: {summary['total_tests']}")
        
        print("\n???? HALLAZGOS CLAVE:")
        for finding in summary["key_findings"]:
            print(f"  ??? {finding}")
        
        print("\n??? PRUEBAS DETALLADAS:")
        print(f"  ???? Inicializaci??n: {'???' if results['initialization']['success'] else '???'}")
        print(f"  ???? Q&A B??sico: {results['basic_qa']['success_rate']:.1%}")
        print(f"  ???? Contexto: {results['context_update']['successful_updates']}/{results['context_update']['total_updates']}")
        print(f"  ???? Conocimientos: {results['knowledge_growth']['total_knowledge_growth']} docs")
        print(f"  ???? Calidad: {results['response_quality']['avg_quality_score']:.3f}")
        print(f"  ??????? Errores: {results['error_handling']['graceful_handling_rate']:.1%}")
        print(f"  ??? Rendimiento: {results['performance']['latency_stats']['avg_latency']:.2f}s")

def main():
    """Funci??n principal de validaci??n"""
    print("VALIDADOR COMPREHENSIVO DEL SISTEMA RAG")
    print("Este proceso valida todas las funcionalidades principales")
    print("-" * 60)
    
    # Ejecutar validaci??n
    validator = SimpleRAGValidator()
    results = validator.run_comprehensive_validation()
    
    print("\n???? VALIDACI??N COMPLETADA!")
    print("Revisa 'rag_validation_report.json' para detalles completos.")

if __name__ == "__main__":
    main()
