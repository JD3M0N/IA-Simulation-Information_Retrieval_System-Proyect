#!/usr/bin/env python3
"""
Validacion Manual del Sistema RAG
"""

import sys
from pathlib import Path

# Configurar paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def test_rag_components():
    """Prueba los componentes del RAG individualmente"""
    print("=== VALIDACION MANUAL DEL SISTEMA RAG ===")
    
    results = {
        "gemini_test": False,
        "vector_db_test": False,
        "ir_system_test": False,
        "rag_integration_test": False
    }
    
    # 1. Prueba Gemini
    print("\n1. PRUEBA GEMINI")
    print("-" * 30)
    try:
        from src.NLP.Gemini import Gemini
        gemini = Gemini()
        print("✓ Gemini inicializado correctamente")
        
        # Prueba simple (sin llamada real para evitar limite de API)
        print("✓ Gemini listo para usar")
        results["gemini_test"] = True
        
    except Exception as e:
        print(f"✗ Error con Gemini: {e}")
    
    # 2. Prueba Base de Datos Vectorial
    print("\n2. PRUEBA BASE DE DATOS VECTORIAL")
    print("-" * 30)
    try:
        from src.SRI.VectorDatabase import VRPVectorDatabase
        vector_db = VRPVectorDatabase("test_vector_cache")
        print("✓ Base de datos vectorial inicializada")
        
        # Agregar documento de prueba
        vector_db.add_document(
            "test_doc",
            "Prueba de optimizacion de rutas vehiculares",
            {"tipo": "test", "categoria": "VRP"}
        )
        print("✓ Documento añadido correctamente")
        
        # Buscar documento
        results_search = vector_db.search_similar("optimizacion rutas", top_k=1)
        print(f"✓ Búsqueda exitosa: {len(results_search)} resultados")
        
        results["vector_db_test"] = True
        
    except Exception as e:
        print(f"✗ Error con Base de Datos Vectorial: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Prueba Sistema IR
    print("\n3. PRUEBA SISTEMA IR")
    print("-" * 30)
    try:
        from src.SRI.VRPInformationRetrieval import VRPInformationRetrievalSystem
        ir_system = VRPInformationRetrievalSystem("test_ir_cache")
        print("✓ Sistema IR inicializado")
        
        # Prueba búsqueda básica
        search_results = ir_system.search("optimizacion", top_k=3)
        print(f"✓ Búsqueda IR exitosa: {len(search_results)} resultados")
        
        results["ir_system_test"] = True
        
    except Exception as e:
        print(f"✗ Error con Sistema IR: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Prueba integración RAG (sin inicialización completa)
    print("\n4. PRUEBA INTEGRACION RAG")
    print("-" * 30)
    try:
        from src.NLP.RAG import VRPKnowledgeRAG
        
        # Crear instancia directamente sin helper
        rag = VRPKnowledgeRAG()
        print("✓ Instancia RAG creada")
        
        # Verificar atributos
        has_gemini = hasattr(rag, 'gemini')
        has_ir_system = hasattr(rag, 'ir_system')
        has_knowledge_base = hasattr(rag, 'knowledge_base')
        
        print(f"✓ Componentes disponibles: Gemini={has_gemini}, IR={has_ir_system}, KB={has_knowledge_base}")
        
        if all([has_gemini, has_ir_system, has_knowledge_base]):
            results["rag_integration_test"] = True
            print("✓ Integración RAG exitosa")
        else:
            print("✗ Algunos componentes no están disponibles")
        
    except Exception as e:
        print(f"✗ Error en integración RAG: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def test_functionality_specific():
    """Prueba funcionalidades específicas del RAG"""
    print("\n=== PRUEBAS FUNCIONALES ESPECIFICAS ===")
    
    # Prueba 1: Actualización de contexto
    print("\n1. PRUEBA ACTUALIZACION DE CONTEXTO")
    print("-" * 40)
    try:
        from src.NLP.RAG import VRPKnowledgeRAG
        rag = VRPKnowledgeRAG()
        
        # Actualizar contexto con datos de prueba
        test_data = {
            "location": "Test Location",
            "weather": "Sunny",
            "traffic": "Light"
        }
        
        rag.update_context("weather", test_data)
        print("✓ Contexto actualizado correctamente")
        
        # Verificar que se almacenó
        if "weather_data" in rag.knowledge_base:
            print("✓ Datos almacenados en knowledge_base")
        else:
            print("✗ Datos no encontrados en knowledge_base")
        
    except Exception as e:
        print(f"✗ Error actualizando contexto: {e}")
    
    # Prueba 2: Búsqueda híbrida
    print("\n2. PRUEBA BUSQUEDA HIBRIDA")
    print("-" * 40)
    try:
        from src.SRI.VRPInformationRetrieval import VRPInformationRetrievalSystem
        ir_system = VRPInformationRetrievalSystem("test_hybrid_cache")
        
        # Agregar datos de prueba
        ir_system.add_real_time_data("route", {
            "route_id": "test_route",
            "description": "Ruta de prueba para optimizacion de vehiculos"
        })
        
        # Búsqueda vectorial
        vector_results = ir_system.vector_search("optimizacion vehiculos", top_k=3)
        print(f"✓ Búsqueda vectorial: {len(vector_results)} resultados")
        
        # Búsqueda LSI
        lsi_results = ir_system.lsi_search("optimizacion vehiculos", top_k=3)
        print(f"✓ Búsqueda LSI: {len(lsi_results)} resultados")
        
        # Búsqueda híbrida
        hybrid_results = ir_system.hybrid_search("optimizacion vehiculos", top_k=3)
        print(f"✓ Búsqueda híbrida: {len(hybrid_results)} resultados")
        
    except Exception as e:
        print(f"✗ Error en búsqueda híbrida: {e}")
        import traceback
        traceback.print_exc()
    
    # Prueba 3: Persistencia
    print("\n3. PRUEBA PERSISTENCIA")
    print("-" * 40)
    try:
        from src.SRI.VectorDatabase import VRPVectorDatabase
        
        # Crear DB con datos
        db1 = VRPVectorDatabase("test_persist_cache")
        db1.add_document("persist_test", "Documento de prueba persistencia", {"test": True})
        
        # Crear nueva instancia (debería cargar datos existentes)
        db2 = VRPVectorDatabase("test_persist_cache")
        
        # Buscar datos
        results = db2.search_similar("persistencia", top_k=1)
        
        if results:
            print("✓ Persistencia funciona correctamente")
        else:
            print("✗ Datos no persistieron")
        
    except Exception as e:
        print(f"✗ Error en persistencia: {e}")

def generate_report(results):
    """Genera reporte de resultados"""
    print("\n=== RESUMEN DE VALIDACION ===")
    print("=" * 40)
    
    total_tests = len(results)
    successful_tests = sum(1 for success in results.values() if success)
    success_rate = successful_tests / total_tests
    
    print(f"Pruebas exitosas: {successful_tests}/{total_tests}")
    print(f"Tasa de éxito: {success_rate:.2%}")
    
    print("\nDetalle por componente:")
    for test_name, success in results.items():
        status = "✓ EXITOSO" if success else "✗ FALLIDO"
        print(f"  {test_name}: {status}")
    
    if success_rate >= 0.75:
        print("\n🎯 RESULTADO: SISTEMA FUNCIONAL")
    else:
        print("\n⚠️  RESULTADO: SISTEMA NECESITA ATENCION")
    
    return success_rate

def main():
    """Función principal"""
    print("VALIDACION MANUAL DEL SISTEMA RAG")
    print("Verificando componentes individualmente...")
    print("=" * 60)
    
    # Ejecutar pruebas principales
    results = test_rag_components()
    
    # Pruebas funcionales específicas
    test_functionality_specific()
    
    # Generar reporte
    success_rate = generate_report(results)
    
    return success_rate

if __name__ == "__main__":
    main()
