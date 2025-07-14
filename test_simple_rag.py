"""
Script de prueba simplificado para verificar componentes básicos
"""

import sys
from pathlib import Path

# Añadir ruta del proyecto
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def test_basic_imports():
    """Prueba imports básicos"""
    print("🔍 Probando imports básicos...")
    
    try:
        # Test ChromaDB
        import chromadb
        print("✅ ChromaDB importado correctamente")
        
        # Test Sentence Transformers
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers importado correctamente")
        
        # Test nuestros módulos
        from src.SRI.VectorDatabase import VRPVectorDatabase
        print("✅ VRPVectorDatabase importado correctamente")
        
        from src.NLP.RAG import VRPKnowledgeRAG
        print("✅ VRPKnowledgeRAG importado correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en imports: {e}")
        return False

def test_simple_vector_db():
    """Prueba básica de base de datos vectorial"""
    print("\n🔍 Probando base de datos vectorial básica...")
    
    try:
        from src.SRI.VectorDatabase import VRPVectorDatabase
        
        # Crear instancia con directorio temporal
        vdb = VRPVectorDatabase("simple_test_cache")
        
        # Datos simples de prueba
        simple_data = {
            "location": "La Habana",
            "impact_factor": 1.2,
            "weather_summary": {"temperature_2m": 25}
        }
        
        # Añadir datos
        doc_id = vdb.add_weather_data(simple_data)
        print(f"✅ Documento añadido con ID: {doc_id}")
        
        # Búsqueda simple
        results = vdb.search("clima La Habana", top_k=3)
        print(f"✅ Búsqueda completada: {len(results)} resultados")
        
        # Estadísticas
        stats = vdb.get_collection_stats()
        print(f"✅ Estadísticas obtenidas: {len(stats)} colecciones")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en VectorDB: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_basic():
    """Prueba básica del sistema RAG"""
    print("\n🔍 Probando sistema RAG básico...")
    
    try:
        from src.NLP.RAG import VRPKnowledgeRAG
        
        # Crear instancia RAG
        rag = VRPKnowledgeRAG()
        print("✅ Sistema RAG inicializado")
        
        # Datos simples
        weather_data = {
            "impact_factor": 1.1,
            "interpretation": "Condiciones normales"
        }
        
        # Actualizar conocimiento
        rag.update_knowledge_base("weather", weather_data)
        print("✅ Base de conocimiento actualizada")
        
        # Estado del sistema
        status = rag.get_system_status()
        print(f"✅ Estado del sistema obtenido")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal simplificada"""
    print("🚀 Pruebas básicas del sistema VRP-RAG")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Imports
    if test_basic_imports():
        success_count += 1
    
    # Test 2: Vector DB
    if test_simple_vector_db():
        success_count += 1
    
    # Test 3: RAG
    if test_rag_basic():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Resultados: {success_count}/{total_tests} pruebas exitosas")
    
    if success_count == total_tests:
        print("✅ ¡Todos los componentes funcionan correctamente!")
        print("\n🎯 Sistema listo para:")
        print("  • Base de datos vectorial")
        print("  • Sistema RAG avanzado")
        print("  • Integración con frontend")
    else:
        print("⚠️ Algunos componentes necesitan revisión")

if __name__ == "__main__":
    main()
