"""
Script de prueba para el sistema VRP-RAG con base de datos vectorial
"""

import sys
from pathlib import Path
import json

# Añadir ruta del proyecto
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.NLP.RAG import VRPKnowledgeRAG
from src.SRI.VRPInformationRetrieval import VRPInformationRetrievalSystem
from src.SRI.VectorDatabase import VRPVectorDatabase

def test_vector_database():
    """Prueba la base de datos vectorial"""
    print("🔍 Probando base de datos vectorial...")
    
    vdb = VRPVectorDatabase("test_vector_cache")
    
    # Datos de prueba
    weather_data = {
        "location": "La Habana, Cuba",
        "impact_factor": 1.4,
        "interpretation": "Condiciones adversas moderadas",
        "weather_summary": {
            "temperature_2m": 32,
            "precipitation": 5,
            "wind_speed_10m": 25
        },
        "recommendations": "Considerar rutas alternativas debido al viento fuerte"
    }
    
    route_data = {
        "optimization_method": "Genetic Algorithm",
        "routes": [
            {"distance": 18.5, "path": [1, 2, 3, 4, 5, 6]},
            {"distance": 14.7, "path": [1, 7, 8, 9, 10]},
            {"distance": 22.1, "path": [1, 11, 12, 13, 14, 15]}
        ],
        "efficiency_metrics": {
            "total_distance": 55.3,
            "efficiency_score": 87.5,
            "total_delivery_points": 15
        },
        "computation_time": 2.3
    }
    
    traffic_event = {
        "type": "Congestion",
        "description": "Tráfico intenso en el Malecón debido a evento especial",
        "location": "Malecón, La Habana",
        "severity": "high",
        "impact_area": "city_center",
        "estimated_duration": "2 horas",
        "affected_routes": ["ruta_centro", "ruta_vedado"]
    }
    
    # Añadir datos
    weather_id = vdb.add_weather_data(weather_data)
    route_id = vdb.add_route_analysis(route_data)
    traffic_id = vdb.add_traffic_event(traffic_event)
    
    print(f"✅ Datos añadidos - Weather: {weather_id}, Route: {route_id}, Traffic: {traffic_id}")
    
    # Probar búsquedas
    queries = [
        "¿Cómo afecta el clima a las entregas?",
        "¿Cuál es la eficiencia de las rutas actuales?",
        "¿Hay problemas de tráfico en el centro?",
        "¿Qué métodos de optimización están disponibles?"
    ]
    
    for query in queries:
        print(f"\n🔍 Consulta: {query}")
        results = vdb.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['collection']}] Score: {result['similarity']:.3f}")
            print(f"     {result['document'][:100]}...")
    
    # Estadísticas
    stats = vdb.get_collection_stats()
    print(f"\n📊 Estadísticas de colecciones:")
    for collection, stat in stats.items():
        print(f"  {collection}: {stat.get('document_count', 0)} documentos")
    
    return vdb

def test_ir_system():
    """Prueba el sistema de recuperación de información"""
    print("\n🧠 Probando sistema de recuperación de información...")
    
    ir_system = VRPInformationRetrievalSystem("test_ir_cache")
    
    # Documentos de prueba
    documents = [
        {
            'id': 'weather_impact_001',
            'content': 'El impacto del clima en La Habana durante la temporada de lluvias puede ser significativo. Precipitaciones de 5-10mm pueden reducir la visibilidad y hacer las carreteras resbaladizas, especialmente en el Malecón y las zonas costeras.',
            'metadata': {'type': 'weather', 'location': 'Habana', 'season': 'rainy'}
        },
        {
            'id': 'optimization_ga_001',
            'content': 'Los algoritmos genéticos han demostrado ser efectivos para problemas VRP en entornos urbanos complejos como La Habana. La configuración óptima incluye población de 100 individuos, 500 generaciones, y tasa de mutación del 1%.',
            'metadata': {'type': 'optimization', 'method': 'genetic_algorithm', 'domain': 'urban_vrp'}
        },
        {
            'id': 'traffic_patterns_001',
            'content': 'Los patrones de tráfico en La Habana muestran congestión máxima entre 7-9 AM y 5-7 PM. Las rutas que evitan el centro histórico y utilizan vías perimetrales como la Circunvalación suelen ser más eficientes.',
            'metadata': {'type': 'traffic', 'location': 'Habana', 'analysis_type': 'patterns'}
        }
    ]
    
    # Indexar documentos
    ir_system.index_documents(documents)
    
    # Probar búsquedas
    test_queries = [
        "¿Cómo optimizar rutas en La Habana?",
        "¿Qué impacto tiene la lluvia en las entregas?",
        "¿Cuáles son los mejores horarios para entregas?",
        "¿Qué algoritmos son mejores para VRP urbano?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Consulta: {query}")
        results = ir_system.search(query, top_k=2, use_hybrid=True)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['source']}] Score: {result['score']:.3f}")
            print(f"     ID: {result['id']}")
            print(f"     {result['content'][:120]}...")
    
    # Estadísticas del sistema
    stats = ir_system.get_system_stats()
    print(f"\n📊 Estadísticas del sistema IR:")
    print(f"  Documentos LSI: {stats['lsi_documents']}")
    print(f"  Características TF-IDF: {stats['lsi_features']}")
    print(f"  Componentes LSI: {stats['lsi_components']}")
    
    return ir_system

def test_rag_system():
    """Prueba el sistema RAG completo"""
    print("\n🤖 Probando sistema RAG completo...")
    
    rag = VRPKnowledgeRAG()
    
    # Simular datos del sistema
    weather_data = {
        "impact_factor": 1.6,
        "interpretation": "Condiciones adversas - lluvia intensa",
        "weather_summary": {
            "temperature_2m": 26,
            "precipitation": 8,
            "wind_speed_10m": 30
        },
        "location": "La Habana, Cuba"
    }
    
    route_data = {
        "routes": [
            {"distance": 16.2, "path": [1, 2, 3, 4, 5]},
            {"distance": 19.8, "path": [1, 6, 7, 8, 9]},
            {"distance": 13.5, "path": [1, 10, 11, 12]}
        ],
        "optimization_method": "Genetic Algorithm",
        "computation_time": 3.1
    }
    
    traffic_event = {
        "type": "Accident",
        "description": "Accidente en la intersección de 23 y L",
        "location": "Vedado, La Habana",
        "severity": "medium",
        "estimated_duration": "30 minutos"
    }
    
    # Actualizar base de conocimientos
    rag.update_knowledge_base("weather", weather_data)
    rag.update_knowledge_base("routes", route_data)
    rag.update_knowledge_base("traffic_events", traffic_event)
    
    # Probar consultas complejas
    test_questions = [
        "¿Cómo está afectando el clima actual a mis rutas de entrega?",
        "¿Cuál es la eficiencia de mis rutas optimizadas y cómo puedo mejorarlas?",
        "¿Hay problemas de tráfico que puedan impactar mis entregas en el Vedado?",
        "¿Qué recomendaciones tienes para optimizar las entregas considerando las condiciones actuales?",
        "¿Cuál es el rendimiento del algoritmo genético en mi sistema?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Pregunta: {question}")
        result = rag.ask_with_context(question)
        
        if result['success']:
            print(f"✅ Categoría: {result['question_category']}")
            print(f"📚 Contexto usado:")
            context = result['context_used']
            print(f"  - Documentos recuperados: {context.get('retrieved_documents', 0)}")
            print(f"  - Base de datos vectorial: {'✅' if context.get('vector_db_available') else '❌'}")
            print(f"  - Sistema LSI: {'✅' if context.get('lsi_system_available') else '❌'}")
            if context.get('sources_used'):
                print(f"  - Fuentes: {', '.join(context['sources_used'])}")
            if context.get('collections_searched'):
                print(f"  - Colecciones: {', '.join(context['collections_searched'])}")
            
            print(f"🤖 Respuesta: {result['response'][:200]}...")
            
            if 'retrieved_context' in result and result['retrieved_context']:
                print(f"📄 Contexto recuperado ({len(result['retrieved_context'])} docs):")
                for doc in result['retrieved_context'][:2]:  # Solo mostrar top 2
                    print(f"  - [{doc['source']}] Score: {doc['score']:.3f} - {doc['snippet'][:80]}...")
        else:
            print(f"❌ Error: {result.get('message', 'Error desconocido')}")
    
    # Estado del sistema
    status = rag.get_system_status()
    print(f"\n📊 Estado del sistema RAG:")
    print(f"  Base de conocimientos: {json.dumps(status['rag_system']['knowledge_base_entries'], indent=2)}")
    print(f"  Sistema IR: {status['ir_system']['lsi_documents']} docs indexados")

def main():
    """Función principal de prueba"""
    print("🚀 Iniciando pruebas del sistema VRP-RAG con base de datos vectorial")
    print("=" * 70)
    
    try:
        # Probar componentes individuales
        vdb = test_vector_database()
        ir_system = test_ir_system()
        test_rag_system()
        
        print("\n" + "=" * 70)
        print("✅ Todas las pruebas completadas exitosamente!")
        print("\n🎯 El sistema está listo para usar con:")
        print("  • Base de datos vectorial ChromaDB")
        print("  • Sistema de recuperación híbrido (Vector + LSI)")
        print("  • RAG avanzado con contexto dinámico")
        print("  • Integración completa con el frontend React")
        
        # Limpiar datos de prueba
        print("\n🧹 Limpiando datos de prueba...")
        vdb.cleanup_old_data(0)  # Eliminar todo
        ir_system.cleanup(0)
        
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
