"""
Script para probar el endpoint RAG directamente
"""

import requests
import json

def test_rag_endpoint():
    """Prueba el endpoint /ask_rag"""
    url = "http://localhost:8767/ask_rag"
    
    # Casos de prueba
    test_cases = [
        {
            "name": "Pregunta simple sin contexto",
            "data": {
                "question": "¿Qué es VRP?"
            }
        },
        {
            "name": "Pregunta con contexto de clima",
            "data": {
                "question": "¿Cómo afecta el clima a mis rutas?",
                "context_data": {
                    "weather": {
                        "impact_factor": 1.3,
                        "interpretation": "Condiciones moderadas"
                    }
                }
            }
        },
        {
            "name": "Pregunta con contexto de rutas",
            "data": {
                "question": "¿Cuál es la eficiencia de mis rutas?",
                "context_data": {
                    "routes": [
                        {"distance": 15.5, "path": [1, 2, 3, 4, 5]},
                        {"distance": 12.3, "path": [1, 6, 7, 8]}
                    ]
                }
            }
        },
        {
            "name": "Pregunta con contexto completo",
            "data": {
                "question": "Dame un análisis completo del sistema",
                "context_data": {
                    "weather": {
                        "impact_factor": 1.2,
                        "interpretation": "Condiciones normales",
                        "weather_summary": {
                            "temperature_2m": 28,
                            "precipitation": 0,
                            "wind_speed_10m": 10
                        }
                    },
                    "routes": [
                        {"distance": 18.2, "path": [1, 2, 3, 4, 5, 6]},
                        {"distance": 14.7, "path": [1, 7, 8, 9]},
                        {"distance": 22.1, "path": [1, 10, 11, 12, 13]}
                    ],
                    "system_status": {
                        "status": "operational",
                        "last_optimization": "2025-07-13T19:30:00"
                    }
                }
            }
        }
    ]
    
    print("🚀 Probando endpoint RAG...")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        
        try:
            # Hacer la petición
            response = requests.post(
                url, 
                json=test_case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    print("✅ Éxito!")
                    print(f"Categoría: {result.get('question_category', 'N/A')}")
                    print(f"Respuesta: {result.get('response', 'N/A')[:200]}...")
                    
                    context_used = result.get('context_used', {})
                    print(f"Documentos recuperados: {context_used.get('retrieved_documents', 0)}")
                    print(f"VectorDB usado: {context_used.get('vector_db_available', False)}")
                    print(f"LSI usado: {context_used.get('lsi_system_available', False)}")
                    
                    sources = context_used.get('sources_used', [])
                    if sources:
                        print(f"Fuentes: {', '.join(sources)}")
                    
                    collections = context_used.get('collections_searched', [])
                    if collections:
                        print(f"Colecciones: {', '.join(collections)}")
                else:
                    print("❌ Error en la respuesta:")
                    print(f"Mensaje: {result.get('message', 'N/A')}")
                    print(f"Error: {result.get('error', 'N/A')}")
            else:
                print(f"❌ Error HTTP: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Mensaje: {error_data.get('message', 'N/A')}")
                except:
                    print(f"Respuesta: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            print("❌ Error: No se puede conectar al servidor")
            print("   Asegúrate de que el servidor esté ejecutándose en localhost:8767")
            break
        except requests.exceptions.Timeout:
            print("❌ Error: Timeout - la solicitud tardó demasiado")
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Pruebas completadas")

if __name__ == "__main__":
    test_rag_endpoint()
