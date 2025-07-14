#!/usr/bin/env python3
"""
Script para probar el endpoint RAG del servidor en vivo
"""

import requests
import json

def test_live_rag_endpoint():
    """Prueba el endpoint RAG del servidor principal en funcionamiento"""
    base_url = "http://localhost:8767"
    
    print("🚀 Probando endpoint RAG en servidor en vivo...")
    print("=" * 60)
    
    # Test 1: Pregunta simple
    print("\n1. Pregunta simple sobre VRP")
    print("-" * 40)
    
    payload = {
        "question": "¿Qué es el VRP y cómo funciona en La Habana?"
    }
    
    try:
        response = requests.post(f"{base_url}/ask_rag", json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Éxito!")
            print(f"Categoría: {result.get('question_category', 'N/A')}")
            print(f"Respuesta: {result.get('response', '')[:100]}...")
            print(f"Documentos recuperados: {result.get('context_used', {}).get('retrieved_documents', 0)}")
            print(f"VectorDB usado: {result.get('context_used', {}).get('vector_db_available', False)}")
            print(f"LSI usado: {result.get('context_used', {}).get('lsi_system_available', False)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Respuesta: {response.text}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
    
    # Test 2: Pregunta con contexto de rutas
    print("\n2. Pregunta sobre optimización de rutas")
    print("-" * 40)
    
    payload = {
        "question": "¿Cómo puedo optimizar mis rutas de entrega considerando el tráfico?",
        "context_data": {
            "routes": [
                {
                    "id": "route_1",
                    "distance": 15.5,
                    "duration": 45,
                    "cost": 120.50,
                    "customers": 8
                },
                {
                    "id": "route_2", 
                    "distance": 12.3,
                    "duration": 38,
                    "cost": 95.25,
                    "customers": 6
                }
            ],
            "weather": {
                "current": "parcialmente_nublado",
                "temperature": 28,
                "humidity": 75,
                "wind_speed": 12,
                "precipitation": 0
            }
        }
    }
    
    try:
        response = requests.post(f"{base_url}/ask_rag", json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Éxito!")
            print(f"Categoría: {result.get('question_category', 'N/A')}")
            print(f"Respuesta: {result.get('response', '')[:150]}...")
            print(f"Documentos recuperados: {result.get('context_used', {}).get('retrieved_documents', 0)}")
            print(f"Fuentes: {result.get('context_used', {}).get('sources_used', [])}")
            print(f"Colecciones: {result.get('context_used', {}).get('collections_searched', [])}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Respuesta: {response.text}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
    
    # Test 3: Pregunta con contexto vacío (para verificar robustez)
    print("\n3. Pregunta con contexto vacío")
    print("-" * 40)
    
    payload = {
        "question": "Dame recomendaciones para mejorar la eficiencia logística",
        "context_data": {}
    }
    
    try:
        response = requests.post(f"{base_url}/ask_rag", json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Éxito!")
            print(f"Categoría: {result.get('question_category', 'N/A')}")
            print(f"Respuesta: {result.get('response', '')[:100]}...")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Respuesta: {response.text}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Pruebas completadas")

if __name__ == "__main__":
    test_live_rag_endpoint()
