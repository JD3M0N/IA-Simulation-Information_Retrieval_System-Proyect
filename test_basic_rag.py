#!/usr/bin/env python3
"""
Validacion Basica del Sistema RAG
"""

import sys
from pathlib import Path

# Configurar paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def test_basic_functionality():
    """Prueba basica de funcionalidad"""
    print("Iniciando prueba basica...")
    
    try:
        from src.NLP.RAG import create_vrp_rag_assistant
        print("✓ Importacion exitosa")
        
        # Crear asistente
        rag_assistant = create_vrp_rag_assistant()
        print("✓ Asistente RAG creado")
        
        # Prueba basica
        question = "Como optimizar rutas?"
        print(f"Pregunta: {question}")
        
        response = rag_assistant.generate_response(question)
        print(f"✓ Respuesta generada: {len(str(response))} caracteres")
        
        # Mostrar respuesta
        if response:
            print(f"Respuesta: {str(response)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    print(f"\nResultado: {'EXITOSO' if success else 'FALLO'}")
