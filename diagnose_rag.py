#!/usr/bin/env python3
"""
Diagnosis del Sistema RAG
"""

import sys
import traceback
from pathlib import Path

# Configurar paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def diagnose_rag_system():
    """Diagnostica el sistema RAG paso a paso"""
    print("=== DIAGNOSTICO DEL SISTEMA RAG ===")
    
    # Paso 1: Importaciones
    print("\n1. Verificando importaciones...")
    try:
        from src.NLP.RAG import create_vrp_rag_assistant
        print("✓ Importacion de RAG exitosa")
    except Exception as e:
        print(f"✗ Error en importacion RAG: {e}")
        traceback.print_exc()
        return False
    
    # Paso 2: Creacion del asistente
    print("\n2. Creando asistente RAG...")
    try:
        rag_assistant = create_vrp_rag_assistant()
        print("✓ Asistente RAG creado")
    except Exception as e:
        print(f"✗ Error creando asistente: {e}")
        traceback.print_exc()
        return False
    
    # Paso 3: Verificar componentes
    print("\n3. Verificando componentes...")
    try:
        has_gemini = hasattr(rag_assistant, 'gemini')
        has_ir_system = hasattr(rag_assistant, 'ir_system')
        has_knowledge_base = hasattr(rag_assistant, 'knowledge_base')
        
        print(f"✓ Gemini disponible: {has_gemini}")
        print(f"✓ Sistema IR disponible: {has_ir_system}")
        print(f"✓ Base de conocimientos disponible: {has_knowledge_base}")
        
        if not all([has_gemini, has_ir_system, has_knowledge_base]):
            print("✗ Algunos componentes no están disponibles")
            return False
            
    except Exception as e:
        print(f"✗ Error verificando componentes: {e}")
        traceback.print_exc()
        return False
    
    # Paso 4: Consulta simple
    print("\n4. Probando consulta simple...")
    try:
        question = "Hola"
        print(f"Pregunta: {question}")
        
        # Usar timeout para evitar cuelgues
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Timeout en generate_response")
        
        # Configurar timeout solo en sistemas Unix
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 segundos timeout
        
        response = rag_assistant.generate_response(question)
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancelar timeout
        
        print(f"✓ Respuesta obtenida: {len(str(response))} caracteres")
        print(f"Respuesta: {str(response)[:100]}...")
        
        return True
        
    except TimeoutError:
        print("✗ Timeout en generate_response")
        return False
    except Exception as e:
        print(f"✗ Error en consulta: {e}")
        traceback.print_exc()
        return False

def test_individual_components():
    """Prueba componentes individualmente"""
    print("\n=== PRUEBA DE COMPONENTES INDIVIDUALES ===")
    
    # Prueba Gemini
    print("\n1. Probando Gemini...")
    try:
        from src.NLP.Gemini import Gemini
        gemini = Gemini()
        print("✓ Gemini inicializado")
        
        # Prueba simple
        # response = gemini.generate_response("Hola")
        # print(f"✓ Respuesta Gemini: {len(str(response))} caracteres")
        
    except Exception as e:
        print(f"✗ Error con Gemini: {e}")
    
    # Prueba Sistema IR
    print("\n2. Probando Sistema IR...")
    try:
        from src.SRI.VRPInformationRetrieval import VRPInformationRetrievalSystem
        ir_system = VRPInformationRetrievalSystem("vector_cache")
        print("✓ Sistema IR inicializado")
        
        # Prueba busqueda simple
        results = ir_system.search("optimizacion", top_k=3)
        print(f"✓ Búsqueda IR exitosa: {len(results)} resultados")
        
    except Exception as e:
        print(f"✗ Error con Sistema IR: {e}")
        traceback.print_exc()

def main():
    """Funcion principal"""
    print("DIAGNOSTICO COMPREHENSIVO DEL SISTEMA RAG")
    print("=" * 50)
    
    # Diagnostico principal
    success = diagnose_rag_system()
    
    # Prueba componentes individuales
    test_individual_components()
    
    print(f"\n=== RESULTADO FINAL ===")
    print(f"Estado del sistema: {'FUNCIONAL' if success else 'CON PROBLEMAS'}")
    
    return success

if __name__ == "__main__":
    main()
