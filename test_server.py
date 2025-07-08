#!/usr/bin/env python3
"""
Test script para verificar que el servidor puede iniciarse correctamente
"""

import sys
import os

def test_imports():
    """Prueba las importaciones críticas"""
    print("🔧 Probando importaciones...")
    
    try:
        import asyncio
        print("✅ asyncio")
    except ImportError as e:
        print(f"❌ asyncio: {e}")
        return False
    
    try:
        import websockets
        print("✅ websockets")
    except ImportError as e:
        print(f"❌ websockets: {e}")
        return False
    
    try:
        import networkx
        print("✅ networkx")
    except ImportError as e:
        print(f"❌ networkx: {e}")
        return False
    
    try:
        import numpy
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import flask
        print("✅ flask")
    except ImportError as e:
        print(f"❌ flask: {e}")
        return False
    
    return True

def test_project_imports():
    """Prueba las importaciones del proyecto"""
    print("\n🔧 Probando importaciones del proyecto...")
    
    try:
        from simulation_config import get_config
        print("✅ simulation_config")
    except ImportError as e:
        print(f"❌ simulation_config: {e}")
        return False
    
    try:
        from src.multiagent.environment import Environment
        print("✅ src.multiagent.environment")
    except ImportError as e:
        print(f"❌ src.multiagent.environment: {e}")
        return False
    
    return True

def main():
    print("🚀 Iniciando pruebas del servidor...")
    
    # Cambiar al directorio del proyecto
    project_dir = r"e:\University\3ro\IA-Sim-Sri\IA-Simulation-Information_Retrieval_System-Proyect"
    os.chdir(project_dir)
    print(f"📂 Directorio de trabajo: {os.getcwd()}")
    
    # Probar importaciones básicas
    if not test_imports():
        print("❌ Faltan dependencias básicas")
        return False
    
    # Probar importaciones del proyecto
    if not test_project_imports():
        print("❌ Problema con módulos del proyecto")
        return False
    
    print("\n✅ Todas las pruebas pasaron correctamente")
    print("🎯 El servidor debería poder iniciarse")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
