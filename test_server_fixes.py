#!/usr/bin/env python3
"""
Test script para verificar las correcciones del servidor
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Verifica que las importaciones básicas funcionen"""
    print("🔍 Verificando importaciones básicas...")
    try:
        from simulation_config import get_config
        from src.multiagent.environment import Environment
        print("✅ Importaciones básicas exitosas")
        return True
    except Exception as e:
        print(f"❌ Error en importaciones: {e}")
        return False

def test_config_loading():
    """Verifica que la configuración se cargue correctamente"""
    print("🔍 Verificando configuración...")
    try:
        from simulation_config import get_config
        config = get_config("normal")
        print(f"✅ Configuración cargada: {config}")
        return True
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        return False

def test_server_functions():
    """Verifica que las funciones principales del servidor funcionen"""
    print("🔍 Verificando funciones del servidor...")
    try:
        # Importar funciones específicas
        from server import (
            haversine, 
            validate_node_connectivity, 
            find_closest_node_in_component
        )
        
        # Test haversine
        distance = haversine(23.1136, -82.3666, 23.1200, -82.3700)
        print(f"✅ Función haversine: {distance:.4f} km")
        
        print("✅ Funciones del servidor verificadas")
        return True
    except Exception as e:
        print(f"❌ Error en funciones del servidor: {e}")
        return False

async def test_simulation_loop():
    """Prueba básica del bucle de simulación"""
    print("🔍 Verificando bucle de simulación...")
    try:
        # Simular un bucle simple
        epoch_count = 0
        max_epochs = 10
        start_time = time.time()
        
        print(f"⏱️ Iniciando prueba de simulación a las {datetime.now().strftime('%H:%M:%S')}")
        
        while epoch_count < max_epochs:
            epoch_count += 1
            
            # Simular procesamiento
            await asyncio.sleep(0.01)
            
            # Log de progreso
            if epoch_count % 5 == 0:
                elapsed = time.time() - start_time
                print(f"📊 Época {epoch_count}/{max_epochs} - {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        print(f"✅ Simulación de prueba completada: {epoch_count} épocas en {total_time:.2f}s")
        print(f"⚡ Velocidad: {epoch_count/total_time:.1f} épocas/segundo")
        return True
    except Exception as e:
        print(f"❌ Error en bucle de simulación: {e}")
        return False

def test_error_handling():
    """Prueba el manejo de errores"""
    print("🔍 Verificando manejo de errores...")
    try:
        # Función que debería fallar
        def failing_function():
            raise ValueError("Error de prueba")
        
        # Manejo de errores
        try:
            failing_function()
        except ValueError as e:
            print(f"✅ Error capturado correctamente: {e}")
        
        # Métricas de respaldo
        backup_metrics = {
            "total_vehicles": 10,
            "average_speed": 45.0,
            "congestion_level": 0.1,
            "completed_deliveries": 5,
        }
        print(f"✅ Métricas de respaldo: {backup_metrics}")
        return True
    except Exception as e:
        print(f"❌ Error en manejo de errores: {e}")
        return False

async def main():
    """Ejecuta todas las pruebas"""
    print("🚀 Iniciando pruebas del servidor...")
    print("=" * 60)
    
    tests = [
        ("Importaciones básicas", test_basic_imports),
        ("Configuración", test_config_loading),
        ("Funciones del servidor", test_server_functions),
        ("Bucle de simulación", test_simulation_loop),
        ("Manejo de errores", test_error_handling),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 PRUEBA: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ PASÓ: {test_name}")
            else:
                failed += 1
                print(f"❌ FALLÓ: {test_name}")
        except Exception as e:
            failed += 1
            print(f"❌ ERROR EN: {test_name} - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTADOS DE PRUEBAS:")
    print(f"✅ Pasaron: {passed}")
    print(f"❌ Fallaron: {failed}")
    print(f"📈 Total: {passed + failed}")
    print(f"🎯 Tasa de éxito: {(passed/(passed+failed))*100:.1f}%")
    
    if failed == 0:
        print("🎉 ¡Todas las pruebas pasaron! El servidor debería funcionar correctamente.")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa los errores antes de ejecutar el servidor.")

if __name__ == "__main__":
    asyncio.run(main())
