"""
Utilidades de Testing y Debugging para el Sistema de Semáforos
Herramientas para probar y depurar el sistema modular
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from src.multiagent.traffic_lights import (
    TrafficLightAgent,
    TrafficLightController, 
    server_traffic_manager,
    get_server_traffic_lights_data,
    modify_server_traffic_light,
    get_server_traffic_metrics
)


class TrafficLightTester:
    """Herramientas para testing del sistema de semáforos"""
    
    def __init__(self):
        self.logger = logging.getLogger("TrafficLightTester")
        self.test_results: List[Dict[str, Any]] = []
        
    async def run_basic_functionality_test(self) -> bool:
        """Prueba funcionalidad básica del sistema"""
        print("🧪 Ejecutando prueba de funcionalidad básica...")
        
        try:
            # Test 1: Verificar que el manager esté inicializado
            if not server_traffic_manager.is_ready():
                print("❌ Test 1 FALLIDO: Manager no está listo")
                return False
            print("✅ Test 1 PASADO: Manager inicializado")
            
            # Test 2: Obtener datos de semáforos
            traffic_data = get_server_traffic_lights_data()
            if not traffic_data:
                print("❌ Test 2 FALLIDO: No hay datos de semáforos")
                return False
            print(f"✅ Test 2 PASADO: {len(traffic_data)} semáforos detectados")
            
            # Test 3: Obtener métricas
            metrics = get_server_traffic_metrics()
            if not metrics:
                print("❌ Test 3 FALLIDO: No se pudieron obtener métricas")
                return False
            print("✅ Test 3 PASADO: Métricas obtenidas")
            
            # Test 4: Modificar estado de semáforo
            if traffic_data:
                test_node = traffic_data[0]['node_id']
                success = await modify_server_traffic_light(test_node, 'yellow', 10.0)
                if not success:
                    print(f"❌ Test 4 FALLIDO: No se pudo modificar semáforo {test_node}")
                    return False
                print(f"✅ Test 4 PASADO: Semáforo {test_node} modificado")
            
            print("🎉 Todas las pruebas básicas PASARON")
            return True
            
        except Exception as e:
            print(f"❌ ERROR en pruebas básicas: {e}")
            return False
    
    async def run_performance_test(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Prueba de rendimiento del sistema"""
        print(f"⚡ Ejecutando prueba de rendimiento ({duration_seconds}s)...")
        
        start_time = time.time()
        operations = 0
        errors = 0
        
        # Obtener lista de semáforos para testing
        traffic_data = get_server_traffic_lights_data()
        if not traffic_data:
            return {"error": "No hay semáforos para probar"}
        
        test_nodes = [tl['node_id'] for tl in traffic_data]
        states = ['green', 'yellow', 'red']
        
        print(f"   Probando con {len(test_nodes)} semáforos...")
        
        while time.time() - start_time < duration_seconds:
            try:
                # Operación aleatoria
                import random
                node_id = random.choice(test_nodes)
                state = random.choice(states)
                
                success = await modify_server_traffic_light(node_id, state, 5.0)
                operations += 1
                
                if not success:
                    errors += 1
                
                # Pausa pequeña para no saturar
                await asyncio.sleep(0.1)
                
            except Exception as e:
                errors += 1
                print(f"   Error en operación: {e}")
        
        # Calcular métricas de rendimiento
        elapsed = time.time() - start_time
        ops_per_second = operations / elapsed
        error_rate = errors / operations if operations > 0 else 0
        
        results = {
            "duration": elapsed,
            "total_operations": operations,
            "errors": errors,
            "ops_per_second": ops_per_second,
            "error_rate": error_rate,
            "success_rate": 1 - error_rate
        }
        
        print(f"📊 Resultados de rendimiento:")
        print(f"   Operaciones/segundo: {ops_per_second:.2f}")
        print(f"   Tasa de éxito: {results['success_rate']:.1%}")
        print(f"   Total operaciones: {operations}")
        print(f"   Errores: {errors}")
        
        return results
    
    def validate_traffic_data_format(self, traffic_data: List[Dict]) -> bool:
        """Valida el formato de datos de semáforos"""
        print("🔍 Validando formato de datos...")
        
        required_fields = ['node_id', 'lat', 'lon', 'state']
        
        for i, traffic_light in enumerate(traffic_data):
            # Verificar campos requeridos
            for field in required_fields:
                if field not in traffic_light:
                    print(f"❌ Semáforo {i}: Falta campo '{field}'")
                    return False
            
            # Verificar tipos de datos
            if not isinstance(traffic_light['node_id'], int):
                print(f"❌ Semáforo {i}: node_id debe ser entero")
                return False
            
            if not isinstance(traffic_light['lat'], (int, float)):
                print(f"❌ Semáforo {i}: lat debe ser numérico")
                return False
            
            if not isinstance(traffic_light['lon'], (int, float)):
                print(f"❌ Semáforo {i}: lon debe ser numérico")
                return False
            
            if traffic_light['state'] not in ['green', 'yellow', 'red']:
                print(f"❌ Semáforo {i}: estado inválido '{traffic_light['state']}'")
                return False
        
        print(f"✅ Formato válido para {len(traffic_data)} semáforos")
        return True
    
    async def run_stress_test(self, concurrent_operations: int = 10) -> Dict[str, Any]:
        """Prueba de estrés con operaciones concurrentes"""
        print(f"💪 Ejecutando prueba de estrés ({concurrent_operations} operaciones concurrentes)...")
        
        traffic_data = get_server_traffic_lights_data()
        if not traffic_data:
            return {"error": "No hay semáforos para probar"}
        
        async def stress_operation(op_id: int):
            """Operación individual de estrés"""
            try:
                import random
                node_id = random.choice([tl['node_id'] for tl in traffic_data])
                state = random.choice(['green', 'yellow', 'red'])
                
                success = await modify_server_traffic_light(node_id, state, 2.0)
                return {"op_id": op_id, "success": success, "error": None}
            except Exception as e:
                return {"op_id": op_id, "success": False, "error": str(e)}
        
        # Ejecutar operaciones concurrentes
        start_time = time.time()
        tasks = [stress_operation(i) for i in range(concurrent_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        
        # Analizar resultados
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed = len(results) - successful
        
        stress_results = {
            "concurrent_operations": concurrent_operations,
            "duration": elapsed,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(results),
            "operations_per_second": len(results) / elapsed
        }
        
        print(f"📊 Resultados de estrés:")
        print(f"   Operaciones exitosas: {successful}/{concurrent_operations}")
        print(f"   Tasa de éxito: {stress_results['success_rate']:.1%}")
        print(f"   Tiempo total: {elapsed:.2f}s")
        
        return stress_results


class TrafficLightDebugger:
    """Herramientas de debugging para semáforos"""
    
    def __init__(self):
        self.logger = logging.getLogger("TrafficLightDebugger")
    
    def print_system_status(self):
        """Imprime estado detallado del sistema"""
        print("🔍 ESTADO DEL SISTEMA DE SEMÁFOROS")
        print("=" * 50)
        
        # Estado del manager
        print(f"Manager listo: {'✅ Sí' if server_traffic_manager.is_ready() else '❌ No'}")
        
        if server_traffic_manager.is_ready():
            # Datos de semáforos
            traffic_data = get_server_traffic_lights_data()
            print(f"Semáforos activos: {len(traffic_data)}")
            
            # Distribución de estados
            state_count = {}
            for tl in traffic_data:
                state = tl['state']
                state_count[state] = state_count.get(state, 0) + 1
            
            print("Distribución de estados:")
            for state, count in state_count.items():
                print(f"   {state}: {count}")
            
            # Métricas
            metrics = get_server_traffic_metrics()
            if metrics:
                print("\nMétricas del sistema:")
                controller_status = metrics.get('controller_status', {})
                performance = metrics.get('performance', {})
                
                print(f"   Optimización: {'✅' if controller_status.get('optimization_enabled') else '❌'}")
                print(f"   Eficiencia promedio: {performance.get('average_efficiency', 0):.3f}")
                print(f"   Eficiencia de red: {performance.get('network_efficiency', 0):.3f}")
        
        print("=" * 50)
    
    def export_traffic_state(self, filename: str = None) -> str:
        """Exporta el estado actual a JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traffic_state_{timestamp}.json"
        
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "manager_ready": server_traffic_manager.is_ready(),
            "traffic_lights": get_server_traffic_lights_data(),
            "metrics": get_server_traffic_metrics()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Estado exportado a: {filename}")
        return filename
    
    async def monitor_system(self, duration_seconds: int = 60, interval: int = 5):
        """Monitorea el sistema durante un período"""
        print(f"👁️ Monitoreando sistema por {duration_seconds}s (cada {interval}s)...")
        
        start_time = time.time()
        monitoring_data = []
        
        while time.time() - start_time < duration_seconds:
            timestamp = datetime.now().isoformat()
            
            # Recopilar datos actuales
            traffic_data = get_server_traffic_lights_data()
            metrics = get_server_traffic_metrics()
            
            monitoring_point = {
                "timestamp": timestamp,
                "traffic_lights_count": len(traffic_data),
                "metrics": metrics
            }
            
            monitoring_data.append(monitoring_point)
            
            # Mostrar progreso
            elapsed = time.time() - start_time
            print(f"   [{elapsed:.0f}s] Semáforos: {len(traffic_data)}, "
                  f"Eficiencia: {metrics.get('performance', {}).get('average_efficiency', 0):.3f}")
            
            await asyncio.sleep(interval)
        
        # Guardar datos de monitoreo
        monitor_filename = f"traffic_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(monitor_filename, 'w', encoding='utf-8') as f:
            json.dump(monitoring_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Datos de monitoreo guardados en: {monitor_filename}")
        return monitoring_data


# Funciones de utilidad rápida

async def quick_test():
    """Prueba rápida del sistema"""
    tester = TrafficLightTester()
    return await tester.run_basic_functionality_test()


def quick_debug():
    """Debug rápido del sistema"""
    debugger = TrafficLightDebugger()
    debugger.print_system_status()


async def quick_performance_test():
    """Prueba rápida de rendimiento"""
    tester = TrafficLightTester()
    return await tester.run_performance_test(duration_seconds=10)


if __name__ == "__main__":
    print("🛠️ Utilidades de Testing del Sistema de Semáforos")
    print("Importa este módulo y usa las funciones quick_test(), quick_debug(), etc.")
