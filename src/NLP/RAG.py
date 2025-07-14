from dotenv import load_dotenv
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Añadir rutas necesarias
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.NLP.Gemini import Gemini
from src.SRI.VRPInformationRetrieval import VRPInformationRetrievalSystem

load_dotenv()

class VRPKnowledgeRAG:
    """
    Sistema RAG especializado en problemas de ruteo de vehículos (VRP)
    con información contextual del clima, rutas optimizadas y análisis de rendimiento
    """
    
    def __init__(self):
        self.gemini = Gemini()
        
        # Sistema de recuperación de información híbrido
        self.ir_system = VRPInformationRetrievalSystem("vector_cache")
        
        # Base de conocimientos en memoria (para compatibilidad)
        self.knowledge_base = {
            "weather_data": {},
            "route_statistics": {},
            "traffic_events": [],
            "optimization_history": [],
            "system_performance": {},
            "crawler_data": {},
            "markov_insights": {},
            "knowledge_graph_rules": {}
        }
        
        # Contexto especializado para VRP
        self.vrp_context = {
            "location": "La Habana, Cuba",
            "coordinates": {"lat": 23.1136, "lon": -82.3666},
            "optimization_methods": ["Genetic Algorithm", "Tabu Search", "VNS", "Simulated Annealing"],
            "weather_factors": ["precipitation", "wind_speed", "visibility", "temperature"],
            "road_types": ["motorway", "primary", "secondary", "residential", "unpaved"]
        }
        
        # Inicializar con conocimiento base
        self._initialize_base_knowledge()
    
    def _initialize_base_knowledge(self):
        """Inicializa la base de conocimientos con información fundamental sobre VRP"""
        base_documents = [
            {
                'id': 'vrp_overview',
                'content': '''El Problema de Ruteo de Vehículos (VRP) es un problema de optimización combinatoria 
                que busca determinar las rutas óptimas para una flota de vehículos que deben visitar un conjunto 
                de clientes. En La Habana, Cuba, factores como el clima tropical, la infraestructura vial y 
                los patrones de tráfico urbano afectan significativamente la planificación de rutas.''',
                'metadata': {'type': 'knowledge_base', 'category': 'vrp_fundamentals', 'source': 'system'}
            },
            {
                'id': 'havana_logistics',
                'content': '''La logística urbana en La Habana presenta desafíos únicos: calles estrechas en el 
                centro histórico, tráfico intenso en horarios pico, infraestructura vial variable, y condiciones 
                climáticas que incluyen lluvias tropicales intensas que pueden afectar la visibilidad y las 
                condiciones de las carreteras. Las rutas deben considerar el Malecón, el Vedado, Centro Habana 
                y las zonas industriales.''',
                'metadata': {'type': 'knowledge_base', 'category': 'location_context', 'source': 'system'}
            },
            {
                'id': 'optimization_methods',
                'content': '''Los métodos de optimización para VRP incluyen: Algoritmos Genéticos (GA) para 
                exploración amplia del espacio de soluciones, Búsqueda Tabú para evitar óptimos locales, 
                Recocido Simulado para balance entre exploración y explotación, y Búsqueda de Vecindad Variable 
                (VNS) para diversificación. Cada método tiene fortalezas específicas según el tamaño y 
                características del problema.''',
                'metadata': {'type': 'knowledge_base', 'category': 'optimization', 'source': 'system'}
            },
            {
                'id': 'weather_impact',
                'content': '''El clima en La Habana afecta las operaciones de entrega de múltiples formas: 
                precipitación reduce la visibilidad y puede hacer las carreteras resbaladizas, vientos fuertes 
                afectan vehículos grandes, temperatura alta puede afectar productos sensibles, y la humedad 
                puede impactar el rendimiento de los vehículos. Un factor de impacto de 1.0 indica condiciones 
                normales, mientras que 2.0+ indica condiciones adversas significativas.''',
                'metadata': {'type': 'knowledge_base', 'category': 'weather_analysis', 'source': 'system'}
            }
        ]
        
        # Indexar documentos base
        self.ir_system.index_documents(base_documents)
    
    def update_knowledge_base(self, data_source: str, data: Dict[str, Any]):
        """
        Actualiza la base de conocimientos con nueva información
        Ahora integrado con base de datos vectorial
        
        Args:
            data_source: Tipo de datos ("weather", "routes", "traffic", etc.)
            data: Información a agregar
        """
        try:
            # Validar entrada
            if not data_source or not isinstance(data_source, str):
                print("Error: data_source debe ser un string válido")
                return
            
            if not data or not isinstance(data, dict):
                print(f"Error: data debe ser un diccionario válido para {data_source}")
                return
            
            timestamp = datetime.now().isoformat()
            
            # Mantener compatibilidad con sistema anterior
            if data_source == "weather":
                weather_info = {
                    **data,
                    "timestamp": timestamp,
                    "impact_analysis": self._analyze_weather_impact(data)
                }
                self.knowledge_base["weather_data"] = weather_info
                
                # Añadir al sistema vectorial de forma segura
                try:
                    self.ir_system.add_real_time_data("weather", data)
                except Exception as e:
                    print(f"Error añadiendo datos meteorológicos al sistema vectorial: {e}")
            
            elif data_source == "routes":
                route_info = {
                    **data,
                    "timestamp": timestamp,
                    "efficiency_metrics": self._calculate_route_efficiency(data)
                }
                self.knowledge_base["route_statistics"] = route_info
                
                # Añadir al sistema vectorial de forma segura
                try:
                    self.ir_system.add_real_time_data("route", data)
                except Exception as e:
                    print(f"Error añadiendo datos de rutas al sistema vectorial: {e}")
            
            elif data_source == "traffic_events":
                event_info = {
                    **data,
                    "timestamp": timestamp
                }
                self.knowledge_base["traffic_events"].append(event_info)
                # Mantener solo los últimos 50 eventos
                self.knowledge_base["traffic_events"] = self.knowledge_base["traffic_events"][-50:]
                
                # Añadir al sistema vectorial de forma segura
                try:
                    self.ir_system.add_real_time_data("traffic", data)
                except Exception as e:
                    print(f"Error añadiendo eventos de tráfico al sistema vectorial: {e}")
            
            elif data_source == "optimization":
                opt_info = {
                    **data,
                    "timestamp": timestamp
                }
                self.knowledge_base["optimization_history"].append(opt_info)
                # Mantener solo las últimas 20 optimizaciones
                self.knowledge_base["optimization_history"] = self.knowledge_base["optimization_history"][-20:]
                
                # Añadir al sistema vectorial de forma segura
                try:
                    performance_data = {
                        "component": "optimization",
                        "metric_type": "optimization_run",
                        "value": data.get("computation_time", 0),
                        "unit": "seconds",
                        "method": data.get("method", "unknown"),
                        "description": f"Optimización completada con {data.get('method', 'método desconocido')}"
                    }
                    self.ir_system.add_real_time_data("performance", performance_data)
                except Exception as e:
                    print(f"Error añadiendo datos de optimización al sistema vectorial: {e}")
            
            elif data_source == "performance":
                perf_info = {
                    **data,
                    "timestamp": timestamp
                }
                self.knowledge_base["system_performance"] = perf_info
                
                # Añadir al sistema vectorial de forma segura
                try:
                    self.ir_system.add_real_time_data("performance", data)
                except Exception as e:
                    print(f"Error añadiendo datos de rendimiento al sistema vectorial: {e}")
            
            elif data_source == "crawler":
                crawler_info = {
                    **data,
                    "timestamp": timestamp
                }
                self.knowledge_base["crawler_data"] = crawler_info
                
                # Añadir al sistema vectorial si hay datos relevantes
                try:
                    if "relevant_events" in data and isinstance(data["relevant_events"], list):
                        for event in data["relevant_events"]:
                            if isinstance(event, dict):
                                traffic_data = {
                                    "type": "crawler_event",
                                    "description": event.get("description", ""),
                                    "source": "web_crawler",
                                    "relevance": event.get("relevance", 0.5)
                                }
                                self.ir_system.add_real_time_data("traffic", traffic_data)
                except Exception as e:
                    print(f"Error añadiendo datos del crawler al sistema vectorial: {e}")
            
            else:
                print(f"Advertencia: Tipo de datos desconocido: {data_source}")
                
        except Exception as e:
            print(f"Error en update_knowledge_base: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_context_prompt(self, user_question: str) -> str:
        """
        Genera un prompt contextualizado con toda la información relevante
        """
        # Información del sistema actual
        current_weather = self.knowledge_base.get("weather_data", {})
        current_routes = self.knowledge_base.get("route_statistics", {})
        recent_events = self.knowledge_base.get("traffic_events", [])[-5:]  # Últimos 5 eventos
        optimization_stats = self._get_optimization_summary()
        
        context_prompt = f"""
Eres un asistente especializado en sistemas de ruteo de vehículos (VRP) para logística urbana en La Habana, Cuba.

INFORMACIÓN CONTEXTUAL ACTUAL:

🌍 UBICACIÓN Y CONFIGURACIÓN:
- Ciudad: {self.vrp_context['location']}
- Coordenadas: {self.vrp_context['coordinates']}
- Métodos de optimización disponibles: {', '.join(self.vrp_context['optimization_methods'])}

🌤️ INFORMACIÓN CLIMÁTICA ACTUAL:
{self._format_weather_context(current_weather)}

📊 ESTADÍSTICAS DE RUTAS ACTUALES:
{self._format_route_context(current_routes)}

🚦 EVENTOS DE TRÁFICO RECIENTES:
{self._format_traffic_context(recent_events)}

📈 HISTORIAL DE OPTIMIZACIÓN:
{optimization_stats}

🔍 DATOS DEL CRAWLER:
{self._format_crawler_context()}

🧠 ANÁLISIS PREDICTIVO:
{self._format_markov_context()}

PREGUNTA DEL USUARIO:
{user_question}

INSTRUCCIONES:
1. Analiza la pregunta en el contexto de nuestro sistema VRP en La Habana
2. Utiliza toda la información contextual disponible para dar una respuesta precisa
3. Si es relevante, menciona el impacto del clima actual en las rutas
4. Proporciona recomendaciones específicas basadas en los datos
5. Si falta información, menciona qué datos adicionales serían útiles
6. Mantén un tono profesional pero accesible
7. Incluye métricas específicas cuando sea apropiado

Responde de manera estructurada y completa:
"""
        
        return context_prompt
    
    def ask_with_context(self, user_question: str) -> Dict[str, Any]:
        """
        Procesa una pregunta del usuario con todo el contexto disponible
        Ahora usa el sistema de recuperación híbrido
        
        Args:
            user_question: Pregunta del usuario
            
        Returns:
            Respuesta estructurada con análisis y recomendaciones
        """
        try:
            print(f"🔍 RAG: Procesando pregunta: {user_question}")
            
            # Validar entrada
            if not user_question or not isinstance(user_question, str):
                return {
                    "success": False,
                    "message": "Pregunta inválida o vacía",
                    "error": "Invalid input"
                }
            
            print("🔍 RAG: Determinando categoría de pregunta...")
            # Determinar categoría de pregunta para búsqueda dirigida
            question_category = self._categorize_question(user_question)
            print(f"🔍 RAG: Categoría identificada: {question_category}")
            
            # Búsqueda contextual en base de datos vectorial
            try:
                print("🔍 RAG: Iniciando búsqueda en sistema IR...")
                retrieved_context = self.ir_system.search(
                    query=user_question,
                    top_k=10,
                    use_hybrid=True,
                    context_type=question_category
                )
                print(f"🔍 RAG: Contexto recuperado: {len(retrieved_context)} documentos")
            except Exception as e:
                print(f"❌ RAG: Error en búsqueda vectorial: {e}")
                retrieved_context = []
            
            # Generar prompt mejorado con contexto recuperado
            try:
                print("🔍 RAG: Generando prompt contextualizado...")
                enhanced_prompt = self._generate_enhanced_prompt(user_question, retrieved_context)
                print(f"🔍 RAG: Prompt generado (longitud: {len(enhanced_prompt)} chars)")
            except Exception as e:
                print(f"❌ RAG: Error generando prompt: {e}")
                enhanced_prompt = f"Eres un asistente VRP. Pregunta: {user_question}"
            
            # Obtener respuesta de Gemini
            try:
                print("🔍 RAG: Consultando a Gemini...")
                response = self.gemini.ask(enhanced_prompt)
                print(f"🔍 RAG: Respuesta de Gemini recibida (longitud: {len(str(response))} chars)")
            except Exception as e:
                print(f"❌ RAG: Error consultando Gemini: {e}")
                response = f"Lo siento, no pude procesar tu pregunta sobre: {user_question}. Error técnico."
            
            # Generar métricas relevantes
            try:
                print("🔍 RAG: Generando métricas relevantes...")
                relevant_metrics = self._generate_relevant_metrics(question_category)
                print("🔍 RAG: Métricas generadas exitosamente")
            except Exception as e:
                print(f"❌ RAG: Error generando métricas: {e}")
                relevant_metrics = {"error": str(e)}
            
            print("✅ RAG: Procesamiento completado exitosamente")
            return {
                "success": True,
                "response": response,
                "question_category": question_category,
                "relevant_metrics": relevant_metrics,
                "context_used": {
                    "retrieved_documents": len(retrieved_context),
                    "vector_db_available": True,
                    "lsi_system_available": self.ir_system.doc_lsi is not None,
                    "sources_used": list(set([doc.get('source', 'unknown') for doc in retrieved_context])),
                    "collections_searched": list(set([doc.get('collection', 'unknown') for doc in retrieved_context if 'collection' in doc]))
                },
                "retrieved_context": [
                    {
                        "id": doc['id'],
                        "score": doc['score'],
                        "source": doc.get('source', 'unknown'),
                        "snippet": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                    }
                    for doc in retrieved_context[:5]  # Solo top 5 para metadata
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ RAG: Error en ask_with_context: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "message": "Error procesando la consulta. Por favor, intenta reformular tu pregunta.",
                "fallback_used": True
            }
    
    def _generate_enhanced_prompt(self, user_question: str, retrieved_context: List[Dict[str, Any]]) -> str:
        """
        Genera un prompt mejorado usando el contexto recuperado
        
        Args:
            user_question: Pregunta del usuario
            retrieved_context: Contexto recuperado del sistema IR
            
        Returns:
            Prompt contextualizado
        """
        try:
            # Información del sistema actual (mantenemos para compatibilidad)
            current_weather = self.knowledge_base.get("weather_data", {})
            current_routes = self.knowledge_base.get("route_statistics", {})
            recent_events = self.knowledge_base.get("traffic_events", [])[-3:]
            optimization_stats = self._get_optimization_summary()
            
            # Formatear contexto recuperado por fuente
            vector_context = [doc for doc in retrieved_context if doc.get('source') == 'vector_db']
            lsi_context = [doc for doc in retrieved_context if doc.get('source') == 'lsi']
            
            context_prompt = f"""
Eres un asistente especializado en sistemas de ruteo de vehículos (VRP) para logística urbana en La Habana, Cuba.

INFORMACIÓN CONTEXTUAL ACTUAL:

🌍 UBICACIÓN Y CONFIGURACIÓN:
- Ciudad: {self.vrp_context['location']}
- Coordenadas: {self.vrp_context['coordinates']}
- Métodos de optimización disponibles: {', '.join(self.vrp_context['optimization_methods'])}

🌤️ INFORMACIÓN CLIMÁTICA ACTUAL:
{self._format_weather_context(current_weather)}

📊 ESTADÍSTICAS DE RUTAS ACTUALES:
{self._format_route_context(current_routes)}

🚦 EVENTOS DE TRÁFICO RECIENTES:
{self._format_traffic_context(recent_events)}

📈 HISTORIAL DE OPTIMIZACIÓN:
{optimization_stats}

📚 CONTEXTO RECUPERADO DE BASE DE CONOCIMIENTOS:
{self._format_retrieved_context(retrieved_context)}

🔍 DATOS DEL CRAWLER:
{self._format_crawler_context()}

🧠 ANÁLISIS PREDICTIVO:
{self._format_markov_context()}

PREGUNTA DEL USUARIO:
{user_question}

INSTRUCCIONES:
1. Analiza la pregunta en el contexto de nuestro sistema VRP en La Habana
2. Utiliza prioritariamente la información del contexto recuperado de la base de conocimientos
3. Combina esta información con los datos actuales del sistema
4. Si es relevante, menciona el impacto del clima actual en las rutas
5. Proporciona recomendaciones específicas basadas en los datos
6. Si falta información, menciona qué datos adicionales serían útiles
7. Mantén un tono profesional pero accesible
8. Incluye métricas específicas cuando sea apropiado
9. Cita las fuentes de información cuando sea relevante

Responde de manera estructurada y completa, priorizando la información más relevante y reciente:
"""
            
            return context_prompt
            
        except Exception as e:
            print(f"Error generando prompt: {e}")
            # Prompt de fallback más simple
            return f"""
Eres un asistente especializado en sistemas de ruteo de vehículos (VRP) para La Habana, Cuba.

PREGUNTA DEL USUARIO:
{user_question}

Responde basándote en tu conocimiento sobre optimización de rutas, logística urbana y las condiciones específicas de La Habana.
"""
    
    def _format_retrieved_context(self, retrieved_context: List[Dict[str, Any]]) -> str:
        """Formatea el contexto recuperado para incluir en el prompt"""
        try:
            if not retrieved_context or not isinstance(retrieved_context, list):
                return "No se encontró contexto específico adicional"
            
            formatted_sections = []
            
            # Agrupar por fuente
            by_source = {}
            for doc in retrieved_context:
                if not isinstance(doc, dict):
                    continue
                source = doc.get('source', 'unknown')
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append(doc)
            
            for source, docs in by_source.items():
                if source == 'vector_db':
                    # Agrupar por colección
                    by_collection = {}
                    for doc in docs:
                        collection = doc.get('collection', 'general')
                        if collection not in by_collection:
                            by_collection[collection] = []
                        by_collection[collection].append(doc)
                    
                    for collection, collection_docs in by_collection.items():
                        section = f"\n--- {collection.replace('_', ' ').title()} (Base de Datos Vectorial) ---"
                        for doc in collection_docs[:3]:  # Top 3 por colección
                            content = doc.get('content', '')
                            if isinstance(content, str) and len(content) > 0:
                                snippet = content[:150] + "..." if len(content) > 150 else content
                                score = doc.get('score', 0)
                                section += f"\n• [Score: {score:.3f}] {snippet}"
                        formatted_sections.append(section)
                
                elif source == 'lsi':
                    section = f"\n--- Análisis Semántico LSI ---"
                    for doc in docs[:3]:  # Top 3 documentos LSI
                        content = doc.get('content', '')
                        if isinstance(content, str) and len(content) > 0:
                            snippet = content[:150] + "..." if len(content) > 150 else content
                            score = doc.get('score', 0)
                            section += f"\n• [Score: {score:.3f}] {snippet}"
                    formatted_sections.append(section)
            
            return "\n".join(formatted_sections) if formatted_sections else "Contexto técnico no disponible"
            
        except Exception as e:
            print(f"Error formateando contexto recuperado: {e}")
            return "Error procesando contexto recuperado"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado completo del sistema RAG"""
        return {
            "rag_system": {
                "knowledge_base_entries": {
                    "weather_data": bool(self.knowledge_base.get("weather_data")),
                    "route_statistics": bool(self.knowledge_base.get("route_statistics")),
                    "traffic_events": len(self.knowledge_base.get("traffic_events", [])),
                    "optimization_history": len(self.knowledge_base.get("optimization_history", [])),
                    "system_performance": bool(self.knowledge_base.get("system_performance")),
                    "crawler_data": bool(self.knowledge_base.get("crawler_data"))
                }
            },
            "ir_system": self.ir_system.get_system_stats(),
            "timestamp": datetime.now().isoformat()
        }
    
    def cleanup_old_data(self, days_to_keep: int = 7):
        """Limpia datos antiguos del sistema completo"""
        # Limpiar base de datos vectorial
        self.ir_system.cleanup(days_to_keep)
        
        # Limpiar datos en memoria si son muy antiguos
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Filtrar eventos de tráfico antiguos
        if self.knowledge_base.get("traffic_events"):
            recent_events = []
            for event in self.knowledge_base["traffic_events"]:
                try:
                    event_date = datetime.fromisoformat(event.get("timestamp", ""))
                    if event_date > cutoff_date:
                        recent_events.append(event)
                except:
                    # Mantener eventos sin timestamp válido por seguridad
                    recent_events.append(event)
            self.knowledge_base["traffic_events"] = recent_events
        
        # Filtrar historial de optimización antiguo
        if self.knowledge_base.get("optimization_history"):
            recent_optimizations = []
            for opt in self.knowledge_base["optimization_history"]:
                try:
                    opt_date = datetime.fromisoformat(opt.get("timestamp", ""))
                    if opt_date > cutoff_date:
                        recent_optimizations.append(opt)
                except:
                    recent_optimizations.append(opt)
            self.knowledge_base["optimization_history"] = recent_optimizations
    
    def _analyze_weather_impact(self, weather_data: Dict) -> Dict:
        """Analiza el impacto del clima en las rutas"""
        try:
            if not isinstance(weather_data, dict):
                return {"impact_level": "Desconocido", "recommendation": "Datos inválidos", "factor": 1.0}
            
            impact_factor = weather_data.get('impact_factor', 1.0)
            
            # Validar que el factor sea numérico
            try:
                impact_factor = float(impact_factor)
            except (ValueError, TypeError):
                impact_factor = 1.0
            
            if impact_factor <= 1.1:
                impact_level = "Mínimo"
                recommendation = "Condiciones ideales para entregas"
            elif impact_factor <= 1.3:
                impact_level = "Bajo"
                recommendation = "Ligero aumento en tiempos de entrega"
            elif impact_factor <= 1.6:
                impact_level = "Moderado"
                recommendation = "Considerar rutas alternativas"
            elif impact_factor <= 2.0:
                impact_level = "Alto"
                recommendation = "Retrasos esperados, ajustar horarios"
            else:
                impact_level = "Severo"
                recommendation = "Considerar postponer entregas no urgentes"
            
            return {
                "impact_level": impact_level,
                "recommendation": recommendation,
                "factor": impact_factor
            }
        except Exception as e:
            print(f"Error analizando impacto climático: {e}")
            return {"impact_level": "Error", "recommendation": "No se pudo analizar", "factor": 1.0}
    
    def _calculate_route_efficiency(self, route_data: Dict) -> Dict:
        """Calcula métricas de eficiencia de rutas"""
        try:
            if not isinstance(route_data, dict):
                return {"efficiency": 0, "metrics": {}}
            
            routes = route_data.get('routes', [])
            
            if not routes or not isinstance(routes, list):
                return {"efficiency": 0, "metrics": {}}
            
            total_distance = 0
            total_points = 0
            
            for route in routes:
                if isinstance(route, dict):
                    distance = route.get('distance', 0)
                    path = route.get('path', [])
                    
                    try:
                        total_distance += float(distance) if distance is not None else 0
                        total_points += len(path) if isinstance(path, list) else 0
                    except (ValueError, TypeError):
                        continue
            
            avg_distance = total_distance / len(routes) if routes else 0
            efficiency_score = (total_points / total_distance * 100) if total_distance > 0 else 0
            
            return {
                "total_distance": round(total_distance, 2),
                "average_distance_per_route": round(avg_distance, 2),
                "total_delivery_points": total_points,
                "routes_count": len(routes),
                "efficiency_score": round(efficiency_score, 2)
            }
        
        except Exception as e:
            print(f"Error calculando eficiencia de rutas: {e}")
            return {"efficiency": 0, "metrics": {}, "error": str(e)}
    
    def _get_optimization_summary(self) -> str:
        """Genera resumen del historial de optimización"""
        history = self.knowledge_base.get("optimization_history", [])
        
        if not history:
            return "No hay historial de optimización disponible"
        
        recent_optimization = history[-1] if history else {}
        avg_time = np.mean([opt.get('computation_time', 0) for opt in history[-5:]])
        
        return f"""
- Última optimización: {recent_optimization.get('timestamp', 'N/A')}
- Método utilizado: {recent_optimization.get('method', 'N/A')}
- Tiempo promedio de cálculo: {avg_time:.2f}s
- Optimizaciones realizadas: {len(history)}
"""
    
    def _format_weather_context(self, weather_data: Dict) -> str:
        """Formatea el contexto climático"""
        if not weather_data:
            return "No hay información climática disponible"
        
        impact_factor = weather_data.get('impact_factor', 1.0)
        interpretation = weather_data.get('interpretation', 'Sin análisis')
        weather_summary = weather_data.get('weather_summary', {})
        
        temp = weather_summary.get('temperature_2m', 'N/A')
        precip = weather_summary.get('precipitation', 'N/A')
        wind = weather_summary.get('wind_speed_10m', 'N/A')
        
        return f"""
- Factor de impacto: {impact_factor:.2f}x
- Interpretación: {interpretation}
- Temperatura: {temp}°C
- Precipitación: {precip}mm
- Viento: {wind}km/h
"""
    
    def _format_route_context(self, route_data: Dict) -> str:
        """Formatea el contexto de rutas"""
        if not route_data:
            return "No hay rutas optimizadas disponibles"
        
        efficiency = route_data.get('efficiency_metrics', {})
        routes = route_data.get('routes', [])
        
        return f"""
- Número de rutas: {len(routes)}
- Distancia total: {efficiency.get('total_distance', 0):.2f} km
- Distancia promedio por ruta: {efficiency.get('average_distance_per_route', 0):.2f} km
- Puntos de entrega totales: {efficiency.get('total_delivery_points', 0)}
- Puntuación de eficiencia: {efficiency.get('efficiency_score', 0):.2f}/100
"""
    
    def _format_traffic_context(self, traffic_events: List) -> str:
        """Formatea el contexto de eventos de tráfico"""
        if not traffic_events:
            return "No hay eventos de tráfico recientes"
        
        events_summary = []
        for event in traffic_events[-3:]:  # Últimos 3 eventos
            events_summary.append(f"- {event.get('type', 'Evento')}: {event.get('description', 'Sin descripción')}")
        
        return "\n".join(events_summary)
    
    def _format_crawler_context(self) -> str:
        """Formatea el contexto de datos del crawler"""
        crawler_data = self.knowledge_base.get("crawler_data", {})
        
        if not crawler_data:
            return "No hay datos del crawler disponibles"
        
        return f"""
- Última actualización: {crawler_data.get('timestamp', 'N/A')}
- Fuentes consultadas: {len(crawler_data.get('sources', []))}
- Eventos relevantes encontrados: {crawler_data.get('relevant_events_count', 0)}
"""
    
    def _format_markov_context(self) -> str:
        """Formatea el contexto del análisis de Markov"""
        markov_data = self.knowledge_base.get("markov_insights", {})
        
        if not markov_data:
            return "Análisis predictivo no disponible"
        
        return f"""
- Predicción climática próximas 6h: {markov_data.get('weather_trend', 'Estable')}
- Probabilidad de condiciones adversas: {markov_data.get('adverse_probability', 0)*100:.1f}%
- Recomendación temporal: {markov_data.get('timing_recommendation', 'Sin recomendación')}
"""
    
    def _categorize_question(self, question: str) -> str:
        """Categoriza el tipo de pregunta del usuario"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['clima', 'tiempo', 'lluvia', 'viento', 'temperatura']):
            return "weather"
        elif any(word in question_lower for word in ['ruta', 'optimización', 'camión', 'entrega', 'distancia']):
            return "routing"
        elif any(word in question_lower for word in ['tráfico', 'congestion', 'eventos', 'incidentes']):
            return "traffic"
        elif any(word in question_lower for word in ['rendimiento', 'estadísticas', 'métricas', 'eficiencia']):
            return "performance"
        elif any(word in question_lower for word in ['predicción', 'futuro', 'tendencia', 'pronóstico']):
            return "prediction"
        else:
            return "general"
    
    def _generate_relevant_metrics(self, category: str) -> Dict:
        """Genera métricas relevantes según la categoría de pregunta"""
        current_weather = self.knowledge_base.get("weather_data", {})
        current_routes = self.knowledge_base.get("route_statistics", {})
        
        base_metrics = {
            "system_status": "Operativo",
            "last_update": datetime.now().strftime("%H:%M:%S")
        }
        
        if category == "weather":
            base_metrics.update({
                "weather_impact_factor": current_weather.get('impact_factor', 1.0),
                "current_conditions": current_weather.get('weather_summary', {})
            })
        
        elif category == "routing":
            efficiency = current_routes.get('efficiency_metrics', {})
            base_metrics.update({
                "active_routes": efficiency.get('routes_count', 0),
                "total_distance": efficiency.get('total_distance', 0),
                "efficiency_score": efficiency.get('efficiency_score', 0)
            })
        
        elif category == "performance":
            optimization_history = self.knowledge_base.get("optimization_history", [])
            base_metrics.update({
                "optimizations_completed": len(optimization_history),
                "avg_computation_time": np.mean([opt.get('computation_time', 0) 
                                               for opt in optimization_history[-5:]]) if optimization_history else 0
            })
        
        return base_metrics

# Función de conveniencia para usar en otros módulos
def create_vrp_rag_assistant() -> VRPKnowledgeRAG:
    """Crea una instancia del asistente RAG para VRP"""
    return VRPKnowledgeRAG()

if __name__ == "__main__":
    # Ejemplo de uso
    rag = VRPKnowledgeRAG()
    
    # Simular datos de prueba
    rag.update_knowledge_base("weather", {
        "impact_factor": 1.3,
        "interpretation": "Condiciones moderadas",
        "weather_summary": {
            "temperature_2m": 28,
            "precipitation": 2,
            "wind_speed_10m": 15
        }
    })
    
    rag.update_knowledge_base("routes", {
        "routes": [
            {"distance": 15.5, "path": [1, 2, 3, 4, 5]},
            {"distance": 12.3, "path": [1, 6, 7, 8]}
        ]
    })
    
    # Probar consulta
    response = rag.ask_with_context("¿Cómo está afectando el clima actual a mis rutas de entrega?")
    print(json.dumps(response, indent=2, ensure_ascii=False))