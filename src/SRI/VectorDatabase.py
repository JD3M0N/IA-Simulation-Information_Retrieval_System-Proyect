"""
Base de datos vectorial especializada para el sistema VRP-RAG
Utiliza ChromaDB para almacenamiento vectorial y Sentence Transformers para embeddings
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import json
import hashlib
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging

class VRPVectorDatabase:
    """
    Base de datos vectorial especializada para el sistema VRP
    Almacena y recupera información contextual para el RAG
    """
    
    def __init__(self, persist_directory: str = "vector_cache"):
        """
        Inicializa la base de datos vectorial
        
        Args:
            persist_directory: Directorio donde persistir la base de datos
        """
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configurar directorio de persistencia
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(exist_ok=True)
        
        # Inicializar ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Modelo de embeddings optimizado para español/dominio técnico
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Colecciones especializadas
        self.collections = {}
        self._initialize_collections()
        
    def _initialize_collections(self):
        """Inicializa las colecciones especializadas"""
        collection_configs = {
            "weather_data": {
                "description": "Información meteorológica e impacto en rutas",
                "metadata_keys": ["timestamp", "location", "impact_factor", "weather_type"]
            },
            "route_analysis": {
                "description": "Análisis de rutas optimizadas y métricas",
                "metadata_keys": ["timestamp", "optimization_method", "efficiency_score", "route_count"]
            },
            "traffic_events": {
                "description": "Eventos de tráfico e incidentes",
                "metadata_keys": ["timestamp", "location", "event_type", "severity", "impact_area"]
            },
            "system_performance": {
                "description": "Métricas de rendimiento del sistema",
                "metadata_keys": ["timestamp", "component", "metric_type", "value"]
            },
            "knowledge_base": {
                "description": "Base de conocimientos general VRP",
                "metadata_keys": ["timestamp", "category", "source", "relevance_score"]
            },
            "historical_insights": {
                "description": "Insights históricos y patrones identificados",
                "metadata_keys": ["timestamp", "pattern_type", "confidence", "time_period"]
            }
        }
        
        for name, config in collection_configs.items():
            try:
                # Intentar obtener colección existente
                collection = self.client.get_collection(name=name)
                self.logger.info(f"Colección '{name}' cargada exitosamente")
            except Exception:
                # Crear nueva colección si no existe
                collection = self.client.create_collection(
                    name=name,
                    metadata={"description": config["description"]}
                )
                self.logger.info(f"Colección '{name}' creada exitosamente")
            
            self.collections[name] = collection
    
    def add_document(self, 
                    collection_name: str, 
                    content: str, 
                    metadata: Dict[str, Any], 
                    doc_id: Optional[str] = None) -> str:
        """
        Añade un documento a la colección especificada
        
        Args:
            collection_name: Nombre de la colección
            content: Contenido textual del documento
            metadata: Metadatos del documento
            doc_id: ID del documento (se genera automáticamente si no se proporciona)
            
        Returns:
            ID del documento insertado
        """
        if collection_name not in self.collections:
            raise ValueError(f"Colección '{collection_name}' no existe")
        
        # Generar ID si no se proporciona
        if doc_id is None:
            doc_id = self._generate_doc_id(content, metadata)
        
        # Generar embedding
        embedding = self.embedding_model.encode(content).tolist()
        
        # Añadir timestamp si no existe
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Insertar en ChromaDB
        collection = self.collections[collection_name]
        collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        self.logger.info(f"Documento añadido a '{collection_name}' con ID: {doc_id}")
        return doc_id
    
    def search(self, 
              query: str, 
              collection_names: Optional[List[str]] = None,
              top_k: int = 5,
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Busca documentos similares al query
        
        Args:
            query: Texto de búsqueda
            collection_names: Colecciones donde buscar (todas si es None)
            top_k: Número máximo de resultados por colección
            filters: Filtros de metadata
            
        Returns:
            Lista de documentos encontrados con scores y metadata
        """
        if collection_names is None:
            collection_names = list(self.collections.keys())
        
        # Generar embedding del query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = []
        
        for collection_name in collection_names:
            if collection_name not in self.collections:
                continue
                
            collection = self.collections[collection_name]
            
            # Realizar búsqueda
            search_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters
            )
            
            # Procesar resultados
            if search_results['documents'] and search_results['documents'][0]:
                for i, doc in enumerate(search_results['documents'][0]):
                    result = {
                        'collection': collection_name,
                        'id': search_results['ids'][0][i],
                        'document': doc,
                        'metadata': search_results['metadatas'][0][i],
                        'distance': search_results['distances'][0][i],
                        'similarity': 1 - search_results['distances'][0][i]  # Convertir distancia a similitud
                    }
                    results.append(result)
        
        # Ordenar por similitud descendente
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results
    
    def add_weather_data(self, weather_info: Dict[str, Any]) -> str:
        """Añade información meteorológica"""
        content = self._format_weather_content(weather_info)
        metadata = {
            "location": weather_info.get("location", "La Habana, Cuba"),
            "impact_factor": weather_info.get("impact_factor", 1.0),
            "weather_type": weather_info.get("weather_type", "normal"),
            "source": "weather_api"
        }
        return self.add_document("weather_data", content, metadata)
    
    def add_route_analysis(self, route_data: Dict[str, Any]) -> str:
        """Añade análisis de rutas"""
        content = self._format_route_content(route_data)
        metadata = {
            "optimization_method": route_data.get("optimization_method", "unknown"),
            "efficiency_score": route_data.get("efficiency_score", 0),
            "route_count": len(route_data.get("routes", [])),
            "total_distance": route_data.get("total_distance", 0),
            "source": "route_optimizer"
        }
        return self.add_document("route_analysis", content, metadata)
    
    def add_traffic_event(self, event_data: Dict[str, Any]) -> str:
        """Añade evento de tráfico"""
        content = self._format_traffic_content(event_data)
        metadata = {
            "event_type": event_data.get("type", "unknown"),
            "severity": event_data.get("severity", "medium"),
            "location": event_data.get("location", "unknown"),
            "impact_area": event_data.get("impact_area", "local"),
            "source": "traffic_monitor"
        }
        return self.add_document("traffic_events", content, metadata)
    
    def add_system_performance(self, performance_data: Dict[str, Any]) -> str:
        """Añade métricas de rendimiento"""
        content = self._format_performance_content(performance_data)
        metadata = {
            "component": performance_data.get("component", "system"),
            "metric_type": performance_data.get("metric_type", "general"),
            "value": performance_data.get("value", 0),
            "unit": performance_data.get("unit", ""),
            "source": "performance_monitor"
        }
        return self.add_document("system_performance", content, metadata)
    
    def search_contextual(self, query: str, context_type: str = "all") -> List[Dict[str, Any]]:
        """
        Búsqueda contextual especializada para RAG
        
        Args:
            query: Pregunta del usuario
            context_type: Tipo de contexto a buscar ('weather', 'routes', 'traffic', 'performance', 'all')
            
        Returns:
            Resultados ordenados por relevancia
        """
        # Mapear tipos de contexto a colecciones
        context_mapping = {
            "weather": ["weather_data"],
            "routes": ["route_analysis"],
            "traffic": ["traffic_events"],
            "performance": ["system_performance"],
            "general": ["knowledge_base", "historical_insights"],
            "all": list(self.collections.keys())
        }
        
        collections = context_mapping.get(context_type, context_mapping["all"])
        
        # Búsqueda con filtros temporales (últimas 24 horas para datos operacionales)
        recent_filter = None
        if context_type in ["weather", "traffic", "performance"]:
            yesterday = (datetime.now() - timedelta(hours=24)).isoformat()
            recent_filter = {"timestamp": {"$gte": yesterday}}
        
        results = self.search(
            query=query,
            collection_names=collections,
            top_k=10,
            filters=recent_filter
        )
        
        return results[:15]  # Límite de contexto para RAG
    
    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene estadísticas de todas las colecciones"""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {
                    "document_count": count,
                    "description": collection.metadata.get("description", ""),
                    "last_updated": datetime.now().isoformat()
                }
            except Exception as e:
                stats[name] = {"error": str(e)}
        
        return stats
    
    def cleanup_old_data(self, days_to_keep: int = 7):
        """Limpia datos antiguos de colecciones operacionales"""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        operational_collections = ["weather_data", "traffic_events", "system_performance"]
        
        for collection_name in operational_collections:
            if collection_name in self.collections:
                collection = self.collections[collection_name]
                
                # ChromaDB no soporta eliminación por filtros directamente
                # Necesitamos obtener IDs y eliminar individualmente
                try:
                    old_docs = collection.get(
                        where={"timestamp": {"$lt": cutoff_date}}
                    )
                    
                    if old_docs['ids']:
                        collection.delete(ids=old_docs['ids'])
                        self.logger.info(f"Eliminados {len(old_docs['ids'])} documentos antiguos de {collection_name}")
                        
                except Exception as e:
                    self.logger.error(f"Error limpiando {collection_name}: {e}")
    
    def _generate_doc_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Genera un ID único para el documento"""
        # Crear hash basado en contenido y metadatos clave
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = metadata.get("timestamp", datetime.now().isoformat())
        source = metadata.get("source", "unknown")
        
        return f"{source}_{content_hash}_{timestamp[:10]}"
    
    def _format_weather_content(self, weather_info: Dict[str, Any]) -> str:
        """Formatea información meteorológica para almacenamiento"""
        summary = weather_info.get("weather_summary", {})
        
        content = f"""
Información Meteorológica - {weather_info.get('location', 'La Habana, Cuba')}
Temperatura: {summary.get('temperature_2m', 'N/A')}°C
Precipitación: {summary.get('precipitation', 'N/A')}mm
Velocidad del viento: {summary.get('wind_speed_10m', 'N/A')}km/h
Factor de impacto en rutas: {weather_info.get('impact_factor', 1.0)}x
Interpretación: {weather_info.get('interpretation', 'Condiciones normales')}
Recomendaciones: {weather_info.get('recommendations', 'Sin recomendaciones especiales')}
        """.strip()
        
        return content
    
    def _format_route_content(self, route_data: Dict[str, Any]) -> str:
        """Formatea análisis de rutas para almacenamiento"""
        routes = route_data.get("routes", [])
        efficiency = route_data.get("efficiency_metrics", {})
        
        content = f"""
Análisis de Rutas Optimizadas
Método de optimización: {route_data.get('optimization_method', 'No especificado')}
Número de rutas: {len(routes)}
Distancia total: {efficiency.get('total_distance', 0):.2f}km
Distancia promedio por ruta: {efficiency.get('average_distance_per_route', 0):.2f}km
Puntos de entrega totales: {efficiency.get('total_delivery_points', 0)}
Puntuación de eficiencia: {efficiency.get('efficiency_score', 0):.2f}/100
Tiempo de cálculo: {route_data.get('computation_time', 0):.2f}s
        """.strip()
        
        return content
    
    def _format_traffic_content(self, event_data: Dict[str, Any]) -> str:
        """Formatea eventos de tráfico para almacenamiento"""
        content = f"""
Evento de Tráfico
Tipo: {event_data.get('type', 'Evento desconocido')}
Ubicación: {event_data.get('location', 'Ubicación no especificada')}
Descripción: {event_data.get('description', 'Sin descripción')}
Severidad: {event_data.get('severity', 'Media')}
Área de impacto: {event_data.get('impact_area', 'Local')}
Duración estimada: {event_data.get('estimated_duration', 'No especificada')}
Rutas afectadas: {', '.join(event_data.get('affected_routes', []))}
        """.strip()
        
        return content
    
    def _format_performance_content(self, performance_data: Dict[str, Any]) -> str:
        """Formatea métricas de rendimiento para almacenamiento"""
        content = f"""
Métricas de Rendimiento del Sistema
Componente: {performance_data.get('component', 'Sistema general')}
Tipo de métrica: {performance_data.get('metric_type', 'General')}
Valor: {performance_data.get('value', 'N/A')} {performance_data.get('unit', '')}
Descripción: {performance_data.get('description', 'Sin descripción')}
Estado: {performance_data.get('status', 'Normal')}
Tendencia: {performance_data.get('trend', 'Estable')}
        """.strip()
        
        return content

# Función de conveniencia para crear instancia
def create_vector_database(persist_dir: str = "vector_cache") -> VRPVectorDatabase:
    """Crea una instancia de la base de datos vectorial"""
    return VRPVectorDatabase(persist_dir)

if __name__ == "__main__":
    # Ejemplo de uso
    vdb = VRPVectorDatabase("test_vector_cache")
    
    # Probar con datos de ejemplo
    weather_example = {
        "location": "La Habana, Cuba",
        "impact_factor": 1.3,
        "interpretation": "Condiciones moderadas",
        "weather_summary": {
            "temperature_2m": 28,
            "precipitation": 2,
            "wind_speed_10m": 15
        }
    }
    
    route_example = {
        "optimization_method": "Genetic Algorithm",
        "routes": [{"distance": 15.5}, {"distance": 12.3}],
        "efficiency_metrics": {
            "total_distance": 27.8,
            "efficiency_score": 85.2
        }
    }
    
    # Añadir datos
    weather_id = vdb.add_weather_data(weather_example)
    route_id = vdb.add_route_analysis(route_example)
    
    # Buscar
    results = vdb.search("¿Cómo afecta el clima a las rutas?")
    print(f"Encontrados {len(results)} resultados")
    
    for result in results:
        print(f"Colección: {result['collection']}")
        print(f"Similitud: {result['similarity']:.3f}")
        print(f"Documento: {result['document'][:100]}...")
        print("---")
