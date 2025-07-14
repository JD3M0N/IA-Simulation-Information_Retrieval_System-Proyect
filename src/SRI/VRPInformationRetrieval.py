"""
Sistema de Recuperación de Información Adaptado para VRP
Integra técnicas de LSI con base de datos vectorial para optimización de consultas VRP
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Importar la base de datos vectorial
from .VectorDatabase import VRPVectorDatabase

# Asegurar que punkt esté descargado
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def tokenize(text):
    """Tokenización básica"""
    return [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]

class VRPInformationRetrievalSystem:
    """
    Sistema de Recuperación de Información especializado para VRP
    Combina LSI tradicional con base de datos vectorial moderna
    """
    
    def __init__(self, persist_directory: str = "vector_cache"):
        """
        Inicializa el sistema de recuperación de información
        
        Args:
            persist_directory: Directorio para persistir la base de datos vectorial
        """
        # Configuración de TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Más flexible para dominio específico
            max_df=0.8,
            sublinear_tf=True,
            max_features=5000  # Límite para eficiencia
        )
        
        # Configuración LSI
        self.svd_model = TruncatedSVD(n_components=100, random_state=42)
        self.doc_lsi = None
        self.tfidf_matrix = None
        
        # Procesamiento de texto
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            # Fallback si no está instalado spacy en inglés
            self.nlp = None
        
        # Base de datos vectorial
        self.vector_db = VRPVectorDatabase(persist_directory)
        
        # Configuración para expansión de consultas (Rocchio)
        self.alpha = 1.0    # Peso query original
        self.beta = 0.75    # Peso documentos relevantes
        self.gamma = 0.15   # Peso documentos no relevantes
        self.R = 3          # Top R docs como relevantes
        self.M = 2          # Next M docs como no relevantes
        
        # Cache de documentos y embeddings
        self.documents = []
        self.doc_ids = []
        self.doc_metadata = []
        
        # Conocimiento específico VRP
        self.vrp_knowledge = self._initialize_vrp_knowledge()
        
    def _initialize_vrp_knowledge(self) -> Dict[str, List[str]]:
        """Inicializa conocimiento específico del dominio VRP"""
        return {
            "optimization_methods": [
                "algoritmo genético", "genetic algorithm", "búsqueda tabú", "tabu search",
                "recocido simulado", "simulated annealing", "búsqueda local", "local search",
                "VNS", "variable neighborhood search", "ACO", "ant colony optimization"
            ],
            "weather_factors": [
                "precipitación", "precipitation", "lluvia", "rain", "viento", "wind",
                "temperatura", "temperature", "visibilidad", "visibility", "clima", "weather"
            ],
            "traffic_terms": [
                "tráfico", "traffic", "congestión", "congestion", "incidente", "incident",
                "accidente", "accident", "obras", "construction", "cierre", "closure"
            ],
            "route_metrics": [
                "distancia", "distance", "tiempo", "time", "eficiencia", "efficiency",
                "costo", "cost", "combustible", "fuel", "capacidad", "capacity"
            ],
            "locations_havana": [
                "habana", "havana", "vedado", "miramar", "centro habana", "plaza",
                "malecón", "malecon", "aeropuerto", "airport", "puerto", "port"
            ]
        }
    
    def preprocess(self, text: str) -> str:
        """
        Preprocesa texto con énfasis en términos VRP
        
        Args:
            text: Texto a procesar
            
        Returns:
            Texto preprocesado
        """
        if self.nlp:
            doc = self.nlp(text.lower())
            tokens = [
                token.lemma_.lower() 
                for token in doc 
                if token.is_alpha and not token.is_stop
            ]
        else:
            # Fallback sin spacy
            tokens = [w.lower() for w in nltk.word_tokenize(text.lower()) if w.isalpha()]
        
        # Expandir términos específicos del dominio
        expanded_tokens = []
        for token in tokens:
            expanded_tokens.append(token)
            # Añadir sinónimos del dominio VRP
            for category, terms in self.vrp_knowledge.items():
                if token in terms:
                    expanded_tokens.extend([t for t in terms if t != token][:2])  # Máximo 2 sinónimos
        
        return " ".join(expanded_tokens)
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Indexa documentos en ambos sistemas (LSI y vectorial)
        
        Args:
            documents: Lista de documentos con campos 'id', 'content', 'metadata'
        """
        self.documents = []
        self.doc_ids = []
        self.doc_metadata = []
        
        # Preparar documentos para LSI
        for doc in documents:
            doc_id = doc.get('id', f"doc_{len(self.doc_ids)}")
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Preprocesar contenido
            processed_content = self.preprocess(content)
            
            self.documents.append(processed_content)
            self.doc_ids.append(doc_id)
            self.doc_metadata.append(metadata)
            
            # Añadir a base de datos vectorial
            collection_name = self._determine_collection(metadata)
            self.vector_db.add_document(
                collection_name=collection_name,
                content=content,
                metadata=metadata,
                doc_id=doc_id
            )
        
        # Entrenar modelo LSI
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            raw_lsi = self.svd_model.fit_transform(self.tfidf_matrix)
            
            # Normalizar vectores LSI
            norms = np.linalg.norm(raw_lsi, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.doc_lsi = raw_lsi / norms
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               use_hybrid: bool = True,
               context_type: str = "all") -> List[Dict[str, Any]]:
        """
        Búsqueda híbrida combinando LSI y búsqueda vectorial
        
        Args:
            query: Consulta del usuario
            top_k: Número de resultados a retornar
            use_hybrid: Si usar búsqueda híbrida o solo LSI
            context_type: Tipo de contexto para búsqueda vectorial
            
        Returns:
            Lista de documentos relevantes con scores
        """
        if not self.documents:
            return []
        
        # Preprocesar query
        processed_query = self.preprocess(query)
        
        results = []
        
        if use_hybrid:
            # Búsqueda vectorial
            vector_results = self.vector_db.search_contextual(
                query=query,
                context_type=context_type
            )
            
            # Convertir resultados vectoriales
            for result in vector_results[:top_k//2]:
                results.append({
                    'id': result['id'],
                    'content': result['document'],
                    'metadata': result['metadata'],
                    'score': result['similarity'],
                    'source': 'vector_db',
                    'collection': result['collection']
                })
        
        # Búsqueda LSI con expansión Rocchio
        if self.doc_lsi is not None:
            lsi_results = self._search_lsi_with_rocchio(processed_query, top_k//2 if use_hybrid else top_k)
            results.extend(lsi_results)
        
        # Combinar y reordenar resultados
        if use_hybrid and len(results) > 1:
            results = self._combine_results(results, query)
        
        return results[:top_k]
    
    def _search_lsi_with_rocchio(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda LSI con expansión de consultas Rocchio"""
        # Vector LSI de la consulta
        q_tfidf = self.vectorizer.transform([query])
        q_lsi = self.svd_model.transform(q_tfidf).flatten()
        q_vec = q_lsi / (np.linalg.norm(q_lsi) + 1e-9)
        
        # Ranking inicial
        similarities = self.doc_lsi @ q_vec
        candidate_indices = np.argsort(similarities)[::-1][:min(50, len(similarities))]
        
        if len(candidate_indices) > self.R + self.M:
            # Aplicar Rocchio
            relevant_indices = candidate_indices[:self.R]
            non_relevant_indices = candidate_indices[self.R:self.R + self.M]
            
            # Calcular centroides
            centroid_rel = self.doc_lsi[relevant_indices].mean(axis=0) if len(relevant_indices) > 0 else 0
            centroid_non_rel = self.doc_lsi[non_relevant_indices].mean(axis=0) if len(non_relevant_indices) > 0 else 0
            
            # Expandir consulta
            q_expanded = (self.alpha * q_vec + 
                         self.beta * centroid_rel - 
                         self.gamma * centroid_non_rel)
            q_expanded /= np.linalg.norm(q_expanded) + 1e-9
            
            # Recalcular similitudes
            new_similarities = self.doc_lsi[candidate_indices] @ q_expanded
            
            # Filtro dinámico
            if len(new_similarities) > 0:
                threshold = 0.6 * np.max(new_similarities)
                valid_mask = new_similarities >= threshold
                valid_indices = candidate_indices[valid_mask]
                valid_similarities = new_similarities[valid_mask]
            else:
                valid_indices = candidate_indices[:top_k]
                valid_similarities = similarities[valid_indices]
        else:
            valid_indices = candidate_indices
            valid_similarities = similarities[valid_indices]
        
        # Crear resultados
        results = []
        sorted_indices = np.argsort(valid_similarities)[::-1][:top_k]
        
        for i in sorted_indices:
            doc_idx = valid_indices[i]
            results.append({
                'id': self.doc_ids[doc_idx],
                'content': self.documents[doc_idx],
                'metadata': self.doc_metadata[doc_idx],
                'score': float(valid_similarities[i]),
                'source': 'lsi'
            })
        
        return results
    
    def _combine_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Combina y reordena resultados de múltiples fuentes"""
        # Normalizar scores entre fuentes
        vector_results = [r for r in results if r['source'] == 'vector_db']
        lsi_results = [r for r in results if r['source'] == 'lsi']
        
        # Normalizar scores vectoriales
        if vector_results:
            vector_scores = [r['score'] for r in vector_results]
            min_score, max_score = min(vector_scores), max(vector_scores)
            if max_score > min_score:
                for r in vector_results:
                    r['normalized_score'] = (r['score'] - min_score) / (max_score - min_score)
            else:
                for r in vector_results:
                    r['normalized_score'] = 1.0
        
        # Normalizar scores LSI
        if lsi_results:
            lsi_scores = [r['score'] for r in lsi_results]
            min_score, max_score = min(lsi_scores), max(lsi_scores)
            if max_score > min_score:
                for r in lsi_results:
                    r['normalized_score'] = (r['score'] - min_score) / (max_score - min_score)
            else:
                for r in lsi_results:
                    r['normalized_score'] = 1.0
        
        # Combinar con pesos
        weight_vector = 0.6  # Mayor peso a búsqueda vectorial
        weight_lsi = 0.4
        
        for r in vector_results:
            r['combined_score'] = weight_vector * r['normalized_score']
        
        for r in lsi_results:
            r['combined_score'] = weight_lsi * r['normalized_score']
        
        # Eliminar duplicados por ID
        seen_ids = set()
        unique_results = []
        for r in results:
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                unique_results.append(r)
        
        # Ordenar por score combinado
        unique_results.sort(key=lambda x: x.get('combined_score', x['score']), reverse=True)
        
        return unique_results
    
    def _determine_collection(self, metadata: Dict[str, Any]) -> str:
        """Determina la colección apropiada basada en metadatos"""
        doc_type = metadata.get('type', '').lower()
        source = metadata.get('source', '').lower()
        
        if 'weather' in doc_type or 'clima' in doc_type:
            return 'weather_data'
        elif 'route' in doc_type or 'ruta' in doc_type:
            return 'route_analysis'
        elif 'traffic' in doc_type or 'trafico' in doc_type:
            return 'traffic_events'
        elif 'performance' in doc_type or 'rendimiento' in doc_type:
            return 'system_performance'
        elif 'historical' in doc_type or 'historico' in doc_type:
            return 'historical_insights'
        else:
            return 'knowledge_base'
    
    def add_real_time_data(self, data_type: str, data: Dict[str, Any]):
        """
        Añade datos en tiempo real al sistema
        
        Args:
            data_type: Tipo de datos ('weather', 'traffic', 'route', 'performance')
            data: Datos a añadir
        """
        timestamp = datetime.now().isoformat()
        
        # Determinar método de adición basado en tipo
        if data_type == 'weather':
            self.vector_db.add_weather_data(data)
        elif data_type == 'traffic':
            self.vector_db.add_traffic_event(data)
        elif data_type == 'route':
            self.vector_db.add_route_analysis(data)
        elif data_type == 'performance':
            self.vector_db.add_system_performance(data)
        else:
            # Datos genéricos
            content = str(data)
            metadata = {'type': data_type, 'timestamp': timestamp, 'source': 'real_time'}
            self.vector_db.add_document('knowledge_base', content, metadata)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de recuperación"""
        stats = {
            'lsi_documents': len(self.documents),
            'lsi_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            'lsi_components': self.svd_model.n_components if hasattr(self.svd_model, 'n_components') else 0,
            'vector_db_stats': self.vector_db.get_collection_stats(),
            'last_update': datetime.now().isoformat()
        }
        return stats
    
    def cleanup(self, days_to_keep: int = 7):
        """Limpia datos antiguos del sistema"""
        self.vector_db.cleanup_old_data(days_to_keep)

# Función de conveniencia
def create_vrp_ir_system(persist_dir: str = "vector_cache") -> VRPInformationRetrievalSystem:
    """Crea una instancia del sistema de recuperación VRP"""
    return VRPInformationRetrievalSystem(persist_dir)

if __name__ == "__main__":
    # Ejemplo de uso
    ir_system = VRPInformationRetrievalSystem("test_ir_cache")
    
    # Documentos de ejemplo
    sample_docs = [
        {
            'id': 'weather_001',
            'content': 'Condiciones meteorológicas en La Habana: temperatura 28°C, precipitación 2mm, viento 15km/h',
            'metadata': {'type': 'weather', 'location': 'Habana', 'timestamp': datetime.now().isoformat()}
        },
        {
            'id': 'route_001',
            'content': 'Ruta optimizada con algoritmo genético: 3 vehículos, 15 puntos de entrega, distancia total 45.6km',
            'metadata': {'type': 'route', 'method': 'genetic_algorithm', 'timestamp': datetime.now().isoformat()}
        }
    ]
    
    # Indexar documentos
    ir_system.index_documents(sample_docs)
    
    # Probar búsqueda
    results = ir_system.search("¿Cómo afecta el clima a las rutas de entrega?")
    
    print(f"Encontrados {len(results)} resultados:")
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Score: {result['score']:.3f}")
        print(f"Fuente: {result['source']}")
        print(f"Contenido: {result['content'][:100]}...")
        print("---")
