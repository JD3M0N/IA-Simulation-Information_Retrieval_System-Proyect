NAME = "Abel Ponce González"
GROUP = "311"
CAREER = "Ciencia de la Computación"
MODEL = "Modelo LSI con Expansión de Consultas"

"""
INFORMACIÓN EXTRA:

Fuente bibliográfica:
- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. Journal of the American Society for Information Science, 41(6), 391-407.
- Rocchio, J. J. (1971). Relevance feedback in information retrieval. In The SMART Retrieval System—Experiments in Automatic Document Processing (pp. 313-323).
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.

Mejora implementada:
- Técnica: Integración de poda eficiente mediante prefiltering con LSI y expansión de consultas con Rocchio, optimizando el conjunto de documentos candidatos antes del reranking.
- Beneficio: Mayor eficiencia computacional al procesar solo documentos candidatos prometedores y mejor precisión gracias a la expansión de consultas mediante retroalimentación simulada.

Definición del modelo:
Q: El espacio de consultas está formado por vectores en el espacio LSI (k=100 dimensiones) obtenidos a partir de la transformación de la representación TF-IDF inicial.
D: Los documentos se representan como vectores normalizados en el espacio LSI reducido (k=100 dimensiones).
F: La función de similitud se basa en el producto escalar de los vectores normalizados en el espacio LSI, refinada mediante el algoritmo de Rocchio.
R: La relación de relevancia se simula asumiendo que los primeros R documentos del ranking inicial son relevantes.

¿Dependencia entre los términos?
Sí, el modelo LSI captura la dependencia semántica entre términos mediante la descomposición SVD de la matriz término-documento. 

Correspondencia parcial documento-consulta:
Sí, el modelo permite correspondencia parcial entre documentos y consultas gracias a la proyección en el espacio semántico latente. Documentos que no comparten términos exactos con la consulta pueden ser recuperados si comparten términos semánticamente relacionados en el espacio LSI.

Ranking:
Sí, mediante proceso multi-etapa que combina similitud coseno en espacio LSI y refinamiento con Rocchio, finalizando con ordenación por similitud y filtrado por umbral.
"""


import ir_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Dict, List, Tuple
from sklearn.decomposition import TruncatedSVD
import spacy
import nltk

def tokenize(text):
    return [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]

class InformationRetrievalModel:
    def __init__(self):
        """
        Inicializa el modelo de recuperación de información.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words='english',    # o 'spanish' según tu corpus
            ngram_range=(1,2),       # unigrama + bigrama
            min_df=2,                # descartar términos raros
            max_df=0.75,              # descartar términos muy frecuentes
            sublinear_tf=True        # usa 1 + log(tf)
        )
        
        self.embed_dim: int = 300
        self.tfidf_matrix = None
        self.documents = []
        self.doc_ids = []
        self.dataset = None
        self.queries = {}
        
        # Para LSI
        self.svd_model = None       # TruncatedSVD tras TF-IDF
        self.doc_lsi = None         # representación de documentos en espacio LSI
        self.nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
        
        self.alpha = 1.0    # Peso de la query original
        self.beta = 0.95     # Peso documentos relevantes
        self.gamma = 0.3    # Peso documentos no relevantes
        self.R = 2           # Top R docs como relevantes
        self.M = 3          # Next M docs como no relevantes
        
    def preprocess(self, text: str) -> str:
        doc = self.nlp(text.lower())
        tokens = [
            token.lemma_.lower() 
            for token in doc 
            if token.is_alpha and not token.is_stop
        ]
        return " ".join(tokens)


    def _document_to_embedding(self, text: str) -> np.ndarray:
        """Convierte texto a embedding denso mediante promedio de vectores de palabras"""
        tokens = self.preprocess(text).split()
        vectors = []
        for word in tokens:
            try:
                vectors.append(self.embedding_model[word])
            except KeyError:
                vectors.append(np.zeros(self.embed_dim))  # Manejo OOV
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.embed_dim)
    
    
    def fit(self, dataset_name: str):
        """
        Carga y procesa un dataset de ir_datasets, incluyendo todas sus queries.
        
        Args:
            dataset_name (str): Nombre del dataset en ir_datasets (ej: 'cranfield')
        """
        # Cargar dataset
        self.dataset = ir_datasets.load(dataset_name)
        
        if not hasattr(self.dataset, 'queries_iter'):
            raise ValueError("Este dataset no tiene queries definidas")
        
        self.documents = []
        self.doc_ids = []
        
        for doc in self.dataset.docs_iter():
            self.doc_ids.append(doc.doc_id)
            # Incluir título con el texto si está disponible
            doc_text = doc.text
            if hasattr(doc, 'title') and doc.title:
                doc_text = f"{doc.title} {doc_text}"
            clean = self.preprocess(doc_text)
            self.documents.append(clean)
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        

        # Ajusta el modelo LSI (SVD truncado)
        self.svd_model = TruncatedSVD(n_components=100, random_state=42)
        
        # Entrena y normaliza representaciones LSI de documentos
        raw = self.svd_model.fit_transform(self.tfidf_matrix)               # (n_docs, k)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)                # (n_docs, 1)
        norms[norms == 0] = 1                                             # evita ceros
        self.doc_lsi = raw / norms
        
        raw_qs = { q.query_id: q.text for q in self.dataset.queries_iter() }
        
        self.queries = {
            qid: self.preprocess(txt)
            for qid, txt in raw_qs.items()
        }
        
    def predict(self, top_k: int = 100) -> Dict[str, Dict[str, List[str]]]:
        prefilter_k = 9  # Un valor mayor que top_k para mantener variedad
        results = {}
        for qid, query_text in self.queries.items():
            # Vector LSI original de la consulta
            q_tfidf = self.vectorizer.transform([query_text])
            q_lsi = self.svd_model.transform(q_tfidf).flatten()
            q_vec = q_lsi / (np.linalg.norm(q_lsi) + 1e-9)

            # Ranking inicial y poda - selección de candidatos
            sims = self.doc_lsi @ q_vec  
            candidate_idxs = np.argsort(sims)[::-1][:prefilter_k]
            
            # Paso 1: Seleccionar Cr (relevantes) y Cnr (no relevantes) de los candidatos
            Cr_indices = candidate_idxs[:self.R]                     
            Cnr_indices = candidate_idxs[self.R:self.R+self.M]       
    
            # Paso 2: Calcular centroides
            centroid_rel = self.doc_lsi[Cr_indices].mean(axis=0) if len(Cr_indices) > 0 else 0
            centroid_nonrel = self.doc_lsi[Cnr_indices].mean(axis=0) if len(Cnr_indices) > 0 else 0
            
            # Paso 3: Aplicar fórmula de Rocchio
            term_relevant = (self.beta/len(Cr_indices)) * centroid_rel if len(Cr_indices) > 0 else 0
            term_nonrelevant = (self.gamma/len(Cnr_indices)) * centroid_nonrel if len(Cnr_indices) > 0 else 0
            
            q_opt = self.alpha * q_vec + term_relevant - term_nonrelevant
            q_opt /= np.linalg.norm(q_opt) + 1e-9  # Normalizar

            # Paso 4: Re-calcular similitudes SOLO para los candidatos
            candidate_vectors = self.doc_lsi[candidate_idxs]
            new_sims = candidate_vectors @ q_opt
            
            # Umbral de similitud para filtrar documentos irrelevantes
            if len(new_sims) > 0:
                max_score = np.max(new_sims)
                dynamic_threshold = 0.7 * max_score  # 70% del valor máximo
            else:
                dynamic_threshold = 0.0
            
            # Filtrar usando el umbral dinámico en lugar del fijo
            filtered_indices = np.where(new_sims >= dynamic_threshold)[0]
            
            # Si no quedan documentos después del filtrado, usar todos los candidatos
            if len(filtered_indices) == 0:
                filtered_indices = np.arange(len(new_sims))
                
            # Ordenar los candidatos filtrados por similitud
            ranked = filtered_indices[np.argsort(new_sims[filtered_indices])[::-1][:top_k]]
            
            # Obtener los IDs de documento correspondientes
            top_docs = [self.doc_ids[candidate_idxs[i]] for i in ranked]
            
            results[qid] = {'text': query_text, 'results': top_docs}
            
        return results
    
    def evaluate(self, top_k: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Evalúa los resultados para TODAS las queries comparando con los qrels oficiales.
        
        Args:
            top_k (int): Número máximo de documentos a considerar por query.
            
        Returns:
            dict: Métricas de evaluación por query y métricas agregadas.
        """
        if not hasattr(self.dataset, 'qrels_iter'):
            raise ValueError("Este dataset no tiene relevancias definidas (qrels)")
        
        predictions = self.predict(top_k=top_k)
        
        qrels = {}
        for qrel in self.dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        
        result = {}
        
        for qid, data in predictions.items():
            if qid not in qrels:
                continue
                
            relevant_docs = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)
            retrieved_docs = set(data['results'])
            relevant_retrieved = relevant_docs & retrieved_docs
            
            result[qid] = {
                'all_relevant': relevant_docs,
                'all_retrieved': retrieved_docs,
                'relevant_retrieved': relevant_retrieved
            }
        
        return result
