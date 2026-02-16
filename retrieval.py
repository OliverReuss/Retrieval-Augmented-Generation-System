from embedding import Embedder
from sentence_transformers import CrossEncoder
import chromadb
import config
import os
import re
import time

class Retriever:

    def __init__(self) -> None:
        self.db_path = config.RETRIEVAL_DB_PATH
        self.collection_name = config.RETRIEVAL_COLLECTION_NAME
        self.top_k = config.RETRIEVAL_TOP_K
        self.reranking_top_k = config.RETRIEVAL_RERANKING_TOP_K
        self.stop_words_path = config.RETRIEVAL_STOP_WORDS_PATH
        self.model_path = config.MODELS_DIRECTORY
        self.stats_path = config.EVALUATION_OUTPUT_PATH
        self.cross_encoder_model_name = config.RETRIEVAL_CROSS_ENCODER_MODEL_NAME
        self.chroma_client = None
        self.collection = None
        self.embedder = Embedder()
        self.cross_encoder = None
        self.stop_words = None
        # Statistiken
        self.query_count = 0
        self.total_chunks_retrieved = 0
        self.retrieval_times_ms = []
        self.reranking_times_ms = []
        self.similarity_scores = []
    
    def connect_to_db(self) -> chromadb.Collection:
        if self.collection is not None:
            return self.collection
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        return self.collection
    
    def load_stop_words(self) -> list[str]:
        self.stop_words = []
        with open(self.stop_words_path, 'r', encoding='utf-8') as file:
            # Jede Zeile ein Wort
            for line in file:
                word = line.strip().lower()
                # Leere Zeilen überspringen
                if word:
                    self.stop_words.append(word)
        return self.stop_words
    
    def load_cross_encoder(self) -> CrossEncoder:
        if self.cross_encoder is not None:
            return self.cross_encoder
        local_model_path = f"{self.model_path}/{self.cross_encoder_model_name.replace('/', '-')}"
        
        if os.path.exists(local_model_path):
            # Modell lokal laden
            self.cross_encoder = CrossEncoder(local_model_path)
        else:
            # Modell herunterladen und speichern
            self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
            self.cross_encoder.save(local_model_path)
        
        return self.cross_encoder
    
    def sanitize_query(self, query: str) -> str:
        # Potenziell schädliche Patterns
        dangerous_patterns = [
            # SQL Injection
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE)\b)",
            r"(--|\;|\"|\')",
            # Script Injection
            r"(<script.*?>.*?</script>)",
            r"(<.*?on\w+\s*=)",
            # Command Injection
            r"(\||&|`|\$\()",
            # Path Traversal
            r"(\.\.\/|\.\.\\)",
        ]
        
        # Alle Pattern überprüfen
        cleaned_query = query
        for pattern in dangerous_patterns:
            cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
        return cleaned_query
    
    def remove_stop_words(self, query: str) -> str:
        if self.stop_words is None:
            self.load_stop_words()
        
        # Anfrage in Wörter aufteilen
        words = query.split()
        
        # Wörter filtern, die keine Stop-Words sind
        filtered_words = []
        for word in words:
            word_lower = word.lower()
            # Satzzeichen entfernen
            word_clean = re.sub(r'[^\w\s]', '', word_lower)
            # Prüfen ob Wort Stop-Word ist
            if word_clean not in self.stop_words:
                filtered_words.append(word)

        # Gefilterte Wörter zusammenfügen
        filtered_query = ' '.join(filtered_words)
        return filtered_query
    
    def search(self, query: str) -> list[dict]:
        if self.collection is None:
            self.connect_to_db()
        
        # Zeitmessung starten
        start_time = time.time()

        clean_query = self.sanitize_query(query)
        processed_query = self.remove_stop_words(clean_query)
        query_embedding = self.embedder.embed_query(processed_query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=self.top_k, include=['documents', 'metadatas', 'distances'])
        search_results = []
        
        # Ergebnisse durchalufen
        if results and 'documents' in results and results['documents']:
            documents = results['documents'][0]  # Erste (und einzige) Query
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]
            
            for i in range(len(documents)):
                similarity = 1 - distances[i]
                result = {
                    'id': ids[i],
                    'text': documents[i],
                    'url': metadatas[i].get('url', ''),
                    'title': metadatas[i].get('title', ''),
                    'section_header': metadatas[i].get('section_header', ''),  # Neu hinzugefügt
                    'distance': distances[i],
                    # Ähnlichkeit = 1 - Distanz bei Cosinus
                    'similarity': similarity
                }
                search_results.append(result)

                # Similarity-Score für Statistik speichern
                self.similarity_scores.append(similarity)
        
        # Retrieval-Zeit messen (vor Reranking)
        retrieval_time_ms = (time.time() - start_time) * 1000
        self.retrieval_times_ms.append(retrieval_time_ms)

        # Statistik aktualisieren
        self.query_count += 1
        self.total_chunks_retrieved += len(search_results)
        search_results = self.rerank_results(query, search_results)
        return search_results
    
    def rerank_results(self, query: str, results: list[dict]) -> list[dict]:
        # Zeitmessung starten
        start_time = time.time()
        
        if self.cross_encoder is None:
            self.load_cross_encoder()
        
        # Paare aus Query und Document erstellen
        pairs = []
        for result in results:
            text = result.get('text', '')
            pairs.append([query, text])
        
        # Scores mit Cross Encoder berechnen
        scores = self.cross_encoder.predict(pairs)
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
        
        # Nach Rerank-Score sortieren (höher ist besser)
        results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        # Nur top_k Ergenisse zurückgeben
        reranked_results = results[:self.reranking_top_k]

        # Reranking-Zeit messen
        reranking_time_ms = (time.time() - start_time) * 1000
        self.reranking_times_ms.append(reranking_time_ms)

        return reranked_results
    
    def get_statistics(self) -> dict:
        # Durchschnittliche Anzahl Chunks pro Query
        avg_chunks_per_query = 0.0
        avg_chunks_per_query = self.total_chunks_retrieved / self.query_count if self.query_count > 0 else 0.0
        
        # Durchschnittliche Retrieval-Zeit
        avg_retrieval_time_ms = 0.0
        if self.retrieval_times_ms:
            avg_retrieval_time_ms = sum(self.retrieval_times_ms) / len(self.retrieval_times_ms)
        
        # Durchschnittliche Reranking-Zeit
        avg_reranking_time_ms = 0.0
        if self.reranking_times_ms:
            avg_reranking_time_ms = sum(self.reranking_times_ms) / len(self.reranking_times_ms)
        
        # Similarity-Score Statistiken
        avg_similarity = 0.0
        min_similarity = 0.0
        max_similarity = 0.0
        
        if self.similarity_scores:
            avg_similarity = sum(self.similarity_scores) / len(self.similarity_scores)
            min_similarity = min(self.similarity_scores)
            max_similarity = max(self.similarity_scores)
        
        stats = {
            'retrieval_avg_chunks_per_query': round(avg_chunks_per_query, 4),
            'retrieval_avg_time_ms': round(avg_retrieval_time_ms, 4),
            'retrieval_avg_similarity': round(avg_similarity, 4),
            'retrieval_min_similarity': round(min_similarity, 4),
            'retrieval_max_similarity': round(max_similarity, 4),
            'retrieval_avg_reranking_time_ms': round(avg_reranking_time_ms, 4)
        }
        
        return stats
    
    def reset_statistics(self) -> None:
        self.query_count = 0
        self.total_chunks_retrieved = 0
        self.retrieval_times_ms = []
        self.reranking_times_ms = []
        self.similarity_scores = []

    def format_context(self, results: list[dict]) -> str:
        context_parts = []
        
        # Kontext aus Ergebnissen zusammenbauen
        for i, result in enumerate(results):
            source_info = f"[Quelle {i+1}]"
            if result.get('title'):
                source_info = source_info + f" {result['title']}"
            text = result.get('text', '')
            context_part = f"{source_info}\n{text}"
            context_parts.append(context_part)
        
        # Alle Teile verbinden
        context = "\n\n---\n\n".join(context_parts)
        return context
    
    def get_sources(self, results: list[dict]) -> list[str]:
        sources = []
        for result in results:
            url = result.get('url', '')
            if url and url not in sources:
                sources.append(url)
        return sources
