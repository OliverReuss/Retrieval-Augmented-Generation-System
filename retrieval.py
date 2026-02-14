from embedding import Embedder
from sentence_transformers import CrossEncoder
import chromadb
import config
import re

class Retriever:

    def __init__(self) -> None:
        self.db_path = config.RETRIEVAL_DB_PATH
        self.collection_name = config.RETRIEVAL_COLLECTION_NAME
        self.top_k = config.RETRIEVAL_TOP_K
        self.reranking_top_k = config.RETRIEVAL_RERANKING_TOP_K
        self.stop_words_path = config.RETRIEVAL_STOP_WORDS_PATH
        self.cross_encoder_model_name = config.RETRIEVAL_CROSS_ENCODER_MODEL
        self.use_reranking = config.RETRIEVAL_USE_RERANKING
        self.chroma_client = None
        self.collection = None
        self.embedder = Embedder()
        self.cross_encoder = None
        self.stop_words = None
    
    def connect_to_db(self) -> chromadb.Collection:
        if self.collection is not None:
            return self.collection
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_collection(name=self.collection_name)
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
        self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
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
                result = {
                    'id': ids[i],
                    'text': documents[i],
                    'url': metadatas[i].get('url', ''),
                    'title': metadatas[i].get('title', ''),
                    'distance': distances[i],
                    # Ähnlichkeit = 1 - Distanz bei Cosinus
                    'similarity': 1 - distances[i]
                }
                search_results.append(result)
        
        if self.use_reranking:
            search_results = self.rerank_results(query, search_results)
        return search_results
    
    def rerank_results(self, query: str, results: list[dict]) -> list[dict]:
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
        return reranked_results
    
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
