# retrieval.py
# Dieses Modul ist für das Retrieval (Suche) in der Vektordatenbank zuständig
# Es sucht nach relevanten Dokumenten basierend auf der Nutzeranfrage

import os
import re

# ChromaDB für die Vektordatenbank
import chromadb

# Cross Encoder für das Reranking
from sentence_transformers import CrossEncoder

# Wir importieren unsere Konfiguration
import config

# Wir importieren den EmbeddingManager für die Query-Embeddings
from embedding import EmbeddingManager


class RetrievalManager:
    """
    Diese Klasse ist für das Retrieval zuständig.
    Sie sucht in der Vektordatenbank nach relevanten Dokumenten.
    """
    
    def __init__(self):
        """
        Initialisiert den RetrievalManager mit den Konfigurationswerten.
        """
        # Der Pfad zur ChromaDB
        self.chroma_db_path = config.CHROMA_DB_PATH
        
        # Der Name der Collection
        self.collection_name = config.CHROMA_COLLECTION_NAME
        
        # Anzahl der Ergebnisse vor dem Reranking
        self.top_k = config.RETRIEVAL_TOP_K
        
        # Anzahl der Ergebnisse nach dem Reranking
        self.reranking_top_k = config.RERANKING_TOP_K
        
        # Pfad zur Stop-Words Datei
        self.stop_words_path = config.STOP_WORDS_PATH
        
        # Cross Encoder Modell für Reranking
        self.cross_encoder_model_name = config.CROSS_ENCODER_MODEL
        
        # Reranking aktiviert/deaktiviert
        self.use_reranking = config.USE_RERANKING
        
        # Die ChromaDB Verbindung
        self.chroma_client = None
        self.collection = None
        
        # Der EmbeddingManager für Query-Embeddings
        self.embedding_manager = EmbeddingManager()
        
        # Der Cross Encoder (wird bei Bedarf geladen)
        self.cross_encoder = None
        
        # Die Stop-Words Liste (wird bei Bedarf geladen)
        self.stop_words = None
        
        print(f"RetrievalManager initialisiert:")
        print(f"  Top-K vor Reranking: {self.top_k}")
        print(f"  Top-K nach Reranking: {self.reranking_top_k}")
        print(f"  Reranking aktiviert: {self.use_reranking}")
    
    def connect_to_db(self):
        """
        Verbindet sich mit der ChromaDB Datenbank.
        """
        # Prüfen ob bereits verbunden
        if self.collection is not None:
            return self.collection
        
        # Prüfen ob die Datenbank existiert
        if not os.path.exists(self.chroma_db_path):
            print(f"Fehler: ChromaDB nicht gefunden: {self.chroma_db_path}")
            print("Bitte zuerst das Embedding ausführen.")
            return None
        
        # ChromaDB Client erstellen
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_db_path
        )
        
        # Die Collection laden
        self.collection = self.chroma_client.get_collection(
            name=self.collection_name
        )
        
        print(f"Mit ChromaDB verbunden: {self.collection_name}")
        print(f"Anzahl Dokumente: {self.collection.count()}")
        
        return self.collection
    
    def load_stop_words(self):
        """
        Lädt die Stop-Words aus der Datei.
        Stop-Words sind Füllwörter, die aus der Suche entfernt werden.
        """
        # Prüfen ob bereits geladen
        if self.stop_words is not None:
            return self.stop_words
        
        # Die Stop-Words Liste initialisieren
        self.stop_words = []
        
        # Prüfen ob die Datei existiert
        if not os.path.exists(self.stop_words_path):
            print(f"Warnung: Stop-Words Datei nicht gefunden: {self.stop_words_path}")
            print("Keine Stop-Words werden entfernt.")
            return self.stop_words
        
        # Die Datei laden
        with open(self.stop_words_path, 'r', encoding='utf-8') as file:
            # Jede Zeile ist ein Stop-Word
            for line in file:
                word = line.strip().lower()
                if word:  # Leere Zeilen überspringen
                    self.stop_words.append(word)
        
        print(f"Stop-Words geladen: {len(self.stop_words)} Wörter")
        return self.stop_words
    
    def load_cross_encoder(self):
        """
        Lädt den Cross Encoder für das Reranking.
        """
        # Prüfen ob bereits geladen
        if self.cross_encoder is not None:
            return self.cross_encoder
        
        print(f"Lade Cross Encoder: {self.cross_encoder_model_name}")
        self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
        print("Cross Encoder geladen.")
        
        return self.cross_encoder
    
    def sanitize_query(self, query):
        """
        Bereinigt die Nutzeranfrage.
        Entfernt potenziell schädlichen Input mit Regex.
        """
        # Die ursprüngliche Anfrage speichern
        original_query = query
        
        # Muster für potenziell schädlichen Input
        # Diese Patterns erkennen typische Injection-Versuche
        dangerous_patterns = [
            # SQL Injection Patterns
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE)\b)",
            r"(--|\;|\"|\')",
            
            # Script Injection Patterns
            r"(<script.*?>.*?</script>)",
            r"(<.*?on\w+\s*=)",
            
            # Command Injection Patterns
            r"(\||&|`|\$\()",
            
            # Path Traversal Patterns
            r"(\.\.\/|\.\.\\)",
        ]
        
        # Jeden Pattern prüfen und entfernen
        cleaned_query = query
        for pattern in dangerous_patterns:
            # re.IGNORECASE macht die Suche case-insensitive
            cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
        
        # Prüfen ob etwas entfernt wurde
        if cleaned_query != original_query:
            print(f"Warnung: Potenziell schädlicher Input entfernt!")
            print(f"  Original: {original_query}")
            print(f"  Bereinigt: {cleaned_query}")
        
        return cleaned_query
    
    def remove_stop_words(self, query):
        """
        Entfernt Stop-Words aus der Anfrage.
        """
        # Stop-Words laden falls nötig
        if self.stop_words is None:
            self.load_stop_words()
        
        # Wenn keine Stop-Words, direkt zurückgeben
        if not self.stop_words:
            return query
        
        # Die Anfrage in Wörter aufteilen
        words = query.split()
        
        # Wörter filtern, die keine Stop-Words sind
        filtered_words = []
        for word in words:
            # Das Wort in Kleinbuchstaben für den Vergleich
            word_lower = word.lower()
            
            # Satzzeichen entfernen für den Vergleich
            word_clean = re.sub(r'[^\w\s]', '', word_lower)
            
            # Prüfen ob das Wort ein Stop-Word ist
            if word_clean not in self.stop_words:
                filtered_words.append(word)
        
        # Die gefilterten Wörter wieder zusammenfügen
        filtered_query = ' '.join(filtered_words)
        
        # Info ausgeben wenn Wörter entfernt wurden
        if len(filtered_words) < len(words):
            removed_count = len(words) - len(filtered_words)
            print(f"Stop-Words entfernt: {removed_count} Wörter")
        
        return filtered_query
    
    def search(self, query, use_reranking=None):
        """
        Führt eine Suche in der Vektordatenbank durch.
        
        Parameter:
            query: Die Suchanfrage des Nutzers
            use_reranking: Ob Reranking mit Cross Encoder verwendet werden soll
                           (None = Config-Einstellung verwenden)
        
        Rückgabe:
            Liste von Suchergebnissen mit Text und Metadaten
        """
        # Reranking-Einstellung: Parameter überschreibt Config
        if use_reranking is None:
            use_reranking = self.use_reranking
        # Mit der Datenbank verbinden
        if self.collection is None:
            self.connect_to_db()
        
        if self.collection is None:
            print("Fehler: Keine Verbindung zur Datenbank!")
            return []
        
        # Die Anfrage bereinigen
        clean_query = self.sanitize_query(query)
        
        # Stop-Words entfernen
        processed_query = self.remove_stop_words(clean_query)
        
        # Wenn die Anfrage nach der Bereinigung leer ist, die originale verwenden
        if not processed_query.strip():
            print("Warnung: Anfrage nach Bereinigung leer, verwende Original")
            processed_query = clean_query
        
        print(f"Suche nach: '{processed_query}'")
        
        # Ein Embedding für die Anfrage erstellen
        query_embedding = self.embedding_manager.get_embedding_for_query(processed_query)
        
        # In ChromaDB suchen
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Die Ergebnisse in ein lesbares Format umwandeln
        search_results = []
        
        # Die Ergebnisse durchgehen
        # ChromaDB gibt die Ergebnisse in Listen zurück
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
                    # Ähnlichkeit = 1 - Distanz (bei Cosinus)
                    'similarity': 1 - distances[i]
                }
                search_results.append(result)
        
        print(f"Gefunden: {len(search_results)} Ergebnisse")
        
        # Reranking mit Cross Encoder
        if use_reranking and len(search_results) > 0:
            search_results = self.rerank_results(query, search_results)
        
        return search_results
    
    def rerank_results(self, query, results):
        """
        Führt ein Reranking der Suchergebnisse mit einem Cross Encoder durch.
        Der Cross Encoder bewertet die Relevanz von Query-Document Paaren.
        """
        # Cross Encoder laden falls nötig
        if self.cross_encoder is None:
            self.load_cross_encoder()
        
        print(f"Reranking von {len(results)} Ergebnissen...")
        
        # Paare aus Query und Document erstellen
        pairs = []
        for result in results:
            text = result.get('text', '')
            pairs.append([query, text])
        
        # Scores mit dem Cross Encoder berechnen
        scores = self.cross_encoder.predict(pairs)
        
        # Die Scores zu den Ergebnissen hinzufügen
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
        
        # Nach Rerank-Score sortieren (höher ist besser)
        results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        # Nur die Top-K nach Reranking zurückgeben
        reranked_results = results[:self.reranking_top_k]
        
        print(f"Nach Reranking: {len(reranked_results)} Ergebnisse")
        
        return reranked_results
    
    def format_context(self, results):
        """
        Formatiert die Suchergebnisse als Kontext-String für das LLM.
        """
        if not results:
            return "Keine relevanten Informationen gefunden."
        
        # Den Kontext aus den Ergebnissen zusammenbauen
        context_parts = []
        
        for i, result in enumerate(results):
            # Nummer und Quelle
            source_info = f"[Quelle {i+1}]"
            if result.get('title'):
                source_info = source_info + f" {result['title']}"
            
            # Der Text des Chunks
            text = result.get('text', '')
            
            # Zusammenfügen
            context_part = f"{source_info}\n{text}"
            context_parts.append(context_part)
        
        # Alle Teile mit Trennlinien verbinden
        context = "\n\n---\n\n".join(context_parts)
        
        return context
    
    def get_sources(self, results):
        """
        Extrahiert die Quellen (URLs) aus den Suchergebnissen.
        """
        sources = []
        
        for result in results:
            url = result.get('url', '')
            if url and url not in sources:
                sources.append(url)
        
        return sources


# Dieser Code wird nur ausgeführt, wenn die Datei direkt gestartet wird
if __name__ == "__main__":
    # Zum Testen können wir eine Suche durchführen
    retriever = RetrievalManager()
    
    # Eine Test-Anfrage
    test_query = "Welche Leistungen bietet die AOK?"
    
    # Suche durchführen
    results = retriever.search(test_query)
    
    # Ergebnisse ausgeben
    print("\nSuchergebnisse:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result.get('title', 'Kein Titel')}")
        print(f"   URL: {result.get('url', '')}")
        print(f"   Ähnlichkeit: {result.get('similarity', 0):.4f}")
        print(f"   Rerank Score: {result.get('rerank_score', 0):.4f}")
        # Nur die ersten 200 Zeichen des Textes
        text = result.get('text', '')[:200]
        print(f"   Text: {text}...")
