# embedding.py
# Dieses Modul ist für das Erstellen von Embeddings zuständig
# Es nutzt ein Sentence Transformer Modell für Dense Embeddings

import os
import json
import ssl

# Proxy-Einstellungen für HuggingFace (muss VOR den imports gesetzt werden)
# Verwende den gleichen Proxy wie für Web Scraping
import config
if config.SCRAPING_PROXY:
    os.environ['HTTP_PROXY'] = config.SCRAPING_PROXY
    os.environ['HTTPS_PROXY'] = config.SCRAPING_PROXY
    os.environ['http_proxy'] = config.SCRAPING_PROXY
    os.environ['https_proxy'] = config.SCRAPING_PROXY
    print(f"Proxy für HuggingFace gesetzt: {config.SCRAPING_PROXY}")

# SSL-Verifizierung deaktivieren (für Unternehmensumgebungen mit Proxy/Firewall)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['SSL_CERT_FILE'] = ''
os.environ['SSL_CERT_DIR'] = ''

# SSL-Kontext global deaktivieren
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# requests und urllib3 SSL deaktivieren
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# httpx SSL deaktivieren (wird von huggingface_hub verwendet)
import httpx
# Monkey-patch für httpx um SSL zu deaktivieren
_original_httpx_client_init = httpx.Client.__init__
def _patched_httpx_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    _original_httpx_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_httpx_client_init

# Sentence Transformers ist eine Library für Embeddings
from sentence_transformers import SentenceTransformer

# ChromaDB für die Vektordatenbank
import chromadb
from chromadb.config import Settings

# Wir importieren unsere Konfiguration
import config

# Wir importieren die Funktion zum Laden der Chunks
from chunking import load_chunks


class EmbeddingManager:
    """
    Diese Klasse ist für das Erstellen und Verwalten von Embeddings zuständig.
    Sie lädt das Modell und erstellt Embeddings für die Chunks.
    """
    
    def __init__(self):
        """
        Initialisiert den EmbeddingManager mit den Konfigurationswerten.
        """
        # Der Name des Embedding-Modells
        self.model_name = config.EMBEDDING_MODEL_NAME
        
        # Der lokale Pfad zum Speichern des Modells
        self.model_path = config.EMBEDDING_MODEL_PATH
        
        # Die Batch-Größe für das Embedding
        self.batch_size = config.EMBEDDING_BATCH_SIZE
        
        # Der Pfad zur ChromaDB
        self.chroma_db_path = config.CHROMA_DB_PATH
        
        # Der Name der Collection
        self.collection_name = config.CHROMA_COLLECTION_NAME
        
        # HNSW Parameter
        self.hnsw_m = config.HNSW_M
        
        # Das Modell wird erst bei Bedarf geladen
        self.model = None
        
        # Die ChromaDB Verbindung wird erst bei Bedarf erstellt
        self.chroma_client = None
        self.collection = None
        
        print(f"EmbeddingManager initialisiert:")
        print(f"  Modell: {self.model_name}")
        print(f"  Batch-Größe: {self.batch_size}")
    
    def load_model(self):
        """
        Lädt das Embedding-Modell.
        Wenn das Modell lokal gespeichert ist, wird es von dort geladen.
        Sonst wird es von HuggingFace heruntergeladen und lokal gespeichert.
        """
        # Prüfen ob das Modell bereits geladen ist
        if self.model is None:
            # Lokales Modellverzeichnis
            local_model_path = f"models/{self.model_name.replace('/', '_')}"
            
            # Prüfen ob das Modell lokal gespeichert ist
            if os.path.exists(local_model_path):
                print(f"Lade Modell von lokalem Pfad: {local_model_path}")
                self.model = SentenceTransformer(local_model_path)
            else:
                # Modell von HuggingFace herunterladen
                print(f"Lade Modell von HuggingFace: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                
                # Modell lokal speichern für später
                self.model.save(local_model_path)
                print(f"Modell gespeichert in: {local_model_path}")
            
            print("Modell erfolgreich geladen.")
        
        return self.model
    
    def save_model(self):
        """
        Speichert das Modell lokal.
        """
        # Prüfen ob wir ein Modell haben
        if self.model is None:
            print("Kein Modell zum Speichern vorhanden!")
            return
        
        # Das Verzeichnis erstellen falls nötig
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Verzeichnis erstellt: {model_dir}")
        
        # Das Modell speichern
        self.model.save(self.model_path)
        print(f"Modell gespeichert in: {self.model_path}")
    
    def create_embeddings(self, texts):
        """
        Erstellt Embeddings für eine Liste von Texten.
        Nutzt Batch-Verarbeitung für bessere Performance.
        """
        # Zuerst das Modell laden falls nötig
        if self.model is None:
            self.load_model()
        
        # Prüfen ob wir Texte haben
        if not texts:
            print("Keine Texte zum Embedden vorhanden!")
            return []
        
        print(f"Erstelle Embeddings für {len(texts)} Texte...")
        print(f"Batch-Größe: {self.batch_size}")
        
        # Embeddings in Batches erstellen
        # Das ist schneller als jeden Text einzeln zu verarbeiten
        all_embeddings = []
        
        # Wir teilen die Texte in Batches auf
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(texts), self.batch_size):
            # Den aktuellen Batch extrahieren
            batch = texts[i:i + self.batch_size]
            
            # Die Batch-Nummer für die Fortschrittsanzeige
            batch_num = (i // self.batch_size) + 1
            print(f"  Verarbeite Batch {batch_num}/{total_batches}...")
            
            # Embeddings für diesen Batch erstellen
            # convert_to_numpy=True gibt uns NumPy Arrays zurück
            batch_embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Die Embeddings zur Gesamtliste hinzufügen
            # Wir konvertieren zu Listen für JSON-Kompatibilität
            for embedding in batch_embeddings:
                all_embeddings.append(embedding.tolist())
        
        print(f"Embeddings erstellt: {len(all_embeddings)}")
        return all_embeddings
    
    def setup_chroma_db(self):
        """
        Richtet die ChromaDB Vektordatenbank ein.
        Erstellt den Client und die Collection.
        """
        # Das Verzeichnis für ChromaDB erstellen falls nötig
        if not os.path.exists(self.chroma_db_path):
            os.makedirs(self.chroma_db_path)
            print(f"ChromaDB Verzeichnis erstellt: {self.chroma_db_path}")
        
        # ChromaDB Client erstellen mit persistentem Speicher
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_db_path
        )
        
        print(f"ChromaDB Client erstellt: {self.chroma_db_path}")
        
        # Collection erstellen oder laden
        # Wir nutzen HNSW mit Cosinus-Ähnlichkeit
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                # HNSW Parameter setzen
                "hnsw:M": self.hnsw_m,
                # Cosinus-Ähnlichkeit als Distanzmetrik
                "hnsw:space": "cosine"
            }
        )
        
        print(f"Collection erstellt/geladen: {self.collection_name}")
        print(f"HNSW M Parameter: {self.hnsw_m}")
        
        return self.collection
    
    def store_embeddings(self, chunks, embeddings):
        """
        Speichert die Embeddings in der ChromaDB.
        """
        # Zuerst die Datenbank einrichten falls nötig
        if self.collection is None:
            self.setup_chroma_db()
        
        # Prüfen ob wir Chunks und Embeddings haben
        if not chunks or not embeddings:
            print("Keine Chunks oder Embeddings zum Speichern!")
            return
        
        # Prüfen ob die Anzahl übereinstimmt
        if len(chunks) != len(embeddings):
            print(f"Fehler: Anzahl der Chunks ({len(chunks)}) und Embeddings ({len(embeddings)}) stimmt nicht überein!")
            return
        
        print(f"Speichere {len(chunks)} Embeddings in ChromaDB...")
        
        # Die Daten für ChromaDB vorbereiten
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Die ID ist einfach der Index als String
            chunk_id = str(chunk.get('chunk_id', i))
            ids.append(chunk_id)
            
            # Der Text des Chunks
            text = chunk.get('text', '')
            documents.append(text)
            
            # Die Metadaten (URL, Titel, etc.)
            metadata = {
                'url': chunk.get('url', ''),
                'title': chunk.get('title', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'total_chunks': chunk.get('total_chunks_in_document', 1)
            }
            metadatas.append(metadata)
        
        # Die Daten in Batches zu ChromaDB hinzufügen
        # ChromaDB hat ein Limit für die Batch-Größe
        batch_size = 500  # ChromaDB empfiehlt max 5000
        
        for i in range(0, len(ids), batch_size):
            # Den Batch extrahieren
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            # Fortschritt anzeigen
            batch_num = (i // batch_size) + 1
            total_batches = (len(ids) + batch_size - 1) // batch_size
            print(f"  Speichere Batch {batch_num}/{total_batches}...")
            
            # Zur Collection hinzufügen
            # upsert fügt hinzu oder aktualisiert wenn ID existiert
            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
        
        print("Embeddings erfolgreich in ChromaDB gespeichert!")
        
        # Anzahl der Einträge in der Collection
        count = self.collection.count()
        print(f"Gesamtzahl Einträge in Collection: {count}")
    
    def process(self):
        """
        Führt den kompletten Embedding-Prozess aus.
        Lädt die Chunks, erstellt Embeddings und speichert sie.
        """
        # Chunks laden
        chunks = load_chunks()
        
        if not chunks:
            print("Keine Chunks gefunden. Abbruch.")
            return
        
        # Modell laden
        self.load_model()
        
        # Die Texte aus den Chunks extrahieren
        texts = []
        for chunk in chunks:
            text = chunk.get('text', '')
            texts.append(text)
        
        # Embeddings erstellen
        embeddings = self.create_embeddings(texts)
        
        # In ChromaDB speichern
        self.store_embeddings(chunks, embeddings)
        
        print("\nEmbedding-Prozess abgeschlossen!")
    
    def get_embedding_for_query(self, query):
        """
        Erstellt ein Embedding für eine einzelne Suchanfrage.
        """
        # Modell laden falls nötig
        if self.model is None:
            self.load_model()
        
        # Embedding erstellen
        embedding = self.model.encode(query, convert_to_numpy=True)
        
        return embedding.tolist()


# Dieser Code wird nur ausgeführt, wenn die Datei direkt gestartet wird
if __name__ == "__main__":
    # Zum Testen können wir das Embedding direkt starten
    embedder = EmbeddingManager()
    embedder.process()
