from chunking import load_chunks
from sentence_transformers import SentenceTransformer
import chromadb
import config
import os

class Embedder:
    
    def __init__(self) -> None:
        self.embedding_model = config.EMBEDDING_MODEL
        self.batch_size = config.EMBEDDING_BATCH_SIZE
        self.model_path = config.MODELS_DIRECTORY
        self.db_path = config.RETRIEVAL_DB_PATH
        self.collection_name = config.RETRIEVAL_COLLECTION_NAME
        self.hnsw_space = config.RETRIEVAL_HNSW_SPACE
        self.hnsw_m = config.RETRIEVAL_HNSW_M
        # Datnebank und Modell erst bei Bedarf laden
        self.model = None
        self.db_client = None
        self.collection = None
        # Statistiken
        self.embedding_dimension = 0

        # Automatisch starten
        self.process()
    
    def load_model(self) -> SentenceTransformer:
        if self.model is None:
            safe_model_name = self.embedding_model.replace('/', '_')
            local_model_path = os.path.join(self.model_path, safe_model_name)
            if os.path.exists(local_model_path):
                self.model = SentenceTransformer(local_model_path)
            else:
                self.model = SentenceTransformer(self.embedding_model, trust_remote_code=True)
                os.makedirs(local_model_path, exist_ok=True)
                self.model.save(local_model_path)
        return self.model
    
    def create_embeddings(self, texts: list[str]) -> list[list]:        
        # Embeddings in Batches erstellen (schneller)
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(texts), self.batch_size):
            # Aktueller Batch
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            print(f"Emebdding | Batch {batch_num}/{total_batches}")
            
            # Embeddings für Batch erstellen
            batch_embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                # Gibt NumPy-Array zurück
                convert_to_numpy=True,
                # Normaliserung für Cosine-Similarity
                normalize_embeddings=True
            )
            
            for embedding in batch_embeddings:
                all_embeddings.append(embedding.tolist())
        
        # Embedding-Dimension ermitteln (aus erstem Embedding)
        if all_embeddings:
            self.embedding_dimension = len(all_embeddings[0])
        
        return all_embeddings
    
    def setup_db(self) -> chromadb.Collection:
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self.db_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.db_client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:M": self.hnsw_m,"hnsw:space": self.hnsw_space})
        return self.collection
    
    def get_statistics(self) -> dict:
        # Collection laden, falls nicht verbunden
        if self.collection is None:
            self.setup_db()
        
        # Embedding-Dimension ermitteln (falls nicht bekannt)
        if self.embedding_dimension == 0 and self.collection is not None:
            # Ein Embedding aus der Datenbank holen
            sample = self.collection.peek(limit=1)
            if sample and sample.get('embeddings') is not None and len(sample['embeddings']) > 0:
                self.embedding_dimension = len(sample['embeddings'][0])
        
        statistics = {
            'embedding_dimension': self.embedding_dimension
        }
        return statistics

    def save_embeddings(self, chunks: list[dict], embeddings: list[list]) -> None:
        if self.collection is None:
            self.setup_db()
        ids = []
        documents = []
        metadatas = []
        
        # Jeden Chunk vorbereiten
        for i, chunk in enumerate(chunks):
            chunk_id = str(chunk.get('chunk_id', i))
            ids.append(chunk_id)
            text = chunk.get('text', '')
            documents.append(text)
            metadata = {
                'url': chunk.get('url', ''),
                'title': chunk.get('page_title', ''),
                'section_header': chunk.get('section_header', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'total_chunks': chunk.get('total_chunks_in_document', 1)
            }
            metadatas.append(metadata)
        
        # Daten in Batches zu Datenbank hinzufügen
        batch_size = 500
        
        for i in range(0, len(ids), batch_size):
            # Batch extrahieren
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            # Fortschrittsanzeige
            batch_num = (i // batch_size) + 1
            total_batches = (len(ids) + batch_size - 1) // batch_size
            
            # Zur Collection hinzufügen
            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

    def process(self) -> None:
        if os.path.exists(self.db_path):
            return
        
        print("\nℹ️ Starte Embedding")
        
        chunks = load_chunks()
        self.load_model()
        texts = []
        for chunk in chunks:
            text = chunk.get('text', '')
            texts.append(text)
        embeddings = self.create_embeddings(texts)
        self.save_embeddings(chunks, embeddings)

        print(f"\nℹ️ Embedding abgeschlossen. Erstellte Emebddings: {len(embeddings)}.")
    
    def embed_query(self, query: str) -> list:
        if self.model is None:
            self.load_model()
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()
