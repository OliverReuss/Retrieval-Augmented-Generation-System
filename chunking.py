from langchain_text_splitters import RecursiveCharacterTextSplitter
from web_scraping import load_documents
import config
import json
import os

class Chunker:
    
    def __init__(self) -> None:
        self.chunk_size = config.CHUNKING_CHUNK_SIZE
        self.chunk_overlap = config.CHUNKING_CHUNK_OVERLAP
        self.output_path = config.CHUNKING_OUTPUT_PATH
        self.input_path = config.WEB_SCRAPING_OUTPUT_PATH
        self.chunks = []
        
        # RecursiveCharacterTextSplitter erstellen
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # Trennzeichen (möglichst an diesen Stellen Trennen)
            separators=["\n\n", "\n", ". ", " ", ""],
            # Länge in Zeichen messen
            length_function=len,
        )
    
    def create_chunks(self) -> list[dict]:
        data = load_documents()
        self.chunks = []
        chunk_id = 0
        
        # Alle Dokumente verarbeiten
        for document in data:
            url = document.get('url', '')
            title = document.get('title', '')
            text = document.get('text', '')
            
            # Überspringen, wenn Text leer
            if not text or len(text.strip()) == 0:
                continue
            
            # Text chunken
            text_chunks = self.text_splitter.split_text(text)
            
            # Für jeden Chunk dict mit Metadaten erstellen
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    'chunk_id': chunk_id,
                    'url': url,
                    'title': title,
                    'text': chunk_text,
                    'chunk_index': i,
                    'total_chunks_in_document': len(text_chunks)
                }
                
                # Chunk zur Liste hinzufügen
                self.chunks.append(chunk)
                chunk_id = chunk_id + 1
        
        print(f"Chunking abgeschlossen: {len(self.chunks)} Chunks erstellt")
        return self.chunks

    def save_chunks(self) -> None:
        os.makedirs(os.path.dirname(self.output_path))
        with open(self.output_path, 'w', encoding='utf-8') as file:
            json.dump(self.chunks, file, ensure_ascii=False, indent=2)
    
    def process(self) -> None:
        if os.path.exists(self.output_path):
            return
        self.create_chunks()
        self.save_chunks()
        print(self.get_statistics())
    
    def get_statistics(self) -> dict:     
        # Berechne verschiedene Statistiken
        total_chunks = len(self.chunks)
        
        # Gesamtlänge
        chunk_lengths = []
        for chunk in self.chunks:
            text = chunk.get('text', '')
            chunk_lengths.append(len(text))
        
        # Durchschnittliche Chunk-Länge
        avg_length = (sum(chunk_lengths) / total_chunks)
        
        # Minimale und maximale Länge
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)
        
        # Eindeutige URLs
        unique_urls = set()
        for chunk in self.chunks:
            url = chunk.get('url', '')
            unique_urls.add(url)
        
        statistics = {
            'chunking_sources_total': unique_urls,
            'chunking_chunks_total': total_chunks,
            'chunking_average_chunk_length': avg_length,
            'chunking_min_chunk_length': min_length,
            'chunking_max_chunk_length': max_length,
        }
        return statistics

def load_chunks() -> list[dict]:
    with open(config.CHUNKING_OUTPUT_PATH, 'r', encoding='utf-8') as file:
        chunks = json.load(file)
    return chunks
