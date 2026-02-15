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
            # Trennzeichen am Ende des vorherigen Chunks behalten
            keep_separator='end',
        )

        # Automatisch starten
        self.process()
    
    def create_chunks(self) -> list[dict]:
        data = load_documents()
        self.chunks = []
        chunk_id = 0
        
        # Alle Dokumente verarbeiten
        for document in data:
            url = document.get('url', '')
            page_title = document.get('page_title', '')
            section_header = document.get('section_header', '')
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
                    'page_title': page_title,
                    'section_header': section_header,
                    'text': chunk_text,
                    'chunk_index': i,
                    'total_chunks_in_document': len(text_chunks)
                }
                
                # Chunk zur Liste hinzufügen
                self.chunks.append(chunk)
                chunk_id = chunk_id + 1

        return self.chunks

    def save_chunks(self) -> None:
        dir_name = os.path.dirname(self.output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as file:
            json.dump(self.chunks, file, ensure_ascii=False, indent=2)
    
    def process(self) -> None:
        if os.path.exists(self.output_path):
            return
        
        print("\nℹ️ Starte Chunking")

        self.create_chunks()
        self.save_chunks()
        print(f"\nℹ️ Chunking abgeschlossen. Erstellte Chunks: {len(self.chunks)}.")
    
    def get_statistics(self) -> dict:     
         # Chunks laden, falls nicht im Speicher
        if not self.chunks and os.path.exists(self.output_path):
            self.chunks = load_chunks()
        
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

        # Überlappungsrate
        overlap_percentage = (self.chunk_overlap / self.chunk_size) * 100
        
        statistics = {
            'chunking_chunks_total': total_chunks,
            'chunking_average_chunk_length': round(avg_length, 2),
            'chunking_min_chunk_length': round(min_length, 2),
            'chunking_max_chunk_length': round(max_length, 2),
            'chunking_overlap_percentage': round(overlap_percentage, 2)
        }
        return statistics

def load_chunks() -> list[dict]:
    with open(config.CHUNKING_OUTPUT_PATH, 'r', encoding='utf-8') as file:
        chunks = json.load(file)
    return chunks
