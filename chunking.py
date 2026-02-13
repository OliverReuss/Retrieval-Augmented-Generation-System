# chunking.py
# Dieses Modul ist für das Aufteilen der Texte in kleinere Chunks zuständig
# Es nutzt LangChain's RecursiveCharacterTextSplitter

import json
import os

# LangChain bietet einen guten Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Wir importieren unsere Konfiguration
import config

# Wir importieren die Funktion zum Laden der gescrapten Daten
from web_scraping import load_scraped_data


class ChunkingManager:
    """
    Diese Klasse ist für das Chunking der Texte zuständig.
    Sie teilt lange Texte in kleinere Stücke auf.
    """
    
    def __init__(self):
        """
        Initialisiert den ChunkingManager mit den Konfigurationswerten.
        """
        # Die Chunk-Größe aus der Konfiguration
        self.chunk_size = config.CHUNKING_CHUNK_SIZE
        
        # Der Overlap zwischen Chunks
        self.chunk_overlap = config.CHUNKING_CHUNK_OVERLAP
        
        # Der Ausgabepfad für die Chunks
        self.output_path = config.CHUNKING_OUTPUT_PATH
        
        # Der Eingabepfad (die gescrapten Daten)
        self.input_path = config.SCRAPING_OUTPUT_PATH
        
        # Hier werden die erstellten Chunks gespeichert
        self.chunks = []
        
        # Wir erstellen den Text Splitter von LangChain
        # RecursiveCharacterTextSplitter ist gut für natürlichen Text
        self.text_splitter = RecursiveCharacterTextSplitter(
            # Die maximale Größe eines Chunks
            chunk_size=self.chunk_size,
            
            # Die Überlappung zwischen Chunks (für Kontext)
            chunk_overlap=self.chunk_overlap,
            
            # Die Trennzeichen, die der Splitter verwendet
            # Er versucht zuerst an diesen Stellen zu trennen
            separators=["\n\n", "\n", ". ", " ", ""],
            
            # Wir messen die Länge in Zeichen
            length_function=len,
        )
        
        print(f"ChunkingManager initialisiert:")
        print(f"  Chunk-Größe: {self.chunk_size} Zeichen")
        print(f"  Chunk-Overlap: {self.chunk_overlap} Zeichen")
    
    def load_data(self):
        """
        Lädt die gescrapten Daten aus der JSON-Datei.
        """
        # Die Hilfsfunktion aus web_scraping nutzen
        data = load_scraped_data()
        return data
    
    def create_chunks(self, data=None):
        """
        Erstellt Chunks aus den gescrapten Daten.
        Jeder Chunk behält die Metadaten (URL, Titel) bei.
        """
        # Wenn keine Daten übergeben wurden, laden wir sie
        if data is None:
            data = self.load_data()
        
        # Prüfen ob wir Daten haben
        if not data:
            print("Keine Daten zum Chunken vorhanden!")
            return []
        
        print(f"Starte Chunking von {len(data)} Dokumenten...")
        
        # Die Liste für alle Chunks leeren
        self.chunks = []
        
        # Zähler für die Chunk-IDs
        chunk_id = 0
        
        # Wir gehen durch jedes gescrapte Dokument
        for document in data:
            # Die URL und den Titel extrahieren
            url = document.get('url', '')
            title = document.get('title', '')
            text = document.get('text', '')
            
            # Wenn der Text leer ist, überspringen wir dieses Dokument
            if not text or len(text.strip()) == 0:
                continue
            
            # Den Text in Chunks aufteilen
            # Der Text Splitter gibt uns eine Liste von Strings zurück
            text_chunks = self.text_splitter.split_text(text)
            
            # Für jeden Chunk erstellen wir ein Dictionary mit Metadaten
            for i, chunk_text in enumerate(text_chunks):
                # Das Chunk-Dictionary erstellen
                chunk = {
                    # Eine eindeutige ID für jeden Chunk
                    'chunk_id': chunk_id,
                    
                    # Die URL der Ursprungsseite
                    'url': url,
                    
                    # Der Titel der Ursprungsseite
                    'title': title,
                    
                    # Der eigentliche Text des Chunks
                    'text': chunk_text,
                    
                    # Die Nummer dieses Chunks innerhalb des Dokuments
                    'chunk_index': i,
                    
                    # Die Gesamtzahl der Chunks aus diesem Dokument
                    'total_chunks_in_document': len(text_chunks)
                }
                
                # Den Chunk zur Liste hinzufügen
                self.chunks.append(chunk)
                
                # Die Chunk-ID erhöhen
                chunk_id = chunk_id + 1
        
        print(f"Chunking abgeschlossen: {len(self.chunks)} Chunks erstellt")
        
        return self.chunks
    
    def save_chunks(self):
        """
        Speichert die Chunks als JSON-Datei.
        """
        # Prüfen ob wir Chunks haben
        if not self.chunks:
            print("Keine Chunks zum Speichern vorhanden!")
            return
        
        # Das Ausgabeverzeichnis erstellen falls nötig
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Verzeichnis erstellt: {output_dir}")
        
        # Die Chunks als JSON speichern
        with open(self.output_path, 'w', encoding='utf-8') as file:
            json.dump(self.chunks, file, ensure_ascii=False, indent=2)
        
        print(f"Chunks gespeichert in: {self.output_path}")
    
    def process(self):
        """
        Führt den kompletten Chunking-Prozess aus.
        Lädt die Daten, erstellt Chunks und speichert sie.
        """
        # Daten laden
        data = self.load_data()
        
        # Chunks erstellen
        self.create_chunks(data)
        
        # Chunks speichern
        self.save_chunks()
        
        # Statistiken ausgeben
        self.print_statistics()
        
        return self.chunks
    
    def print_statistics(self):
        """
        Gibt Statistiken über die erstellten Chunks aus.
        """
        if not self.chunks:
            print("Keine Chunks vorhanden für Statistiken.")
            return
        
        # Berechne verschiedene Statistiken
        total_chunks = len(self.chunks)
        
        # Die Längen aller Chunks sammeln
        chunk_lengths = []
        for chunk in self.chunks:
            text = chunk.get('text', '')
            chunk_lengths.append(len(text))
        
        # Durchschnittliche Chunk-Länge
        avg_length = sum(chunk_lengths) / total_chunks
        
        # Minimale und maximale Länge
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)
        
        # Anzahl der eindeutigen URLs (Quell-Dokumente)
        unique_urls = set()
        for chunk in self.chunks:
            url = chunk.get('url', '')
            unique_urls.add(url)
        
        print("\nChunking Statistiken:")
        print(f"  Gesamtzahl Chunks: {total_chunks}")
        print(f"  Anzahl Quell-Dokumente: {len(unique_urls)}")
        print(f"  Durchschnittliche Chunk-Länge: {avg_length:.1f} Zeichen")
        print(f"  Minimale Chunk-Länge: {min_length} Zeichen")
        print(f"  Maximale Chunk-Länge: {max_length} Zeichen")


def load_chunks():
    """
    Hilfsfunktion um die Chunks aus der JSON-Datei zu laden.
    Wird von anderen Modulen genutzt.
    """
    # Pfad aus der Konfiguration
    file_path = config.CHUNKING_OUTPUT_PATH
    
    # Prüfen ob die Datei existiert
    if not os.path.exists(file_path):
        print(f"Fehler: Datei nicht gefunden: {file_path}")
        print("Bitte zuerst das Chunking ausführen.")
        return []
    
    # Datei öffnen und JSON laden
    with open(file_path, 'r', encoding='utf-8') as file:
        chunks = json.load(file)
    
    print(f"Chunks geladen: {len(chunks)} Einträge")
    return chunks


# Dieser Code wird nur ausgeführt, wenn die Datei direkt gestartet wird
if __name__ == "__main__":
    # Zum Testen können wir das Chunking direkt starten
    chunker = ChunkingManager()
    chunker.process()
