# main.py
# Dies ist die Hauptdatei der RAG-Pipeline für das AOK-Projekt
# Von hier aus kann die gesamte Pipeline gesteuert werden

import os
import sys

# Wir importieren unsere Module
from web_scraping import WebScrapingManager
from chunking import ChunkingManager
from embedding import EmbeddingManager
from retrieval import RetrievalManager
from generation import GenerationManager, run_rag_query
from evaluation import EvaluationManager, create_sample_test_cases

# Wir importieren die Konfiguration
import config


def print_header():
    """
    Gibt den Header der Anwendung aus.
    """
    print("\n" + "="*60)
    print("  AOK RAG-Pipeline")
    print("  Retrieval-Augmented Generation für AOK Informationen")
    print("="*60)


def print_menu():
    """
    Gibt das Hauptmenü aus.
    """
    print("\nWas möchten Sie tun?")
    print("-" * 40)
    print("1. Web Scraping starten (Daten von AOK-Webseite holen)")
    print("2. Chunking durchführen (Texte in Stücke teilen)")
    print("3. Embedding erstellen (Texte in Vektoren umwandeln)")
    print("4. Komplette Pipeline ausführen (1-3 nacheinander)")
    print("5. Frage stellen (RAG-Anfrage)")
    print("6. Evaluation durchführen")
    print("7. Beispiel-Testfälle erstellen")
    print("8. Beenden")
    print("-" * 40)


def ensure_directories():
    """
    Stellt sicher, dass alle benötigten Verzeichnisse existieren.
    """
    # Die Verzeichnisse aus der Konfiguration
    directories = [
        config.DATA_DIR,
        config.MODELS_DIR,
    ]
    
    # Jedes Verzeichnis erstellen falls nötig
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Verzeichnis erstellt: {directory}")


def run_web_scraping():
    """
    Führt das Web Scraping durch.
    """
    print("\n--- WEB SCRAPING ---")
    print(f"Start-URL: {config.SCRAPING_START_URL}")
    print(f"Maximale Tiefe: {config.SCRAPING_MAX_DEPTH}")
    
    # Bestätigung vom Nutzer holen
    print("\nMöchten Sie das Web Scraping starten? (j/n)")
    answer = input("> ").strip().lower()
    
    if answer == 'j' or answer == 'ja':
        # Den WebScrapingManager erstellen und starten
        scraper = WebScrapingManager()
        stats = scraper.start_scraping()
        
        print("\nWeb Scraping abgeschlossen!")
        print("\nStatistiken:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("Web Scraping abgebrochen.")


def run_chunking():
    """
    Führt das Chunking durch.
    """
    print("\n--- CHUNKING ---")
    print(f"Chunk-Größe: {config.CHUNKING_CHUNK_SIZE} Zeichen")
    print(f"Chunk-Overlap: {config.CHUNKING_CHUNK_OVERLAP} Zeichen")
    
    # Prüfen ob gescrapte Daten existieren
    if not os.path.exists(config.SCRAPING_OUTPUT_PATH):
        print("\nFehler: Keine gescrapten Daten gefunden!")
        print("Bitte zuerst das Web Scraping durchführen.")
        return
    
    # Den ChunkingManager erstellen und starten
    chunker = ChunkingManager()
    chunker.process()
    
    print("\nChunking abgeschlossen!")


def run_embedding():
    """
    Führt das Embedding durch.
    """
    print("\n--- EMBEDDING ---")
    print(f"Modell: {config.EMBEDDING_MODEL_NAME}")
    print(f"Batch-Größe: {config.EMBEDDING_BATCH_SIZE}")
    
    # Prüfen ob Chunks existieren
    if not os.path.exists(config.CHUNKING_OUTPUT_PATH):
        print("\nFehler: Keine Chunks gefunden!")
        print("Bitte zuerst das Chunking durchführen.")
        return
    
    # Den EmbeddingManager erstellen und starten
    embedder = EmbeddingManager()
    embedder.process()
    
    print("\nEmbedding abgeschlossen!")


def run_full_pipeline():
    """
    Führt die komplette Pipeline aus (Scraping, Chunking, Embedding).
    """
    print("\n--- KOMPLETTE PIPELINE ---")
    print("Dies führt Web Scraping, Chunking und Embedding nacheinander aus.")
    print("Das kann einige Zeit dauern.")
    
    # Bestätigung vom Nutzer holen
    print("\nMöchten Sie die komplette Pipeline starten? (j/n)")
    answer = input("> ").strip().lower()
    
    if answer == 'j' or answer == 'ja':
        # 1. Web Scraping
        print("\n" + "="*40)
        print("SCHRITT 1: Web Scraping")
        print("="*40)
        scraper = WebScrapingManager()
        stats = scraper.start_scraping()
        
        # 2. Chunking
        print("\n" + "="*40)
        print("SCHRITT 2: Chunking")
        print("="*40)
        chunker = ChunkingManager()
        chunker.process()
        
        # 3. Embedding
        print("\n" + "="*40)
        print("SCHRITT 3: Embedding")
        print("="*40)
        embedder = EmbeddingManager()
        embedder.process()
        
        print("\n" + "="*40)
        print("PIPELINE ABGESCHLOSSEN!")
        print("="*40)
        print("Die Datenbank ist jetzt bereit für Anfragen.")
    else:
        print("Pipeline abgebrochen.")


def run_query_mode():
    """
    Startet den Frage-Modus.
    Der Nutzer kann Fragen stellen und bekommt Antworten.
    """
    print("\n--- FRAGE-MODUS ---")
    print("Stellen Sie Ihre Fragen zur AOK.")
    print("Geben Sie 'exit' ein, um den Modus zu beenden.")
    
    # Prüfen ob die Datenbank existiert
    if not os.path.exists(config.CHROMA_DB_PATH):
        print("\nFehler: Keine Vektordatenbank gefunden!")
        print("Bitte zuerst die Pipeline ausführen (Option 4).")
        return
    
    # Den GenerationManager erstellen
    generator = GenerationManager()
    
    # Schleife für mehrere Fragen
    while True:
        print("\nIhre Frage:")
        question = input("> ").strip()
        
        # Prüfen ob der Nutzer beenden möchte
        if question.lower() == 'exit':
            print("Frage-Modus beendet.")
            break
        
        # Prüfen ob eine Frage eingegeben wurde
        if not question:
            print("Bitte geben Sie eine Frage ein.")
            continue
        
        # Die Anfrage ausführen
        result = generator.generate_answer(question)
        
        # Die Antwort ausgeben
        print("\n" + "-"*40)
        print("ANTWORT:")
        print("-"*40)
        print(result['answer'])
        
        # Die Quellen ausgeben
        if result['sources']:
            print("\nQuellen:")
            for i, source in enumerate(result['sources']):
                print(f"  {i+1}. {source}")


def run_evaluation():
    """
    Führt die Evaluation durch.
    """
    print("\n--- EVALUATION ---")
    
    # Prüfen ob die Datenbank existiert
    if not os.path.exists(config.CHROMA_DB_PATH):
        print("\nFehler: Keine Vektordatenbank gefunden!")
        print("Bitte zuerst die Pipeline ausführen (Option 4).")
        return
    
    # Prüfen ob Testfälle existieren
    if not os.path.exists(config.EVALUATION_TEST_CASES_PATH):
        print("\nKeine Testfälle gefunden!")
        print("Möchten Sie Beispiel-Testfälle erstellen? (j/n)")
        answer = input("> ").strip().lower()
        
        if answer == 'j' or answer == 'ja':
            create_sample_test_cases()
        else:
            print("Evaluation abgebrochen.")
            return
    
    # Bestätigung vom Nutzer holen
    print(f"\nTestfälle: {config.EVALUATION_TEST_CASES_PATH}")
    print("Möchten Sie die Evaluation starten? (j/n)")
    answer = input("> ").strip().lower()
    
    if answer == 'j' or answer == 'ja':
        # Den EvaluationManager erstellen und starten
        evaluator = EvaluationManager()
        results = evaluator.run_evaluation()
        
        if results:
            print(f"\nErgebnisse wurden gespeichert in: {config.EVALUATION_OUTPUT_PATH}")
    else:
        print("Evaluation abgebrochen.")


def main():
    """
    Die Hauptfunktion der Anwendung.
    Zeigt das Menü und verarbeitet die Nutzereingaben.
    """
    # Header ausgeben
    print_header()
    
    # Verzeichnisse erstellen
    ensure_directories()
    
    # Hauptschleife
    while True:
        # Menü anzeigen
        print_menu()
        
        # Eingabe vom Nutzer holen
        print("Ihre Wahl:")
        choice = input("> ").strip()
        
        # Die Wahl verarbeiten
        if choice == '1':
            run_web_scraping()
        elif choice == '2':
            run_chunking()
        elif choice == '3':
            run_embedding()
        elif choice == '4':
            run_full_pipeline()
        elif choice == '5':
            run_query_mode()
        elif choice == '6':
            run_evaluation()
        elif choice == '7':
            create_sample_test_cases()
        elif choice == '8':
            print("\nAuf Wiedersehen!")
            break
        else:
            print("\nUngültige Eingabe. Bitte wählen Sie 1-8.")


# Dieser Code wird nur ausgeführt, wenn die Datei direkt gestartet wird
if __name__ == "__main__":
    main()
