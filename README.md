# AOK RAG-Pipeline

Eine Retrieval-Augmented Generation (RAG) Pipeline für AOK-Informationen.
Dieses Projekt wurde als Lernprojekt entwickelt und demonstriert die grundlegenden Konzepte einer RAG-Pipeline.

## Projektstruktur

```
├── config.py           # Zentrale Konfigurationsdatei
├── web_scraping.py     # Web Scraping mit Scrapy
├── chunking.py         # Text-Chunking mit LangChain
├── embedding.py        # Embedding-Erstellung mit Sentence Transformers
├── retrieval.py        # Suche in der Vektordatenbank
├── generation.py       # Antwortgenerierung mit LLM
├── evaluation.py       # Evaluation mit RAGAS
├── main.py             # Hauptprogramm mit Menü
├── stop_words.txt      # Deutsche Stoppwörter
├── requirements.txt    # Python-Abhängigkeiten
├── data/               # Daten-Verzeichnis
│   ├── scraped_data.json
│   ├── chunks.json
│   ├── test_cases.json
│   ├── evaluation_results.json
│   └── chroma_db/
└── models/             # Modell-Verzeichnis
    └── embedding_model/
```

## Installation

1. Python 3.9 oder höher wird benötigt.

2. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

4. Konfiguration anpassen:
   - Öffnen Sie `config.py`
   - Setzen Sie `LLM_API_KEY` auf Ihren API-Schlüssel
   - Passen Sie ggf. weitere Parameter an

## Verwendung

### Hauptprogramm starten

```bash
python main.py
```

Das Programm zeigt ein Menü mit folgenden Optionen:

1. **Web Scraping** - Lädt Daten von der AOK-Webseite
2. **Chunking** - Teilt die Texte in kleinere Stücke
3. **Embedding** - Erstellt Vektoren und speichert sie in ChromaDB
4. **Komplette Pipeline** - Führt Schritte 1-3 nacheinander aus
5. **Frage stellen** - Interaktiver Frage-Modus
6. **Evaluation** - Führt die Evaluation mit RAGAS durch
7. **Beispiel-Testfälle erstellen** - Erstellt eine Test-Datei

### Einzelne Module ausführen

Sie können auch einzelne Module direkt ausführen:

```bash
python web_scraping.py   # Nur Scraping
python chunking.py       # Nur Chunking
python embedding.py      # Nur Embedding
python retrieval.py      # Test-Suche
python generation.py     # Test-Anfrage
python evaluation.py     # Nur Evaluation
```

## Konfiguration

Alle Parameter können in `config.py` angepasst werden:

### Web Scraping
- `SCRAPING_START_URL` - Die Start-URL für das Crawling
- `SCRAPING_MAX_DEPTH` - Maximale Crawling-Tiefe
- `SCRAPING_EXCLUDED_TERMS` - Ausgeschlossene URL-Begriffe

### Chunking
- `CHUNKING_CHUNK_SIZE` - Maximale Chunk-Größe in Zeichen
- `CHUNKING_CHUNK_OVERLAP` - Überlappung zwischen Chunks

### Embedding & Retrieval
- `EMBEDDING_MODEL_NAME` - Name des Embedding-Modells
- `EMBEDDING_BATCH_SIZE` - Batch-Größe für das Embedding
- `RETRIEVAL_TOP_K` - Anzahl der Suchergebnisse
- `HNSW_M` - HNSW-Parameter für ChromaDB

### LLM
- `LLM_API_URL` - URL des LLM-API-Endpunkts
- `LLM_API_KEY` - API-Schlüssel
- `LLM_MODEL_NAME` - Name des Modells
- `LLM_TEMPERATURE` - Temperatur (0.0-1.0)
- `LLM_MAX_TOKENS` - Maximale Token-Anzahl
- `LLM_SYSTEM_PROMPT` - System-Prompt für das LLM

## Evaluation

Die Evaluation nutzt RAGAS und bewertet folgende Metriken:

### Retrieval-Metriken
- `context_entity_recall` - Werden wichtige Begriffe gefunden?
- `context_precision` - Wie relevant ist der Kontext?
- `context_recall` - Wie vollständig ist der Kontext?

### Generation-Metriken
- `answer_correctness` - Faktische Korrektheit der Antwort
- `answer_relevancy` - Beantwortet die Antwort die Frage?
- `answer_similarity` - Semantische Ähnlichkeit zur Ground Truth
- `faithfulness` - Basiert die Antwort auf dem Kontext?
- `summarization_score` - Zusammenfassungsqualität

### Web Scraping-Metriken
- `webscraping_elements_total` - Gesamtzahl der Elemente
- `webscraping_elements_with_text` - Elemente mit Text
- `webscraping_extraction_success_rate` - Erfolgsrate
- `webscraping_duplicate_rate_by_text` - Duplikatrate
- `webscraping_average_text_length` - Durchschnittliche Textlänge

## Testfälle

Testfälle werden in `data/test_cases.json` im folgenden Format gespeichert:

```json
[
  {
    "type": "leistung",
    "prompt": "Die Frage des Nutzers",
    "truth": "Die erwartete korrekte Antwort",
    "source": "https://url-der-quelle.de"
  }
]
```

## Hinweise

- Das Web Scraping respektiert die robots.txt der AOK-Webseite
- Zwischen Anfragen wird eine Wartezeit eingehalten
- Das Embedding-Modell wird lokal gespeichert für schnelleren Zugriff
- ChromaDB speichert die Vektordatenbank persistent

## Lizenz

Dieses Projekt ist ein Bildungsprojekt und dient nur zu Lernzwecken.
