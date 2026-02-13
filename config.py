from dotenv import load_dotenv
load_dotenv()
import os

# config.py
# Diese Datei enthält alle Konfigurationsparameter für die RAG-Pipeline
# Hier können alle Einstellungen zentral angepasst werden

# ============================================
# WEB SCRAPING KONFIGURATION
# ============================================

# Die Start-URL von der aus gecrawlt wird
SCRAPING_START_URL = "https://www.aok.de"

# Maximale Tiefe für das Crawling (wie viele Links tief verfolgt werden)
SCRAPING_MAX_DEPTH = 2

# Diese Begriffe in URLs führen zum Ausschluss der Seite
SCRAPING_EXCLUDED_TERMS = [
    "nordwest", "bw", "bayern", "bremen", "hessen", 
    "nordost", "plus", "niedersachsen", "rps", "rh", 
    "gp", "fk", "fm", "pp", "magazin", 
    "datenschutzerklaerung", "karriere"
]

# Pfad für die gespeicherten Scraping-Ergebnisse
SCRAPING_OUTPUT_PATH = "data/scraped_data.json"

# Wartezeit zwischen Anfragen (in Sekunden) - respektiert robots.txt
SCRAPING_DOWNLOAD_DELAY = 1.0

# Proxy-Einstellungen für Firmennetzwerke (optional)
# Setze auf None wenn kein Proxy benötigt wird
# Beispiel: "http://proxy.firma.de:8080" oder "http://user:passwort@proxy.firma.de:8080"
SCRAPING_PROXY = "http://proxy.your-company.com:8080"  # z.B. "http://proxy.example.com:8080"

# ============================================
# CHUNKING KONFIGURATION
# ============================================

# Maximale Länge eines Chunks in Zeichen
CHUNKING_CHUNK_SIZE = 1000

# Überlappung zwischen Chunks in Zeichen
CHUNKING_CHUNK_OVERLAP = 200

# Pfad für die gespeicherten Chunks
CHUNKING_OUTPUT_PATH = "data/chunks.json"

# ============================================
# EMBEDDING KONFIGURATION
# ============================================

# Name des Embedding-Modells (wird von HuggingFace geladen)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# Lokaler Pfad zum Speichern des Modells
EMBEDDING_MODEL_PATH = "models/infloat"

# Batch-Größe für das Embedding (wie viele Texte gleichzeitig verarbeitet werden)
EMBEDDING_BATCH_SIZE = 32

# ============================================
# RETRIEVAL / VEKTORDATENBANK KONFIGURATION
# ============================================

# Pfad zur ChromaDB Datenbank
CHROMA_DB_PATH = "data/chroma_db"

# Name der Collection in ChromaDB
CHROMA_COLLECTION_NAME = "aok_documents"

# Anzahl der zurückgegebenen Ergebnisse bei einer Suche
RETRIEVAL_TOP_K = 10

# HNSW Parameter M (Anzahl der Verbindungen pro Knoten im Graph)
# Höhere Werte = bessere Qualität, aber mehr Speicher
HNSW_M = 16

# Pfad zur Datei mit Stoppwörtern
STOP_WORDS_PATH = "stop_words.txt"

# Cross Encoder Reranking aktivieren/deaktivieren
# True = Reranking verwenden (bessere Ergebnisse, aber langsamer)
# False = Nur Vektorsuche (schneller, aber weniger präzise)
USE_RERANKING = False

# Cross Encoder Modell für Reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Anzahl der Ergebnisse nach dem Reranking
RERANKING_TOP_K = 5

# ============================================
# LLM / GENERATION KONFIGURATION
# ============================================

# URL des LLM-API-Endpunkts (z.B. OpenAI-kompatibel)
LLM_API_URL = "https://api.your-company.com/v1/chat/completions"

# API-Schlüssel für das LLM (sollte als Umgebungsvariable gesetzt werden)
LLM_API_KEY = os.getenv("CHATBOT_API_KEY")

# Name des zu verwendenden Modells
LLM_MODEL_NAME = "Qwen3-30B-A3B-Q6_K.gguf:latest"

# Temperatur für die Textgenerierung (0.0 = deterministisch, 1.0 = kreativ)
LLM_TEMPERATURE = 0.3

# Maximale Anzahl der generierten Tokens
LLM_MAX_TOKENS = 1024

# Top-P Sampling Parameter
LLM_TOP_P = 0.9

# Frequency Penalty (reduziert Wiederholungen)
LLM_FREQUENCY_PENALTY = 0.0

# Seed für reproduzierbare Ergebnisse (None = zufällig)
LLM_SEED = 42

# System-Prompt für das LLM
LLM_SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent der AOK Sachsen-Anhalt. 
Deine Aufgabe ist es, Fragen zu Gesundheitsthemen und AOK-Leistungen zu beantworten.
Basiere deine Antworten ausschließlich auf den bereitgestellten Kontext-Informationen.
Wenn du eine Frage nicht basierend auf dem Kontext beantworten kannst, sage das ehrlich.
Antworte immer auf Deutsch und in einem freundlichen, verständlichen Ton."""

# ============================================
# EVALUATION KONFIGURATION
# ============================================

# Pfad zur Datei mit Testfällen
EVALUATION_TEST_CASES_PATH = "data/test_cases.json"

# Pfad für die Evaluations-Ergebnisse
EVALUATION_OUTPUT_PATH = "data/evaluation_results.json"

# ============================================
# ALLGEMEINE KONFIGURATION
# ============================================

# Verzeichnis für alle Daten
DATA_DIR = "data"

# Verzeichnis für Modelle
MODELS_DIR = "models"
