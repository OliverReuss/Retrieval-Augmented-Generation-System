from dotenv import load_dotenv
load_dotenv()
import os
import logging

# Verzeichnisse, Proxy, Zertifikat
DATA_DIRECTORY = "data"
MODELS_DIRECTORY = "models"
PROXY = "http://proxy.your-company.com:8080"
CERTIFICATE = r"./cert/cacert.pem"

# Proxy
if PROXY:
    os.environ["HTTP_PROXY"] = PROXY
    os.environ["HTTPS_PROXY"] = PROXY
    os.environ["http_proxy"] = PROXY
    os.environ["https_proxy"] = PROXY

# SSL-Zertifikat
if CERTIFICATE:
    os.environ["CURL_CA_BUNDLE"] = CERTIFICATE
    os.environ["REQUESTS_CA_BUNDLE"] = CERTIFICATE
    os.environ["SSL_CERT_FILE"] = CERTIFICATE

# Konsolen-Output unterdrücken
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1"
logging.getLogger("ragas.prompt.pydantic_prompt").setLevel(logging.ERROR)

# Web Scraping
WEB_SCRAPING_START_URL = "https://www.example.com/"
WEB_SCRAPING_MAX_DEPTH = 4
WEB_SCRAPING_EXCLUDED_TERMS = ["nordwest", "bw", "bayern", "bremen", "hessen", "nordost", "plus", "niedersachsen", "rps", "rh", "fm", "pp", "magazin", "datenschutzerklaerung", "karriere"]
WEB_SCRAPING_DOWNLOAD_DELAY = 0.0
WEB_SCRAPING_OUTPUT_PATH = "data/documents.json"

# Chunking
CHUNKING_CHUNK_SIZE = 512
CHUNKING_CHUNK_OVERLAP = 75
CHUNKING_OUTPUT_PATH = "data/chunks.json"

# Embedding
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 32

# Retrieval
RETRIEVAL_DB_PATH = "data/database"
RETRIEVAL_COLLECTION_NAME = "documents_collection"
RETRIEVAL_TOP_K = 20
RETRIEVAL_HNSW_SPACE = "cosine"
RETRIEVAL_HNSW_M = 32
RETRIEVAL_CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
RETRIEVAL_RERANKING_TOP_K = 5
RETRIEVAL_RERANKING_THRESHOLD = 0.5

# Generation
GENERATION_API_URL = "https://api.your-company.com/v1/chat/completions" # "https://api.openai.com/v1/chat/completions" "https://openrouter.ai/api/v1/chat/completions"
GENERATION_API_KEY = os.getenv("CHATBOT_API_KEY")
GENERATION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct" # "Qwen3-30B-A3B-Q6_K.gguf:latest" "gpt-4o"
GENERATION_TEMPERATURE = 0.0
GENERATION_MAX_TOKENS = 1024
GENERATION_TOP_P = 0.1
GENERATION_FREQUENCY_PENALTY = 0.1
GENERATION_SEED = 1
GENERATION_TIMEOUT = 60
GENERATION_SYSTEM_PROMPT = """Experten-Rolle: Du bist ein kompetenter Fachberater. Dein Ziel ist es, Nutzeranfragen basierend auf den bereitgestellten Dokumenten präzise, empathisch und fachlich korrekt zu beantworten.

Kern-Regeln:
1. Kontext-Treue: Nutze ausschließlich die bereitgestellten Informationen. Wenn eine Antwort nicht im Kontext steht, sage dies klar (keine Erfindungen!).
2. Korrektur von Fehlannahmen: Wenn eine Nutzerfrage auf einer falschen Prämisse beruht (z. B. "Warum zahlt die Kasse X nicht?"), korrigiere dies aktiv, falls der Kontext das Gegenteil belegt.
3. Umgang mit Ambiguität: Wenn eine Frage zu vage ist (z. B. "Tipps zu meiner Krankheit"), nenne kurz die verfügbaren Themen aus dem Kontext und stelle eine gezielte Rückfrage.
4. Sprache & Stil: Antworte seriös, aber verständlich. Vermeide reines Copy-Paste von Textwüsten; fasse Informationen stattdessen sinnvoll zusammen, ohne Fakten zu verfälschen.
5. Belegpflicht & Zitate: Jede fachliche Aussage muss durch eine entsprechende Stelle im Kontext gedeckt sein.

Umgang mit Wissenslücken:
- Keine Infos vorhanden: "Zu dieser spezifischen Frage liegen im bereitgestellten Kontext keine Informationen vor."
- Teilweise Infos vorhanden: Beantworte, was möglich ist, und weise auf die fehlenden Details hin.

Formatierung:
- Gliedere lange Antworten in logische Abschnitte."""

# Evaluation
EVALUATION_MODEL = "gpt-4.1"
EVALUATION_EMBEDDING_MODEL = "text-embedding-3-large"
EVALUATION_TEST_CASES_PATH = "test_cases.json"