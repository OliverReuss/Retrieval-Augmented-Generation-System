from dotenv import load_dotenv
load_dotenv()
import os

# Verzeichnisse, Proxy, Zertifikat
DATA_DIRECTORY = "data"
MODELS_DIRECTORY = "models"
PROXY = "http://proxy.your-company.com:8080"
CERTIFICATE = "certificate.crt"

# Proxy
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY
os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY

# SSL-Zertifikat
os.environ["CURL_CA_BUNDLE"] = CERTIFICATE
os.environ["REQUESTS_CA_BUNDLE"] = CERTIFICATE
os.environ["SSL_CERT_FILE"] = CERTIFICATE

# Konsolen-Output unterdrücken
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Web Scraping
WEB_SCRAPING_START_URL = "https://www.example.com/"
WEB_SCRAPING_MAX_DEPTH = 1
WEB_SCRAPING_EXCLUDED_TERMS = ["nordwest", "bw", "bayern", "bremen", "hessen", "nordost", "plus", "niedersachsen", "rps", "rh", "gp", "fk", "fm", "pp", "magazin", "datenschutzerklaerung", "karriere"]
WEB_SCRAPING_DOWNLOAD_DELAY = 0.0
WEB_SCRAPING_OUTPUT_PATH = "data/documents.json"

# Chunking
CHUNKING_CHUNK_SIZE = 1200
CHUNKING_CHUNK_OVERLAP = 100
CHUNKING_OUTPUT_PATH = "data/chunks.json"

# Embedding
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" # "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 32

# Retrieval
RETRIEVAL_DB_PATH = "data/database"
RETRIEVAL_COLLECTION_NAME = "documents_collection"
RETRIEVAL_TOP_K = 10
RETRIEVAL_HNSW_SPACE = "cosine"
RETRIEVAL_HNSW_M = 32
RETRIEVAL_STOP_WORDS_PATH = "data/stop_words.txt"
RETRIEVAL_USE_RERANKING = True
RETRIEVAL_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_RERANKING_TOP_K = 5

# Generation
GENERATION_API_URL = "https://api.your-company.com/v1/chat/completions" # "https://openrouter.ai/api/v1/chat/completions"
GENERATION_API_KEY = os.getenv("CHATBOT_API_KEY")
GENERATION_MODEL_NAME = "Qwen3-30B-A3B-Q6_K.gguf:latest" # "qwen/qwen3-30b-a3b:free"
GENERATION_TEMPERATURE = 0.1
GENERATION_MAX_TOKENS = 800
GENERATION_TOP_P = 0.9
GENERATION_FREQUENCY_PENALTY = 0.1
GENERATION_SEED = 1
GENERATION_TIMEOUT = 60
GENERATION_SYSTEM_PROMPT = """Du bist ein Experte für die Beantwortung von Fragen basierend auf bereitgestellten Dokumenten. 

    AUFGABE: Beantworte die Nutzerfrage präzise und vollständig unter Verwendung AUSSCHLIESSLICH der bereitgestellten Kontextinformationen.

    REGELN:
    - Nutze nur Informationen aus dem bereitgestellten Kontext
    - Nutze relevante Textpassagen wörtlich, wenn möglich
    - Strukturiere deine Antwort logisch und übersichtlich
    - Erkläre komplexe Sachverhalte verständlich
    - Bei unvollständigen Informationen im Kontext: ergänze mit "Weitere Details sind im bereitgestellten Kontext nicht verfügbar"
    - Bei fehlenden Informationen: "Zu dieser Frage liegen im bereitgestellten Kontext keine Informationen vor"
    - Bei unzurecheichendem oder widersprüchlichem Input: "Bitte präzisiere deine Frage oder gib mehr Informationen an."

    ANTWORTFORMAT:
    1. Beginne direkt mit der Antwort
    2. Verwende Absätze für bessere Lesbarkeit
    3. Markiere wichtige Punkte deutlich"""

# Evaluation
EVALUATION_EVALUATE = True
EVALUATION_MODEL_NAME = "gpt-4o"
EVALUATION_TEST_CASES_PATH = "data/test_cases.json"
EVALUATION_OUTPUT_PATH = "data/evaluation_results.json"