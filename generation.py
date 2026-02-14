from requests.adapters import HTTPAdapter
from retrieval import Retriever
from urllib3.util.retry import Retry
import config
import requests

"""
# SSL-Warnungen deaktivieren (für Firmen-Proxy)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
"""

class Generator:

    def __init__(self) -> None:
        self.api_url = config.GENERATION_API_URL
        self.api_key = config.GENERATION_API_KEY
        self.model_name = config.GENERATION_MODEL_NAME
        self.temperature = config.GENERATION_TEMPERATURE
        self.max_tokens = config.GENERATION_MAX_TOKENS
        self.top_p = config.GENERATION_TOP_P
        self.frequency_penalty = config.GENERATION_FREQUENCY_PENALTY
        self.seed = config.GENERATION_SEED
        self.system_prompt = config.GENERATION_SYSTEM_PROMPT
        self.timeout = config.GENERATION_TIMEOUT
        self.retrieval_manager = Retriever()
        
        # Session mit Retry-Strategie erstellen
        self.session = self.create_retry_session()
    
    def create_retry_session(self) -> requests.Session:
        session = requests.Session()
        if config.PROXY:
            session.proxies = {
                'http': config.PROXY,
                'https': config.PROXY
            }
            # SSL-Verifizierung deaktivieren
            session.verify = False
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session
    
    def create_prompt(self, query: str, context: str) -> str:
        # Kombiniert Kontext und Query
        prompt = f"""Basierend auf den folgenden Informationen, beantworte die Frage des Nutzers.

        KONTEXT:
        {context}

        FRAGE:
        {query}

        ANTWORT:"""
        
        return prompt
    
    def call_llm(self, user_prompt: str) -> str:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        # Request-Body im OpenAI-Format (von anderen LLMs auch genutzt)
        payload = {
            'model': self.model_name,
            'messages': [
                {
                    'role': 'system',
                    'content': self.system_prompt
                },
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'seed': self.seed
        }
 
        # Anfrage mit Retry-Session stellen
        try:
            response = self.session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Antwort generieren (Format {"choices": [{"message": {"content": "..."}}]})
            response_text = data['choices'][0]['message']['content'].strip()
            # Antwort bereinigen
            cleaned_response = response_text.replace("*", "").replace("#", "").strip()
            return cleaned_response
        except Exception as e:
            return f"Fehler bei der Anfrage an das LLMe: {e}"
    
    def generate(self, query: str) -> dict:
        search_results = self.retrieval_manager.search(query)
        context = self.retrieval_manager.format_context(search_results)
        sources = self.retrieval_manager.get_sources(search_results)
        prompt = self.create_prompt(query, context)
        answer = self.call_llm(prompt)
        result = {
            'query': query,
            'answer': answer,
            'context': context,
            'sources': sources,
            'search_results': search_results
        }
        return result