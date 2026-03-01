from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import config
import requests
import time
import tiktoken

class Generator:

    def __init__(self, retriever) -> None:
        self.api_url = config.GENERATION_API_URL
        self.api_key = config.GENERATION_API_KEY
        self.model = config.GENERATION_MODEL
        self.temperature = config.GENERATION_TEMPERATURE
        self.max_tokens = config.GENERATION_MAX_TOKENS
        self.top_p = config.GENERATION_TOP_P
        self.frequency_penalty = config.GENERATION_FREQUENCY_PENALTY
        self.seed = config.GENERATION_SEED
        self.system_prompt = config.GENERATION_SYSTEM_PROMPT
        self.timeout = config.GENERATION_TIMEOUT
        self.retriever = retriever
        # Statistiken
        self.generation_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.generation_times_ms = []
        self.response_lengths = []
        self.api_errors = 0
        self.retries = 0
        # Session mit Retry-Strategie erstellen
        self.session = self.create_retry_session()
        # Tokenizer für Token-Zählung
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

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
    
    def call_llm(self, user_prompt: str) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        # Request-Body im OpenAI-Format (von anderen LLMs auch genutzt)
        payload = {
            'model': self.model,
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

        # Input-Tokens zählen
        input_text = self.system_prompt + user_prompt
        input_tokens = self.count_tokens(input_text)
        
        # Zeitmessung starten
        start_time = time.time()
 
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

            # Zeitmessung beenden
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Antwort generieren (Format {"choices": [{"message": {"content": "..."}}]})
            response_text = data['choices'][0]['message']['content'].strip()
            # Antwort bereinigen
            cleaned_response = response_text.replace("*", "").replace("#", "").replace("<think>", "").replace("</think>", "").strip()
            
            # Output-Tokens zählen (aus API-Response oder selbst berechnen)
            if 'usage' in data:
                output_tokens = data['usage'].get('completion_tokens', self.count_tokens(cleaned_response))
                input_tokens = data['usage'].get('prompt_tokens', input_tokens)
            else:
                output_tokens = self.count_tokens(cleaned_response)
            
            # Statistiken aktualisieren
            self.generation_count += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.generation_times_ms.append(generation_time_ms)
            self.response_lengths.append(len(cleaned_response))
            
            return {
                'text': cleaned_response,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'generation_time_ms': generation_time_ms
            }
            
        except Exception as e:
            self.api_errors += 1
            return {
                'text': f"Fehler bei der Anfrage an das LLM: {e}",
                'input_tokens': input_tokens,
                'output_tokens': 0,
                'generation_time_ms': 0
            }
    
    def generate(self, query: str) -> dict:
        search_results = self.retriever.search(query)
        context = self.retriever.format_context(search_results)
        sources = self.retriever.get_sources(search_results)
        prompt = self.create_prompt(query, context)
        llm_response = self.call_llm(prompt)

        result = {
            'query': query,
            'answer': llm_response['text'],
            'context': context,
            'sources': sources,
            'search_results': search_results,
            # Metriken für diese einzelne Generierung
            'generation_metrics': {
                'input_tokens': llm_response['input_tokens'],
                'output_tokens': llm_response['output_tokens'],
                'generation_time_ms': round(llm_response['generation_time_ms'], 2),
                'response_length': len(llm_response['text'])
            }
        }
        return result

    def get_statistics(self) -> dict:
        # Durchschnittliche Antwortlänge
        avg_response_length = 0.0
        if self.response_lengths:
            avg_response_length = sum(self.response_lengths) / len(self.response_lengths)
        
        # Durchschnittliche Generierungszeit
        avg_generation_time_ms = 0.0
        if self.generation_times_ms:
            avg_generation_time_ms = sum(self.generation_times_ms) / len(self.generation_times_ms)
        
        # Durchschnittliche Token-Anzahlen
        avg_input_tokens = 0.0
        avg_output_tokens = 0.0
        if self.generation_count > 0:
            avg_input_tokens = self.total_input_tokens / self.generation_count
            avg_output_tokens = self.total_output_tokens / self.generation_count
        
        stats = {
            'generation_avg_length': round(avg_response_length, 4),
            'generation_avg_time': round(avg_generation_time_ms, 4),
            'generation_avg_input_tokens': round(avg_input_tokens, 4),
            'generation_avg_output_tokens': round(avg_output_tokens, 4),
            'generation_api_errors': self.api_errors,
        }

        return stats
    
    def reset_statistics(self) -> None:
        self.generation_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.generation_times_ms = []
        self.response_lengths = []
        self.api_errors = 0
        self.retries = 0