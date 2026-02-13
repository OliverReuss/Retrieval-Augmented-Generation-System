# generation.py
# Dieses Modul ist für die Antwortgenerierung mit dem LLM zuständig
# Es kombiniert die Nutzeranfrage mit dem Kontext und sendet sie an das LLM

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import json

# SSL-Warnungen deaktivieren (für Firmen-Proxy)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Wir importieren unsere Konfiguration
import config

# Wir importieren den RetrievalManager für die Suche
from retrieval import RetrievalManager


class GenerationManager:
    """
    Diese Klasse ist für die Generierung von Antworten zuständig.
    Sie nutzt ein LLM (Large Language Model) um Antworten zu erstellen.
    """
    
    def __init__(self):
        """
        Initialisiert den GenerationManager mit den Konfigurationswerten.
        """
        # Die URL des LLM-API-Endpunkts
        self.api_url = config.LLM_API_URL
        
        # Der API-Schlüssel
        self.api_key = config.LLM_API_KEY
        
        # Der Name des Modells
        self.model_name = config.LLM_MODEL_NAME
        
        # Die Temperatur für die Textgenerierung
        self.temperature = config.LLM_TEMPERATURE
        
        # Maximale Anzahl der generierten Tokens
        self.max_tokens = config.LLM_MAX_TOKENS
        
        # Top-P Parameter
        self.top_p = config.LLM_TOP_P
        
        # Frequency Penalty
        self.frequency_penalty = config.LLM_FREQUENCY_PENALTY
        
        # Seed für reproduzierbare Ergebnisse
        self.seed = config.LLM_SEED
        
        # Der System-Prompt
        self.system_prompt = config.LLM_SYSTEM_PROMPT
        
        # Der RetrievalManager für die Suche
        self.retrieval_manager = RetrievalManager()
        
        # Session mit Retry-Strategie erstellen
        self.session = self.create_retry_session()
        
        print(f"GenerationManager initialisiert:")
        print(f"  Modell: {self.model_name}")
        print(f"  Temperatur: {self.temperature}")
        print(f"  Max Tokens: {self.max_tokens}")
    
    def create_retry_session(self):
        """
        Erstellt eine Session mit automatischen Retries.
        Bei bestimmten HTTP-Fehlern wird die Anfrage automatisch wiederholt.
        """
        session = requests.Session()
        
        # Proxy-Einstellungen aus der Config verwenden
        if config.SCRAPING_PROXY:
            session.proxies = {
                'http': config.SCRAPING_PROXY,
                'https': config.SCRAPING_PROXY
            }
            # SSL-Verifizierung deaktivieren für Firmen-Proxy
            session.verify = False
            print(f"Proxy für LLM gesetzt: {config.SCRAPING_PROXY}")
        
        retry_strategy = Retry(
            total=3,  # Maximal 3 Versuche
            backoff_factor=1,  # Wartezeit zwischen Versuchen
            status_forcelist=[429, 500, 502, 503, 504],  # Bei diesen Status-Codes wiederholen
            allowed_methods=["POST"]  # Nur POST-Anfragen wiederholen
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def create_prompt(self, user_query, context):
        """
        Erstellt den vollständigen Prompt für das LLM.
        Kombiniert den Kontext mit der Nutzeranfrage.
        """
        # Der Prompt enthält den Kontext und die Frage
        prompt = f"""Basierend auf den folgenden Informationen, beantworte die Frage des Nutzers.

KONTEXT:
{context}

FRAGE:
{user_query}

ANTWORT:"""
        
        return prompt
    
    def call_llm(self, user_prompt):
        """
        Sendet den Prompt an das LLM und gibt die Antwort zurück.
        
        Parameter:
            user_prompt: Der vollständige Prompt mit Kontext und Frage
        
        Rückgabe:
            Die generierte Antwort als String
        """
        # Die HTTP-Header für die Anfrage
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        # Der Request-Body im OpenAI-Format
        # Dieses Format wird von vielen LLM-APIs unterstützt
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
        }
        
        # Seed hinzufügen wenn gesetzt
        if self.seed is not None:
            payload['seed'] = self.seed
        
        print(f"Sende Anfrage an LLM: {self.api_url}")
        
        # Die Anfrage an das LLM senden mit der Retry-Session
        try:
            response = self.session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60  # 60 Sekunden Timeout
            )
            response.raise_for_status()
            
            # Die Antwort als JSON parsen
            data = response.json()
            
            # Die generierte Antwort extrahieren
            # Das Format ist: {"choices": [{"message": {"content": "..."}}]}
            if 'choices' in data and len(data['choices']) > 0:
                response_text = data['choices'][0]['message']['content'].strip()
                # Antwort bereinigen (Markdown-Formatierung entfernen)
                cleaned_response = response_text.replace("*", "").replace("#", "").strip()
                return cleaned_response
            else:
                print("Unerwartetes Antwortformat vom LLM")
                return "Fehler: Konnte keine Antwort aus der LLM-Antwort extrahieren."
                
        except requests.exceptions.Timeout:
            print("Fehler: Timeout bei der LLM-Anfrage")
            return "Fehler: Die Anfrage an das LLM hat zu lange gedauert."
            
        except requests.exceptions.ConnectionError:
            print("Fehler: Keine Verbindung zum LLM-Server")
            return "Fehler: Konnte keine Verbindung zum LLM-Server herstellen."
        
        except requests.exceptions.HTTPError as e:
            print(f"Fehler vom LLM-Server: {e}")
            return f"Fehler bei der Anfrage an das LLM: {e}"
            
        except Exception as e:
            print(f"Unerwarteter Fehler: {str(e)}")
            return f"Fehler bei der LLM-Anfrage: {str(e)}"
    
    def generate_answer(self, user_query):
        """
        Generiert eine Antwort auf die Nutzeranfrage.
        Führt die vollständige RAG-Pipeline aus:
        1. Suche nach relevantem Kontext
        2. Erstelle Prompt mit Kontext
        3. Generiere Antwort mit LLM
        
        Rückgabe:
            Ein Dictionary mit Antwort, Kontext und Quellen
        """
        print(f"\nGeneriere Antwort für: '{user_query}'")
        
        # 1. Suche nach relevantem Kontext
        print("\n1. Suche relevante Informationen...")
        search_results = self.retrieval_manager.search(user_query)
        
        # Den Kontext formatieren
        context = self.retrieval_manager.format_context(search_results)
        
        # Die Quellen extrahieren
        sources = self.retrieval_manager.get_sources(search_results)
        
        # 2. Den Prompt erstellen
        print("\n2. Erstelle Prompt...")
        prompt = self.create_prompt(user_query, context)
        
        # 3. Das LLM aufrufen
        print("\n3. Generiere Antwort mit LLM...")
        answer = self.call_llm(prompt)
        
        print("\nAntwort generiert!")
        
        # Das Ergebnis als Dictionary zurückgeben
        result = {
            'query': user_query,
            'answer': answer,
            'context': context,
            'sources': sources,
            'search_results': search_results
        }
        
        return result
    
    def generate_answer_with_context(self, user_query, context, sources):
        """
        Generiert eine Antwort mit bereits vorhandenem Kontext.
        Nützlich wenn der Kontext bereits aus einer anderen Quelle kommt.
        
        Rückgabe:
            Die generierte Antwort als String
        """
        # Den Prompt erstellen
        prompt = self.create_prompt(user_query, context)
        
        # Das LLM aufrufen
        answer = self.call_llm(prompt)
        
        return answer


def run_rag_query(query):
    """
    Hilfsfunktion um eine RAG-Anfrage auszuführen.
    Kann von anderen Modulen oder der Kommandozeile genutzt werden.
    """
    # Den GenerationManager erstellen
    generator = GenerationManager()
    
    # Die Anfrage ausführen
    result = generator.generate_answer(query)
    
    return result


# Dieser Code wird nur ausgeführt, wenn die Datei direkt gestartet wird
if __name__ == "__main__":
    # Zum Testen können wir eine Anfrage direkt starten
    
    # Eine Test-Anfrage
    test_query = "Welche Leistungen bietet die AOK für Familien?"
    
    # Die Anfrage ausführen
    result = run_rag_query(test_query)
    
    # Das Ergebnis ausgeben
    print("\n" + "="*60)
    print("ERGEBNIS")
    print("="*60)
    
    print(f"\nFrage: {result['query']}")
    
    print(f"\nAntwort:\n{result['answer']}")
    
    print(f"\nQuellen:")
    for i, source in enumerate(result['sources']):
        print(f"  {i+1}. {source}")
