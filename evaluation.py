from datasets import Dataset
from datetime import datetime
from generation import Generator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (context_entity_recall, context_precision, context_recall, answer_correctness, answer_relevancy, answer_similarity, faithfulness)
from ragas.metrics._summarization import SummarizationScore
from retrieval import RetrievalManager
from web_scraping import Web_Scraper, load_scraped_data
import config
import json
import os

class Evaluator:

    def __init__(self):
        self.test_cases_path = config.EVALUATION_TEST_CASES_PATH
        self.output_path = config.EVALUATION_OUTPUT_PATH
        self.retrieval_manager = RetrievalManager()
        self.generator = Generator()
        self.test_cases = []
        self.results = []
        self.webscraping_stats = None
        self.llm = None
        self.embeddings = None
    
    def setup_ragas_llm(self):
        """
        Richtet das LLM für RAGAS ein.
        RAGAS braucht ein LLM um einige Metriken zu berechnen.
        """
        # Prüfen ob bereits eingerichtet
        if self.llm is not None:
            return
        
        print("Richte RAGAS LLM ein...")
        
        # Das LLM von LangChain erstellen mit LangchainLLMWrapper
        base_llm = ChatOpenAI(
            model=config.LLM_MODEL_NAME,
            api_key=config.LLM_API_KEY,
            temperature=0,  # Für Evaluation deterministisch
            base_url=config.LLM_API_URL.replace('/chat/completions', ''),
            max_completion_tokens=8192,
            http_client=None,
            http_async_client=None,
            timeout=60
        )
        self.llm = LangchainLLMWrapper(base_llm)
        
        # Embeddings für semantische Ähnlichkeit mit LangchainEmbeddingsWrapper
        base_embeddings = OpenAIEmbeddings(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_API_URL.replace('/chat/completions', ''),
            http_client=None,
            http_async_client=None,
            timeout=60
        )
        self.embeddings = LangchainEmbeddingsWrapper(base_embeddings)
        
        print("RAGAS LLM eingerichtet.")
    
    def load_test_cases(self):
        """
        Lädt die Testfälle aus der JSON-Datei.
        Format: {"type", "prompt", "truth", "source"}
        """
        # Prüfen ob die Datei existiert
        if not os.path.exists(self.test_cases_path):
            print(f"Fehler: Testfälle nicht gefunden: {self.test_cases_path}")
            return []
        
        # Die Datei laden
        with open(self.test_cases_path, 'r', encoding='utf-8') as file:
            self.test_cases = json.load(file)
        
        print(f"Testfälle geladen: {len(self.test_cases)} Einträge")
        return self.test_cases
    
    def calculate_webscraping_stats(self):
        """
        Berechnet die Web Scraping Statistiken.
        Diese werden nur einmal am Anfang berechnet.
        """
        print("Berechne Web Scraping Statistiken...")
        
        # Die gescrapten Daten laden
        scraped_data = load_scraped_data()
        
        if not scraped_data:
            print("Keine gescrapten Daten gefunden!")
            self.webscraping_stats = {
                'webscraping_elements_total': 0,
                'webscraping_elements_with_text': 0,
                'webscraping_extraction_success_rate': 0.0,
                'webscraping_duplicate_rate_by_text': 0.0,
                'webscraping_average_text_length': 0.0
            }
            return self.webscraping_stats
        
        # Gesamtanzahl der Elemente
        total_elements = len(scraped_data)
        
        # Anzahl der Elemente mit Text
        elements_with_text = 0
        text_lengths = []
        all_texts = []
        
        for item in scraped_data:
            text = item.get('text', '')
            if text and len(text.strip()) > 0:
                elements_with_text = elements_with_text + 1
                text_lengths.append(len(text))
                all_texts.append(text)
        
        # Erfolgsrate berechnen
        if total_elements > 0:
            extraction_success_rate = elements_with_text / total_elements
        else:
            extraction_success_rate = 0.0
        
        # Duplikatrate berechnen
        unique_texts = set(all_texts)
        if len(all_texts) > 0:
            duplicate_rate = 1 - (len(unique_texts) / len(all_texts))
        else:
            duplicate_rate = 0.0
        
        # Durchschnittliche Textlänge
        if len(text_lengths) > 0:
            average_text_length = sum(text_lengths) / len(text_lengths)
        else:
            average_text_length = 0.0
        
        # Die Statistiken speichern
        self.webscraping_stats = {
            'webscraping_elements_total': total_elements,
            'webscraping_elements_with_text': elements_with_text,
            'webscraping_extraction_success_rate': round(extraction_success_rate, 4),
            'webscraping_duplicate_rate_by_text': round(duplicate_rate, 4),
            'webscraping_average_text_length': round(average_text_length, 2)
        }
        
        print("Web Scraping Statistiken berechnet.")
        return self.webscraping_stats
    
    def run_single_test(self, test_case, index):
        """
        Führt einen einzelnen Testfall durch.
        
        Parameter:
            test_case: Der Testfall mit type, prompt, truth, source
            index: Der Index des Testfalls
        
        Rückgabe:
            Ein Dictionary mit allen Ergebnissen
        """
        print(f"\n--- Testfall {index + 1} ---")
        
        # Die Werte aus dem Testfall extrahieren
        test_type = test_case.get('type', '')
        prompt = test_case.get('prompt', '')
        truth = test_case.get('truth', '')
        source = test_case.get('source', '')
        
        print(f"Typ: {test_type}")
        print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")
        
        # 1. Die Suche durchführen (Retrieval)
        search_results = self.retrieval_manager.search(prompt)
        
        # Den Kontext formatieren
        context = self.retrieval_manager.format_context(search_results)
        
        # Die Quellen extrahieren
        sources = self.retrieval_manager.get_sources(search_results)
        
        # 2. Die Antwort generieren
        response = self.generator.generate_answer_with_context(prompt, context, sources)
        
        print(f"Antwort: {response[:50]}..." if len(response) > 50 else f"Antwort: {response}")
        
        # Das Ergebnis-Dictionary erstellen
        result = {
            'index': index,
            'type': test_type,
            'prompt': prompt,
            'truth': truth,
            'source': source,
            'context': context,
            'sources': sources,
            'response': response,
            'metriken': {}
        }
        
        return result
    
    def evaluate_ragas(self, prompt, response, context, truth):
        """
        Führt die RAGAS Evaluation für einen einzelnen Testfall durch.
        
        Parameter:
            prompt: Die Frage/Anfrage
            response: Die generierte Antwort
            context: Der abgerufene Kontext (als Liste)
            truth: Die Ground Truth Antwort
        
        Rückgabe:
            Dictionary mit den RAGAS Metriken
        """
        # Daten vorbereiten im korrekten RAGAS Format
        data = {
            "question": [prompt],
            "answer": [response],
            "retrieved_contexts": [context] if isinstance(context, list) else [[context]],
            "ground_truths": [[truth]],
            "reference": [truth],
            "reference_contexts": [context] if isinstance(context, list) else [[context]]
        }
        
        # Dict zu Dataset umwandeln
        dataset = Dataset.from_dict(data)
        
        # Summarization Score Instanz erstellen
        summarization_score = SummarizationScore()
        
        try:
            # RAGAS Evaluation ausführen
            ragas_results = evaluate(
                dataset=dataset,
                metrics=[
                    answer_correctness,
                    answer_relevancy,
                    answer_similarity,
                    context_entity_recall,
                    context_precision,
                    context_recall,
                    faithfulness,
                    summarization_score,
                ],
                llm=self.llm,
                embeddings=self.embeddings,
                show_progress=False
            )
            return ragas_results.scores[0]
        except Exception as e:
            print(f"\n❗ Bei der Evaluation des Testfalls mit RAGAS ist ein Fehler aufgetreten: {e}")
            return {}

    def calculate_metrics(self, results):
        """
        Berechnet die RAGAS Metriken für alle Ergebnisse.
        """
        print("\n=== Berechne Metriken ===")
        
        # RAGAS LLM einrichten
        self.setup_ragas_llm()
        
        print("Berechne Metriken mit RAGAS...")
        
        # Für jeden Testfall einzeln evaluieren
        for i, result in enumerate(results):
            print(f"  Evaluiere Testfall {i + 1}/{len(results)}...")
            
            prompt = result['prompt']
            response = result['response']
            context = result['context']
            truth = result['truth']
            
            # RAGAS Evaluation für diesen Testfall
            ragas_scores = self.evaluate_ragas(prompt, response, context, truth)
            
            # Die Metriken zum Ergebnis hinzufügen
            metriken = {}
            
            # Alle verfügbaren Metriken übernehmen
            for metric_name, value in ragas_scores.items():
                if value is not None:
                    metriken[metric_name] = float(value) if not isinstance(value, (int, float)) else value
            
            result['metriken'] = metriken
        
        # Zusätzliche Metriken berechnen (die nicht in RAGAS sind)
        self.calculate_additional_metrics(results)
        
        print("Metriken berechnet.")
        return results
    
    def calculate_additional_metrics(self, results):
        """
        Berechnet zusätzliche Metriken, die nicht in RAGAS enthalten sind.
        """
        print("Berechne zusätzliche Metriken...")
        
        for result in results:
            metriken = result.get('metriken', {})
            
            # Summarization Score
            # Wir berechnen einen einfachen Score basierend auf der Kontextabdeckung
            context = result.get('context', '')
            response = result.get('response', '')
            
            if context and response:
                # Einfache Berechnung: Wie viele Wörter aus dem Kontext sind in der Antwort?
                context_words = set(context.lower().split())
                response_words = set(response.lower().split())
                common_words = context_words.intersection(response_words)
                
                if len(context_words) > 0:
                    summarization_score = len(common_words) / len(context_words)
                else:
                    summarization_score = 0.0
                
                metriken['summarization_score'] = round(summarization_score, 4)
            else:
                metriken['summarization_score'] = 0.0
            
            # ECE Score (Expected Calibration Error)
            # Da wir keine Wahrscheinlichkeiten vom LLM bekommen, setzen wir einen Placeholder
            metriken['ece_score'] = 0.0
            
            # Brier Score
            # Auch hier setzen wir einen Placeholder
            metriken['brier_score'] = 0.0
            
            # Discrimination Score
            metriken['discrimination'] = 0.0
            
            # Die aktualisierten Metriken speichern
            result['metriken'] = metriken
    
    def calculate_averages(self, results):
        """
        Berechnet die Durchschnittswerte aller Metriken über alle Testfälle.
        """
        print("Berechne Durchschnittswerte...")
        
        # Alle Metrik-Namen sammeln
        all_metric_names = set()
        for result in results:
            metriken = result.get('metriken', {})
            for name in metriken.keys():
                all_metric_names.add(name)
        
        # Durchschnitte berechnen
        averages = {}
        
        for metric_name in all_metric_names:
            values = []
            for result in results:
                metriken = result.get('metriken', {})
                if metric_name in metriken:
                    value = metriken[metric_name]
                    # Prüfen ob der Wert eine Zahl ist
                    if isinstance(value, (int, float)) and not (value != value):  # NaN Check
                        values.append(value)
            
            # Durchschnitt berechnen
            if len(values) > 0:
                average = sum(values) / len(values)
                averages[metric_name] = round(average, 4)
            else:
                averages[metric_name] = 0.0
        
        return averages
    
    def get_config_parameters(self):
        """
        Sammelt alle Konfigurationsparameter für die Ergebnisdatei.
        """
        parameters = {
            # Web Scraping
            'scraping_start_url': config.SCRAPING_START_URL,
            'scraping_max_depth': config.SCRAPING_MAX_DEPTH,
            'scraping_excluded_terms': config.SCRAPING_EXCLUDED_TERMS,
            
            # Chunking
            'chunking_chunk_size': config.CHUNKING_CHUNK_SIZE,
            'chunking_chunk_overlap': config.CHUNKING_CHUNK_OVERLAP,
            
            # Embedding
            'embedding_model_name': config.EMBEDDING_MODEL_NAME,
            'embedding_batch_size': config.EMBEDDING_BATCH_SIZE,
            
            # Retrieval
            'retrieval_top_k': config.RETRIEVAL_TOP_K,
            'hnsw_m': config.HNSW_M,
            'reranking_top_k': config.RERANKING_TOP_K,
            'cross_encoder_model': config.CROSS_ENCODER_MODEL,
            
            # Generation
            'llm_model_name': config.LLM_MODEL_NAME,
            'llm_temperature': config.LLM_TEMPERATURE,
            'llm_max_tokens': config.LLM_MAX_TOKENS,
            'llm_top_p': config.LLM_TOP_P,
            'llm_frequency_penalty': config.LLM_FREQUENCY_PENALTY,
            'llm_seed': config.LLM_SEED,
        }
        
        return parameters
    
    def run_evaluation(self):
        """
        Führt die vollständige Evaluation durch.
        """
        print("\n" + "="*60)
        print("STARTE EVALUATION")
        print("="*60)
        
        # 1. Testfälle laden
        print("\n1. Lade Testfälle...")
        test_cases = self.load_test_cases()
        
        if not test_cases:
            print("Keine Testfälle gefunden. Abbruch.")
            return None
        
        # 2. Web Scraping Statistiken berechnen
        print("\n2. Berechne Web Scraping Statistiken...")
        webscraping_stats = self.calculate_webscraping_stats()
        
        # 3. Jeden Testfall durchführen
        print(f"\n3. Führe {len(test_cases)} Testfälle durch...")
        results = []
        
        for i, test_case in enumerate(test_cases):
            result = self.run_single_test(test_case, i)
            results.append(result)
        
        # 4. Metriken berechnen
        print("\n4. Berechne Metriken...")
        results = self.calculate_metrics(results)
        
        # 5. Durchschnitte berechnen
        print("\n5. Berechne Durchschnittswerte...")
        averages = self.calculate_averages(results)
        
        # 6. Konfigurationsparameter sammeln
        print("\n6. Sammle Konfigurationsparameter...")
        config_params = self.get_config_parameters()
        
        # 7. Ergebnisse zusammenstellen
        print("\n7. Stelle Ergebnisse zusammen...")
        
        final_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'configuration': config_params,
            'webscraping_statistics': webscraping_stats,
            'metric_averages': averages,
            'test_results': results
        }
        
        # 8. Ergebnisse speichern
        print("\n8. Speichere Ergebnisse...")
        self.save_results(final_results)
        
        print("\n" + "="*60)
        print("EVALUATION ABGESCHLOSSEN")
        print("="*60)
        
        # Zusammenfassung ausgeben
        self.print_summary(final_results)
        
        return final_results
    
    def save_results(self, results):
        """
        Speichert die Evaluations-Ergebnisse als JSON-Datei.
        """
        # Das Ausgabeverzeichnis erstellen falls nötig
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Die Ergebnisse als JSON speichern
        with open(self.output_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
        
        print(f"Ergebnisse gespeichert in: {self.output_path}")
    
    def print_summary(self, results):
        """
        Gibt eine Zusammenfassung der Evaluation aus.
        """
        print("\n--- ZUSAMMENFASSUNG ---")
        
        # Web Scraping Statistiken
        print("\nWeb Scraping Statistiken:")
        ws_stats = results.get('webscraping_statistics', {})
        for key, value in ws_stats.items():
            print(f"  {key}: {value}")
        
        # Durchschnittliche Metriken
        print("\nDurchschnittliche Metriken:")
        averages = results.get('metric_averages', {})
        for key, value in averages.items():
            print(f"  {key}: {value}")
        
        # Anzahl der Testfälle
        test_results = results.get('test_results', [])
        print(f"\nAnzahl ausgewerteter Testfälle: {len(test_results)}")


def create_sample_test_cases():
    """
    Erstellt eine Beispiel-Datei mit Testfällen.
    Kann genutzt werden um das Format zu zeigen.
    """
    sample_cases = [
        {
            "type": "leistung",
            "prompt": "Welche Leistungen bietet die AOK für Familien?",
            "truth": "Die AOK bietet verschiedene Leistungen für Familien an, darunter Vorsorgeuntersuchungen für Kinder, Schwangerschaftsbegleitung und Familienprogramme.",
            "source": "https://www.example.com/leistungen/familie/"
        },
        {
            "type": "service",
            "prompt": "Wie kann ich mich bei der AOK anmelden?",
            "truth": "Die Anmeldung bei der AOK kann online, telefonisch oder in einer Geschäftsstelle erfolgen. Sie benötigen Ihre persönlichen Daten und ggf. Nachweise.",
            "source": "https://www.example.com/mitglied-werden/"
        },
        {
            "type": "information",
            "prompt": "Was ist die elektronische Patientenakte?",
            "truth": "Die elektronische Patientenakte (ePA) ist eine digitale Akte, in der Gesundheitsdaten gespeichert werden können. Sie wird von der Krankenkasse bereitgestellt.",
            "source": "https://www.example.com/epa/"
        }
    ]
    
    # Das Verzeichnis erstellen falls nötig
    output_dir = os.path.dirname(config.EVALUATION_TEST_CASES_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Die Testfälle speichern
    with open(config.EVALUATION_TEST_CASES_PATH, 'w', encoding='utf-8') as file:
        json.dump(sample_cases, file, ensure_ascii=False, indent=2)
    
    print(f"Beispiel-Testfälle erstellt: {config.EVALUATION_TEST_CASES_PATH}")


# Dieser Code wird nur ausgeführt, wenn die Datei direkt gestartet wird
if __name__ == "__main__":
    # Prüfen ob Testfälle existieren
    if not os.path.exists(config.EVALUATION_TEST_CASES_PATH):
        print("Keine Testfälle gefunden. Erstelle Beispiel-Datei...")
        create_sample_test_cases()
    
    # Die Evaluation starten
    evaluator = EvaluationManager()
    results = evaluator.run_evaluation()
