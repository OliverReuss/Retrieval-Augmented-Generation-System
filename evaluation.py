from datasets import Dataset
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (context_entity_recall, context_precision, context_recall, answer_correctness, answer_relevancy, answer_similarity, faithfulness)
from ragas.metrics._summarization import SummarizationScore
import config
import json

class Evaluator:

    def __init__(self, web_scraper, chunker, embedder, retriever, generator) -> None:
        self.test_cases_path = config.EVALUATION_TEST_CASES_PATH
        self.model = config.EVALUATION_MODEL
        self.embedding_model = config.EVALUATION_EMBEDDING_MODEL
        self.web_scraper = web_scraper
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.test_cases = []
        self.results = []
        self.llm = None
        self.embeddings = None

        # Automatisch starten
        self.process()
    
    def setup_ragas_llm(self) -> None:
        # LLM mit LangchainLLMWrapper erstellen
        base_llm = ChatOpenAI(
            max_completion_tokens=8192,
            http_client=None,
            http_async_client=None,
            timeout=60,
            model=self.model,
            temperature=0.0,
            n=1,
        )
        self.llm = LangchainLLMWrapper(base_llm)
        
        # Embeddings mit LangchainEmbeddingsWrapper erstellen
        base_embeddings = OpenAIEmbeddings(
            http_client=None,
            http_async_client=None,
            timeout=60,
            model=self.embedding_model
        )
        self.embeddings = LangchainEmbeddingsWrapper(base_embeddings)
    
    def load_test_cases(self) -> list[dict]:      
        with open(self.test_cases_path, 'r', encoding='utf-8') as file:
            self.test_cases = json.load(file)
        return self.test_cases
    
    def run_single_test(self, test_case: dict, index: int) -> dict:
        test_type = test_case.get('type', '')
        prompt = test_case.get('prompt', '')
        truth = test_case.get('truth', '')
        source = test_case.get('source', '')
        
        response = self.generator.generate(prompt)

        # Ergebnisse runden
        rounded_search_results = []
        for sr in response['search_results']:
            rounded_sr = sr.copy()
            rounded_sr['similarity'] = round(rounded_sr['similarity'], 4)
            rounded_sr['rerank_score'] = round(rounded_sr['rerank_score'], 4)
            rounded_search_results.append(rounded_sr)

        result = {
            'index': index,
            'type': test_type,
            'prompt': prompt,
            'truth': truth,
            'source': source,
            'context': response['context'],
            'sources': response['sources'],
            'search_results': rounded_search_results,  # Gerundete Version
            'response': response['answer'],
            'metrics': {}
        }
        
        return result
    
    def evaluate_ragas(self, prompt: str, response: str, context: list[str], truth: str, test_type: str) -> dict:
        data = {
            "question": [prompt],
            "answer": [response],
            "contexts": [context],
            "retrieved_contexts": [context],
            "reference_contexts": [context],
            "ground_truths": [[truth]],
            "reference": [truth]
        }
        
        # Dict zu Dataset umwandeln
        dataset = Dataset.from_dict(data)
        
        # Metriken in Gruppen definieren
        summary_score = SummarizationScore()
        
        # Metriken passend zu Test-Typ zuweisen, da nicht alle für jeden Test-Typ relevant
        metrics_to_run = []
        
        if test_type in ["retrieval: exact match", "retrieval: synonym/paraphrase"]:
            metrics_to_run = [context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness]
        elif test_type == "retrieval: noise":
            metrics_to_run = [context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness]
        elif test_type == "generation: summary":
            metrics_to_run = [summary_score, faithfulness, answer_correctness, answer_similarity]
        elif test_type == "generation: transformation":
            metrics_to_run = [answer_correctness, answer_relevancy, answer_similarity, faithfulness]
        elif test_type == "robustness: not in context":
            metrics_to_run = [answer_correctness, answer_similarity]
        elif test_type == "robustness: ambiguous":
            metrics_to_run = [answer_correctness, answer_relevancy, answer_similarity]
        elif test_type == "hallucination: misleading":
            metrics_to_run = [faithfulness, answer_correctness, answer_similarity, context_precision]
        else:
            metrics_to_run = [context_precision, context_recall, answer_correctness, answer_similarity, faithfulness]
        
        try:
            # RAGAS Evaluation ausführen
            ragas_results = evaluate(
                dataset=dataset,
                metrics=metrics_to_run,
                llm=self.llm,
                embeddings=self.embeddings,
                show_progress=False
            )
            return ragas_results.scores[0]
        except Exception as e:
            print(f"\n❗ Bei der Evaluation des Testfalls mit RAGAS ist ein Fehler aufgetreten: {e}")
            return {}

    def calculate_metrics(self, results: list[dict]) -> list[dict]:
        self.setup_ragas_llm()

        # Für jeden Testfall
        for i, result in enumerate(results):
            print(f"Evaluation | Berechne Metriken für Testfall: {i + 1}/{len(results)}")
            
            prompt = result['prompt']
            response = result['response']
            context = result['context']
            truth = result['truth']
            test_type = result['type']
            
            # RAGAS Evaluation für Testfall
            ragas_scores = self.evaluate_ragas(prompt, response, context, truth, test_type)
            
            metrics = {}
            
            # Metriken zum Ergebnis hinzufügen
            for metric_name, value in ragas_scores.items():
                if value is not None:
                    # NaN-Werte ignorieren
                    if isinstance(value, float) and value != value:
                        continue
                    else:
                        # Auf 4 Nachkommastellen runden
                        metrics[metric_name] = round(float(value), 4)
            
            result['metrics'] = metrics
        
        return results
    
    def calculate_averages(self, results: list[dict]) -> dict:

        # Alle Metrik-Namen sammeln
        all_metric_names = set()
        for result in results:
            metrics = result.get('metrics', {})
            for name in metrics.keys():
                all_metric_names.add(name)
        
        # Durchschnitte berechnen
        averages = {}
        
        for metric_name in all_metric_names:
            values = []
            for result in results:
                metrics = result.get('metrics', {})
                if metric_name in metrics:
                    value = metrics[metric_name]
                    # Prüfen ob Wert Zahl ist
                    if isinstance(value, (int, float)) and not (value != value):
                        values.append(value)
            
            # Durchschnitt berechnen
            if values:
                average = sum(values) / len(values)
                averages[metric_name] = round(average, 4)
            else:
                averages[metric_name] = 0.0
        
        return averages
    
    def get_config_parameters(self) -> dict:
        parameters = {
            'WEB_SCRAPING_START_URL': config.WEB_SCRAPING_START_URL,
            'WEB_SCRAPING_MAX_DEPTH': config.WEB_SCRAPING_MAX_DEPTH,
            'WEB_SCRAPING_EXCLUDED_TERMS': config.WEB_SCRAPING_EXCLUDED_TERMS,
            'WEB_SCRAPING_DOWNLOAD_DELAY': config.WEB_SCRAPING_DOWNLOAD_DELAY,
            
            'CHUNKING_CHUNK_SIZE': config.CHUNKING_CHUNK_SIZE,
            'CHUNKING_CHUNK_OVERLAP': config.CHUNKING_CHUNK_OVERLAP,
            
            'EMBEDDING_MODEL': config.EMBEDDING_MODEL,
            'EMBEDDING_BATCH_SIZE': config.EMBEDDING_BATCH_SIZE,
            
            'RETRIEVAL_TOP_K': config.RETRIEVAL_TOP_K,
            'RETRIEVAL_HNSW_SPACE': config.RETRIEVAL_HNSW_SPACE,
            'RETRIEVAL_HNSW_M': config.RETRIEVAL_HNSW_M,
            'RETRIEVAL_CROSS_ENCODER_MODEL': config.RETRIEVAL_CROSS_ENCODER_MODEL,
            'RETRIEVAL_RERANKING_TOP_K': config.RETRIEVAL_RERANKING_TOP_K,
            'RETRIEVAL_RERANKING_THRESHOLD': config.RETRIEVAL_RERANKING_THRESHOLD,
            
            'GENERATION_API_URL': config.GENERATION_API_URL,
            'GENERATION_MODEL': config.GENERATION_MODEL,
            'GENERATION_TEMPERATURE': config.GENERATION_TEMPERATURE,
            'GENERATION_MAX_TOKENS': config.GENERATION_MAX_TOKENS,
            'GENERATION_TOP_P': config.GENERATION_TOP_P,
            'GENERATION_FREQUENCY_PENALTY': config.GENERATION_FREQUENCY_PENALTY,
            'GENERATION_SEED': config.GENERATION_SEED,
            'GENERATION_TIMEOUT': config.GENERATION_TIMEOUT,
            'GENERATION_SYSTEM_PROMPT': config.GENERATION_SYSTEM_PROMPT,

            'EVALUATION_MODEL': config.EVALUATION_MODEL,
            'EVALUATION_EMBEDDING_MODEL': config.EVALUATION_EMBEDDING_MODEL
        }

        return parameters
    
    def collect_statistics(self) -> dict:
        statistics = {}
        
        web_stats = self.web_scraper.get_statistics()
        statistics.update(web_stats)

        chunk_stats = self.chunker.get_statistics()
        statistics.update(chunk_stats)
    
        embed_stats = self.embedder.get_statistics()
        statistics.update(embed_stats)
    
        retrieval_stats = self.retriever.get_statistics()
        statistics.update(retrieval_stats)
        self.retriever.reset_statistics()
    
        generation_stats = self.generator.get_statistics()
        statistics.update(generation_stats)
        self.generator.reset_statistics()

        return statistics

    def process(self) -> dict:
        print("\nℹ️ Starte Evaluation")
        
        test_cases = self.load_test_cases()
        
        results = []
        
        # Einzelne Testfälle ausführen
        for i, test_case in enumerate(test_cases):
            print(f"Evaluation | Generiere Antwort für Testfall: {i+1}/{len(test_cases)}")
            result = self.run_single_test(test_case, i)
            results.append(result)
        
        # RAGAS Evaluation für alle Testfälle ausführen
        results = self.calculate_metrics(results)
        # Statistiken sammeln
        statistics = self.collect_statistics()
        # Durchschnitte berechnen
        averages = self.calculate_averages(results)
        # Konfiguration speichern
        config_params = self.get_config_parameters()

        # Zeitstempel für eindeutige Dateinamen hinzufügen
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_path = f"evalation_results_{timestamp}.json"
        
        final_results = {
            'evaluation_timestamp': timestamp,
            'configuration': config_params,
            'statistics': statistics,
            'metric_averages': averages,
            'test_results': results
        }
        
        self.save_results(final_results)
        print(f"\nℹ️ Evaluation. Verarbeitete Testfälle: {len(final_results['test_results'])}.")
        return final_results
    
    def save_results(self, results: dict) -> None:      
        with open(self.output_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
