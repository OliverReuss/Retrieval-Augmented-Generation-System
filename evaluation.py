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
import os

class Evaluator:

    def __init__(self, web_scraper, chunker, embedder, retriever, generator) -> None:
        self.test_cases_path = config.EVALUATION_TEST_CASES_PATH
        self.output_path = config.EVALUATION_OUTPUT_PATH
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
        if config.EVALUATION_EVALUATE:
            self.process()
    
    def setup_ragas_llm(self) -> None:
        # LLM mit LangchainLLMWrapper erstellen
        base_llm = ChatOpenAI(
            max_completion_tokens=8192,
            http_client=None,
            http_async_client=None,
            timeout=60,
            model="gpt-4.1"
        )
        self.llm = LangchainLLMWrapper(base_llm)
        
        # Embeddings mit LangchainEmbeddingsWrapper erstellen
        base_embeddings = OpenAIEmbeddings(
            http_client=None,
            http_async_client=None,
            timeout=60
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

        result = {
            'index': index,
            'type': test_type,
            'prompt': prompt,
            'truth': truth,
            'source': source,
            'context': response['context'],
            'sources': response['sources'],
            'search_results': response['search_results'],
            'response': response['answer'],
            'metrics': {}
        }
        
        return result
    
    def evaluate_ragas(self, prompt: str, response: str, context: str, truth: str) -> list[dict]:
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
        
        # Summarization Score erstellen
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

    def calculate_metrics(self, results: list[dict]) -> list[dict]:
        self.setup_ragas_llm()

        # Für jeden Testfall
        for i, result in enumerate(results):
            print(f"Evaluation | Testfall: {i + 1}/{len(results)}")
            
            prompt = result['prompt']
            response = result['response']
            context = result['context']
            truth = result['truth']
            
            # RAGAS Evaluation für Testfall
            ragas_scores = self.evaluate_ragas(prompt, response, context, truth)
            
            metrics = {}
            
            # Metriken zum Ergebnis hinzufügen
            for metric_name, value in ragas_scores.items():
                if value is not None:
                    metrics[metric_name] = float(value) if not isinstance(value, (int, float)) else value
            
            result['metrics'] = metrics
        
        # Nicht-RAGAS Metriken berechnen
        self.calculate_additional_metrics(results)
        return results
    
    def calculate_additional_metrics(self, results: list[dict]) -> None:
        for result in results:
            metrics = result.get('metrics', {})
            
            # Summarization Score
            context = result.get('context', '')
            response = result.get('response', '')
            
            if context and response:
                # Wie viele Wörter aus Kontext sind in Antwort?
                context_words = set(context.lower().split())
                response_words = set(response.lower().split())
                common_words = context_words.intersection(response_words)
                
                if len(context_words) > 0:
                    summarization_score = len(common_words) / len(context_words)
                else:
                    summarization_score = 0.0
                
                metrics['summarization_score'] = round(summarization_score, 4)
            else:
                metrics['summarization_score'] = 0.0
            
            # Platzhalter
            # ECE Score (Expected Calibration Error)
            metrics['ece_score'] = 0.0
            # Brier Score
            metrics['brier_score'] = 0.0
            # Discrimination Score
            metrics['discrimination'] = 0.0
            
            # Speichern
            result['metrics'] = metrics
    
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
            average = sum(values) / len(values)
            averages[metric_name] = round(average, 4)
        
        return averages
    
    def get_config_parameters(self) -> dict:
        parameters = {
            'web_scraping_start_url': config.WEB_SCRAPING_START_URL,
            'WEB_SCRAPING_MAX_DEPTH': config.WEB_SCRAPING_MAX_DEPTH,
            'WEB_SCRAPING_EXCLUDED_TERMS': config.WEB_SCRAPING_EXCLUDED_TERMS,
            'WEB_SCRAPING_DOWNLOAD_DELAY': config.WEB_SCRAPING_DOWNLOAD_DELAY,
            
            'CHUNKING_CHUNK_SIZE': config.CHUNKING_CHUNK_SIZE,
            'CHUNKING_CHUNK_OVERLAP': config.CHUNKING_CHUNK_OVERLAP,
            
            'EMBEDDING_MODEL_NAME': config.EMBEDDING_MODEL_NAME,
            'EMBEDDING_BATCH_SIZE': config.EMBEDDING_BATCH_SIZE,
            
            'RETRIEVAL_TOP_K': config.RETRIEVAL_TOP_K,
            'RETRIEVAL_HNSW_SPACE': config.RETRIEVAL_HNSW_SPACE,
            'RETRIEVAL_HNSW_M': config.RETRIEVAL_HNSW_M,
            'RETRIEVAL_USE_RERANKING': config.RETRIEVAL_USE_RERANKING,
            'RETRIEVAL_CROSS_ENCODER_MODEL': config.RETRIEVAL_CROSS_ENCODER_MODEL,
            'RETRIEVAL_RERANKING_TOP_K': config.RETRIEVAL_RERANKING_TOP_K,
            
            'GENERATION_API_URL': config.GENERATION_API_URL,
            'GENERATION_MODEL_NAME': config.GENERATION_MODEL_NAME,
            'GENERATION_TEMPERATURE': config.GENERATION_TEMPERATURE,
            'GENERATION_MAX_TOKENS': config.GENERATION_MAX_TOKENS,
            'GENERATION_TOP_P': config.GENERATION_TOP_P,
            'GENERATION_FREQUENCY_PENALTY': config.GENERATION_FREQUENCY_PENALTY,
            'GENERATION_SEED': config.GENERATION_SEED,
            'GENERATION_TIMEOUT': config.GENERATION_TIMEOUT,
            'GENERATION_SYSTEM_PROMPT': config.GENERATION_SYSTEM_PROMPT
        }

        return parameters
    
    def process(self) -> dict:
        print("\nℹ️ Starte Evaluation")
        
        test_cases = self.load_test_cases()

        # web_scrpaing_stats = self.web_scraper.get_statistics()
        # chunking_stats = self.chunker.get_statistics()
        #emebdding_stats = self.embedder.get_statistics()
        #retrieval_stats = self.retriever.get_statistics()
        #generation_stats = self.generator.get_statisitcs()
        
        results = []
        
        # Einzelne Testfälle ausführen
        for i, test_case in enumerate(test_cases):
            result = self.run_single_test(test_case, i)
            results.append(result)
        
        # RAGAS Evaluation für alle Testfälle ausführen
        results = self.calculate_metrics(results)

        averages = self.calculate_averages(results)
        config_params = self.get_config_parameters()
        
        final_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'configuration': config_params,
            #'web_scrpaing_stats': web_scrpaing_stats,
            #'chunking_stats': chunking_stats,
            #'emebdding_stats': emebdding_stats,
            #'retrieval_stats': retrieval_stats,
            #'generation_stats': generation_stats,
            'metric_averages': averages,
            'test_results': results
        }
        
        self.save_results(final_results)
        print(f"\nℹ️ Evaluation. Verarbeitete Testfälle: {len(final_results['test_results'])}.")
        return final_results
    
    def save_results(self, results: dict) -> None:
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(self.output_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
