from web_scraping import Web_Scaper
from chunking import Chunker
from embedding import Embedder
from retrieval import Retriever
from generation import Generator
from evaluation import Evaluator

web_scraper = Web_Scaper()
web_scraper.process()
chunker = Chunker()
chunker.process()
embedder = Embedder()
embedder.process()
retriever = Retriever()
generator = Generator()
"""
evaluator = Evaluator()
evaluator.process()
"""

while True:
    query = input("Frage eingeben: ")
    search_results = retriever.search(query)
    answer = generator.generate(query)
    print(answer)
