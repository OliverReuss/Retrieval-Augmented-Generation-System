from web_scraping import Web_Scraper
from chunking import Chunker
from embedding import Embedder
from retrieval import Retriever
from generation import Generator
# from evaluation import Evaluator

web_scraper = Web_Scraper()
chunker = Chunker()
embedder = Embedder()
retriever = Retriever()
generator = Generator()
# evaluator = Evaluator()

while True:
    query = input("\n➡️ Frage eingeben: ")
    search_results = retriever.search(query)
    answer = generator.generate(query)
    print(answer)
