from web_scraping import Web_Scraper
from chunking import Chunker
from embedding import Embedder
from retrieval import Retriever
from generation import Generator
from evaluation import Evaluator

web_scraper = Web_Scraper()
print("A")
chunker = Chunker()
print("B")
embedder = Embedder()
print("C")
retriever = Retriever()
print("D")
generator = Generator(retriever)
print("E")
evaluator = Evaluator(web_scraper, chunker, embedder, retriever, generator)

while True:
    query = input("\n➡️ Frage eingeben: ")
    search_results = retriever.search(query)
    answer = generator.generate(query)
    print(f"\n{answer['answer']}")
