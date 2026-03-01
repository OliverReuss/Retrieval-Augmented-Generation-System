from web_scraping import WebScraper
from chunking import Chunker
from embedding import Embedder
from retrieval import Retriever
from generation import Generator
from evaluation import Evaluator

web_scraper = WebScraper()
chunker = Chunker()
embedder = Embedder()
retriever = Retriever()
generator = Generator(retriever)
if input("\n➡️ Evaluierung durchführen? (j/n): ").strip().lower() == 'j':
        evaluator = Evaluator(web_scraper, chunker, embedder, retriever, generator)

while True:
    query = input("\n➡️ Frage eingeben: ")
    answer = generator.generate(query)
    print(f"\n{answer['answer']}")
    print(f"\nQuellen")
    for i, source in enumerate(answer['sources'], start=1):
        print(f"{i}. {source}")