import json

with open("test_cases.json", "r", encoding="utf-8") as file:
    test_cases = json.load(file)

sources = set()

for test_case in test_cases:
    sources.add(test_case['source'])

with open("data/documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

document_sources = set()

for document in documents:
    document_sources.add(document['url'])

for source in sources:
    if source not in document_sources:
        print(f"Nicht enthalten: {source}")

"""
Nicht enthalten: https://www.example.com/chronische-erkrankungen/diabetes/
Nicht enthalten: https://bluthochdruck.aok.de/unsere-experten/
Nicht enthalten: https://www.example.com/thema/66-tage-challenge/
Nicht enthalten: https://www.example.com/leistungen/studium-beruf/studierendenservice/
"""