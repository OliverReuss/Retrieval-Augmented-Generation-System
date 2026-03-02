# Praxisarbeit: Wenn Websites sprechen lernen: Implementierung eines Retrieval-Augmented-Generation-Systems auf Grundlage von Web-Scraping-Daten

## Beschreibung

Es wurde ein Retrieval-Augmented Generation (RAG) System auf Basis von Web-Scraping-Daten implementiert. Das Projekt enthält die komplette Pipeline von der Datenextraktion bis hin zur Nutzung und der automatisierten Evaluation.

## Voraussetzungen und Installation

- Python>=3.11.9
- Empfohlen: Erstellung eines Virtual Environments mit `python -m venv .venv` und `.venv\Scripts\activate`
- Installation der Abhängigkeiten aus der requirements.txt mit dem Befehl `pip install -r requirements.txt`
- Ein OpenAI-Konto mit Guthaben wird für die Evaluation und Generierung benötigt und ein API-Key muss in der `.env` hinterlegt werden (`OPENAI_API_KEY=VALUE`) (Hinweis: ein kompletter Testlauf mit dem Modell `gpt-4o-mini` kostet ca. 0,12 €)
- Alternativ zu OpenAI kann für die Generierung (nicht Evaluation) auch ein kostenloses Modell von OpenRouter eingebunden werden (`OPENROUTER_API_KEY=VALUE`)
- Optional aber sehr empfohlen: ein Hugging Face Konto erstellen und einen API-Key für einen schnelleren Download von Modellen hinterlegen (`HF_TOKEN=VALUE`)

## Nutzung

Das Ausführen von `main.py` startet den gesamten Prozess und legt alle benötigten Daten an bzw. lädt die benötigten Modelle herunter. Der Nutzer kann auswählen, ob eine Evaluation durchgeführt werden soll, oder ob er zur Nutzung übergeht. Die Nutzung erfolgt über Ein- und Ausgaben im Terminal.

## Konfiguration

- Das System kann über `config.py` angepasst werden
- Proxy-Server (optional): `PROXY`
- SSL-Zertifikat (optional): `CERTIFICATE`
- LLM-Anbieter: `GENERATION_API_URL`
- LLM-Anbieter API-Key: `GENERATION_API_KEY`
- LLM-Modell: `GENERATION_MODEL`

## Infos

- Autor: Oliver Reuß (Mtr. Nr.: 368522)
- Vorgelegt bei: Prof. Dr. Michael Spangenberg
- Hochschule für Angewandte Wissenschaften Hof | Fakultät Informatik | Studiengang Medieninformatik | Wintersemester 2025/26
