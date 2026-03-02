# Praxisarbeit: Wenn Websites sprechen lernen: Implementierung eines Retrieval-Augmented-Generation-Systems auf Grundlage von Web-Scraping-Daten

## Beschreibung
Es wurde ein Retrieval-Augmented Generation (RAG) System auf Basis von Web-Scraping-Daten implementiert. Das Projekt enthält die komplette Pipeline von der Datenextraktion bis hin zur Nutzung und der automatisierten Evaluation.

## Voraussetzungen und Installation
- Python 3.11.9+
- Installation der Abhängigkeiten aus der requirements.txt mit dem Befehl `pip install -r requirements.txt`
- Ein OpenAI-Konto mit Guthaben wird für die Evaluation benötigt (ein kompletter Testlauf mit gpt-4o-mini kostet ca. 0,12 €)
- Hinterlegung des API-Keys in der `.env` (`OPENAI_API_KEY="VALUE"`)

## Nutzung
Das Ausführen von `main.py` startet den gesamten Prozess und legt alle benötigten Daten an bzw. lädt die benötigten Modelle herunter. Der Nutzer kann auswählen, ob eine Evaluation durchgeführt werden soll, oder ob er zur Nutzung übergeht. Die Nutzung erfolgt über Ein- und Ausgaben im Terminal.

## Konfiguration
Das System kann über `config.py` angepasst werden. Optional kann hier ein Proxy-Server und SSL-Zertifikat hinterlegt werden.

## Autor
Oliver Reuß (Mtr. Nr.: 368522)

Vorgelegt bei: Prof. Dr. Michael Spangenberg

Hochschule für Angewandte Wissenschaften Hof | Fakultät Informatik | Studiengang Medieninformatik | Wintersemester 2025/26