# download_model.py
# Führe dieses Script auf einem Rechner OHNE Proxy-Einschränkungen aus
# (z.B. zu Hause oder mit mobilem Hotspot)
# Dann kopiere den Ordner "models/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2" 
# auf deinen Arbeitsrechner

from sentence_transformers import SentenceTransformer
import os

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
local_path = f"models/{model_name.replace('/', '_')}"

print(f"Lade Modell: {model_name}")
model = SentenceTransformer(model_name)

print(f"Speichere Modell nach: {local_path}")
os.makedirs("models", exist_ok=True)
model.save(local_path)

print("Fertig! Kopiere den Ordner 'models' auf deinen Arbeitsrechner.")
