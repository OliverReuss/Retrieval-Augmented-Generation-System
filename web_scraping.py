# web_scraping.py
# Dieses Modul ist für das Web Scraping der AOK-Webseiten zuständig
# Es nutzt Scrapy um die Seiten zu crawlen und den Text zu extrahieren

import json
import os
import re
from urllib.parse import urlparse

# Scrapy ist ein Framework für Web Scraping
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor

# Wir importieren unsere Konfiguration
import config


class WebScrapingManager:
    """
    Diese Klasse verwaltet das Web Scraping.
    Sie startet den Crawler und speichert die Ergebnisse.
    """
    
    def __init__(self):
        """
        Initialisiert den WebScrapingManager mit den Konfigurationswerten.
        """
        # Die Start-URL aus der Konfiguration holen
        self.start_url = config.SCRAPING_START_URL
        
        # Die maximale Crawling-Tiefe
        self.max_depth = config.SCRAPING_MAX_DEPTH
        
        # Die ausgeschlossenen Begriffe
        self.excluded_terms = config.SCRAPING_EXCLUDED_TERMS
        
        # Der Ausgabepfad für die JSON-Datei
        self.output_path = config.SCRAPING_OUTPUT_PATH
        
        # Wartezeit zwischen Anfragen
        self.download_delay = config.SCRAPING_DOWNLOAD_DELAY
        
        # Proxy-Einstellung (optional für Firmennetzwerke)
        self.proxy = getattr(config, 'SCRAPING_PROXY', None)
        
        # Hier werden die gescrapten Daten gespeichert
        self.scraped_data = []
        
    def start_scraping(self):
        """
        Startet den Scraping-Prozess.
        Erstellt einen Scrapy Crawler und führt ihn aus.
        """
        # Zuerst erstellen wir das Ausgabeverzeichnis, falls es nicht existiert
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Verzeichnis erstellt: {output_dir}")
        
        # Wir konfigurieren den Scrapy Crawler
        # Diese Einstellungen steuern das Verhalten des Crawlers
        process_settings = {
            # Respektiert die robots.txt der Webseite
            'ROBOTSTXT_OBEY': True,
            
            # Wartezeit zwischen Anfragen (höflich sein)
            'DOWNLOAD_DELAY': self.download_delay,
            
            # User-Agent setzen (identifiziert unseren Crawler)
            'USER_AGENT': 'AOK-RAG-Bot/1.0 (Bildungsprojekt)',
            
            # Logging-Level (INFO zeigt wichtige Meldungen)
            'LOG_LEVEL': 'INFO',
            
            # Maximale Tiefe für das Crawling
            'DEPTH_LIMIT': self.max_depth,
            
            # Wir deaktivieren einige Standard-Middlewares die wir nicht brauchen
            'COOKIES_ENABLED': False,
            
            # Retry-Einstellungen für instabile Verbindungen
            'RETRY_TIMES': 3,
            'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        }
        
        # Proxy-Einstellungen hinzufügen falls konfiguriert
        if self.proxy:
            # Setze Umgebungsvariablen für den Proxy (wird von Scrapy/Twisted genutzt)
            os.environ['http_proxy'] = self.proxy
            os.environ['https_proxy'] = self.proxy
            os.environ['HTTP_PROXY'] = self.proxy
            os.environ['HTTPS_PROXY'] = self.proxy
            print(f"Proxy konfiguriert: {self.proxy}")
        
        # Wir erstellen den Crawler-Prozess mit unseren Einstellungen
        process = CrawlerProcess(settings=process_settings)
        
        # Wir starten den Spider mit unseren Parametern
        # Der Spider ist die Klasse, die das eigentliche Crawling macht
        process.crawl(
            AOKSpider,
            start_url=self.start_url,
            excluded_terms=self.excluded_terms,
            scraped_data=self.scraped_data,
            proxy=self.proxy  # Proxy an Spider übergeben
        )
        
        # Hier startet das eigentliche Crawling
        # Der Prozess blockiert bis alles fertig ist
        print("Starte Web Scraping...")
        print(f"Start-URL: {self.start_url}")
        print(f"Maximale Tiefe: {self.max_depth}")
        if self.proxy:
            print(f"Verwende Proxy: {self.proxy}")
        else:
            print("Kein Proxy konfiguriert (direkte Verbindung)")
        process.start()
        
        # Nach dem Crawling speichern wir die Daten
        self.save_data()
        
        # Wir geben die Statistiken zurück
        return self.get_statistics()
    
    def save_data(self):
        """
        Speichert die gescrapten Daten als JSON-Datei.
        """
        # Wir öffnen die Datei zum Schreiben
        with open(self.output_path, 'w', encoding='utf-8') as file:
            # json.dump schreibt die Daten als JSON
            # ensure_ascii=False erlaubt deutsche Umlaute
            # indent=2 macht die Datei lesbar
            json.dump(self.scraped_data, file, ensure_ascii=False, indent=2)
        
        print(f"Daten gespeichert in: {self.output_path}")
        print(f"Anzahl der Einträge: {len(self.scraped_data)}")
    
    def get_statistics(self):
        """
        Berechnet Statistiken über die gescrapten Daten.
        Diese werden später für die Evaluation benötigt.
        """
        # Gesamtanzahl der Elemente
        total_elements = len(self.scraped_data)
        
        # Anzahl der Elemente mit Text (nicht leer)
        elements_with_text = 0
        text_lengths = []
        all_texts = []
        
        # Wir gehen durch alle gescrapten Elemente
        for item in self.scraped_data:
            text = item.get('text', '')
            if text and len(text.strip()) > 0:
                elements_with_text = elements_with_text + 1
                text_lengths.append(len(text))
                all_texts.append(text)
        
        # Berechne die Erfolgsrate
        if total_elements > 0:
            extraction_success_rate = elements_with_text / total_elements
        else:
            extraction_success_rate = 0.0
        
        # Berechne die Duplikatrate (gleiche Texte)
        unique_texts = set(all_texts)
        if len(all_texts) > 0:
            duplicate_rate = 1 - (len(unique_texts) / len(all_texts))
        else:
            duplicate_rate = 0.0
        
        # Berechne die durchschnittliche Textlänge
        if len(text_lengths) > 0:
            average_text_length = sum(text_lengths) / len(text_lengths)
        else:
            average_text_length = 0.0
        
        # Wir erstellen ein Dictionary mit allen Statistiken
        statistics = {
            'webscraping_elements_total': total_elements,
            'webscraping_elements_with_text': elements_with_text,
            'webscraping_extraction_success_rate': round(extraction_success_rate, 4),
            'webscraping_duplicate_rate_by_text': round(duplicate_rate, 4),
            'webscraping_average_text_length': round(average_text_length, 2)
        }
        
        return statistics


class AOKSpider(scrapy.Spider):
    """
    Der Spider ist die Scrapy-Komponente, die das eigentliche Crawling macht.
    Er besucht die Webseiten und extrahiert den Text.
    """
    
    # Name des Spiders (wird von Scrapy benötigt)
    name = 'aok_spider'
    
    def __init__(self, start_url, excluded_terms, scraped_data, proxy=None, *args, **kwargs):
        """
        Initialisiert den Spider mit den übergebenen Parametern.
        """
        super(AOKSpider, self).__init__(*args, **kwargs)
        
        # Die Start-URL als Liste (Scrapy erwartet eine Liste)
        self.start_urls = [start_url]
        
        # Die Domain aus der URL extrahieren
        # Das brauchen wir um nur auf aok.de zu bleiben
        parsed_url = urlparse(start_url)
        self.allowed_domains = [parsed_url.netloc]
        
        # Die ausgeschlossenen Begriffe speichern
        self.excluded_terms = excluded_terms
        
        # Referenz auf die Liste für die gescrapten Daten
        self.scraped_data = scraped_data
        
        # Proxy-URL speichern (für Firmennetzwerke)
        self.proxy = proxy
        
        # Link Extractor erstellen - findet alle Links auf einer Seite
        self.link_extractor = LinkExtractor(
            allow_domains=['aok.de'],
            deny_extensions=['pdf', 'jpg', 'png', 'gif', 'css', 'js']
        )
    
    def start_requests(self):
        """
        Überschreibt die Standard-Methode um Proxy für Start-Requests zu setzen.
        """
        for url in self.start_urls:
            if self.proxy:
                yield scrapy.Request(url, callback=self.parse, meta={'proxy': self.proxy})
            else:
                yield scrapy.Request(url, callback=self.parse)
    
    def parse(self, response):
        """
        Diese Methode wird für jede besuchte Seite aufgerufen.
        Sie extrahiert den Text und findet neue Links.
        """
        # Zuerst prüfen wir, ob die URL erlaubt ist
        current_url = response.url
        
        # Prüfen ob "aok.de" in der URL ist
        if 'aok.de' not in current_url:
            # Diese URL ignorieren wir
            return
        
        # Prüfen ob einer der ausgeschlossenen Begriffe in der URL ist
        url_lower = current_url.lower()
        should_exclude = False
        for term in self.excluded_terms:
            if term in url_lower:
                should_exclude = True
                break
        
        if should_exclude:
            # Diese URL wird ausgeschlossen
            self.logger.info(f"URL ausgeschlossen: {current_url}")
            return
        
        # Jetzt extrahieren wir den Text von der Seite
        # Wir nutzen CSS-Selektoren um nur den Hauptinhalt zu bekommen
        
        # Diese Selektoren versuchen den Hauptinhalt zu finden
        # und Navigations-Elemente auszuschließen
        main_content_selectors = [
            'main ::text',
            'article ::text',
            '.content ::text',
            '#content ::text',
            '.main-content ::text',
            '#main-content ::text',
            '.article-content ::text',
        ]
        
        # Elemente die wir ausschließen wollen (Navigation, Footer, etc.)
        exclude_selectors = [
            'nav',
            'header',
            'footer',
            '.navigation',
            '.nav',
            '.menu',
            '.sidebar',
            '.footer',
            '.header',
            '#nav',
            '#navigation',
            '#footer',
            '#header',
            '.cookie-banner',
            '.breadcrumb',
        ]
        
        # Wir versuchen den Hauptinhalt zu extrahieren
        extracted_text = ""
        
        # Zuerst versuchen wir die spezifischen Content-Selektoren
        for selector in main_content_selectors:
            texts = response.css(selector).getall()
            if texts:
                # Wir haben Text gefunden
                extracted_text = ' '.join(texts)
                break
        
        # Falls kein spezifischer Content gefunden wurde,
        # nehmen wir den Body-Text aber ohne die ausgeschlossenen Elemente
        if not extracted_text:
            # Wir holen den gesamten Body
            body_selector = response.css('body')
            
            if body_selector:
                # Wir entfernen die unerwünschten Elemente
                for exclude in exclude_selectors:
                    body_selector.css(exclude).remove()
                
                # Jetzt extrahieren wir den Text
                texts = response.css('body ::text').getall()
                extracted_text = ' '.join(texts)
        
        # Text bereinigen
        cleaned_text = self.clean_text(extracted_text)
        
        # Den Titel der Seite holen
        title = response.css('title::text').get()
        if title:
            title = title.strip()
        else:
            title = ""
        
        # Die Daten speichern
        page_data = {
            'url': current_url,
            'title': title,
            'text': cleaned_text
        }
        
        # Zur Liste hinzufügen
        self.scraped_data.append(page_data)
        self.logger.info(f"Seite verarbeitet: {current_url}")
        
        # Jetzt suchen wir nach neuen Links auf der Seite
        # und folgen ihnen (bis zur maximalen Tiefe)
        links = self.link_extractor.extract_links(response)
        
        for link in links:
            link_url = link.url
            
            # Prüfen ob die URL erlaubt ist
            if 'aok.de' not in link_url:
                continue
            
            # Prüfen ob einer der ausgeschlossenen Begriffe in der URL ist
            link_lower = link_url.lower()
            link_excluded = False
            for term in self.excluded_terms:
                if term in link_lower:
                    link_excluded = True
                    break
            
            if not link_excluded:
                # Diesem Link folgen (mit Proxy falls konfiguriert)
                if self.proxy:
                    yield response.follow(link_url, callback=self.parse, meta={'proxy': self.proxy})
                else:
                    yield response.follow(link_url, callback=self.parse)
    
    def clean_text(self, text):
        """
        Bereinigt den extrahierten Text.
        Entfernt überflüssige Leerzeichen und Zeilenumbrüche.
        """
        if not text:
            return ""
        
        # Mehrfache Leerzeichen durch einzelnes Leerzeichen ersetzen
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Führende und folgende Leerzeichen entfernen
        cleaned = cleaned.strip()
        
        return cleaned


def load_scraped_data():
    """
    Hilfsfunktion um die gescrapten Daten aus der JSON-Datei zu laden.
    Wird von anderen Modulen genutzt.
    """
    # Pfad aus der Konfiguration holen
    file_path = config.SCRAPING_OUTPUT_PATH
    
    # Prüfen ob die Datei existiert
    if not os.path.exists(file_path):
        print(f"Fehler: Datei nicht gefunden: {file_path}")
        print("Bitte zuerst das Web Scraping ausführen.")
        return []
    
    # Datei öffnen und JSON laden
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    print(f"Gescrapte Daten geladen: {len(data)} Einträge")
    return data


# Dieser Code wird nur ausgeführt, wenn die Datei direkt gestartet wird
if __name__ == "__main__":
    # Zum Testen können wir das Scraping direkt starten
    scraper = WebScrapingManager()
    stats = scraper.start_scraping()
    print("\nStatistiken:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
