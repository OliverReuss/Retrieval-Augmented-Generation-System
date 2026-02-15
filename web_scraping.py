from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urlparse
import config
import json
import os
import re
import scrapy

class Web_Scraper:

    def __init__(self) -> None:
        self.start_url = config.WEB_SCRAPING_START_URL
        self.max_depth = config.WEB_SCRAPING_MAX_DEPTH
        self.excluded_terms = config.WEB_SCRAPING_EXCLUDED_TERMS
        self.output_path = config.WEB_SCRAPING_OUTPUT_PATH
        self.download_delay = config.WEB_SCRAPING_DOWNLOAD_DELAY
        self.proxy = getattr(config, 'PROXY', None)
        self.scraped_data = []

        # Automatisch starten
        self.process()
        
    def process(self) -> None:
        if os.path.exists(self.output_path):
            return
        
        print("ℹ️ Starte Web Scraping")
    
        process_settings = {
            'ROBOTSTXT_OBEY': True,
            'DOWNLOAD_DELAY': self.download_delay,
            'USER_AGENT': 'RAG-Bot/1.0',
            'LOG_LEVEL': 'WARNING',
            'DEPTH_LIMIT': self.max_depth,
            'COOKIES_ENABLED': False,
            'RETRY_TIMES': 3,
            'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
            'STATS_DUMP': False
        }
        
        # Crawler-Prozess erstellen
        process = CrawlerProcess(settings=process_settings)
        
        # Spider für eigentliches Web Scraping erstellen
        process.crawl(
            Spider,
            start_url=self.start_url,
            excluded_terms=self.excluded_terms,
            scraped_data=self.scraped_data,
            proxy=self.proxy
        )

        process.start()
        self.save_documents()
        print(f"\nℹ️ Web Scraping abgeschlossen. Verarbeitete Seiten: {len(self.scraped_data)}.")
    
    def save_documents(self) -> None:
        dir_name = os.path.dirname(self.output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as file:
            json.dump(self.scraped_data, file, ensure_ascii=False, indent=2)
    
    def get_statistics(self) -> dict:
        # Daten laden, falls nicht im Speicher
        if not self.scraped_data and os.path.exists(self.output_path):
            self.scraped_data = load_documents()
        
        total_elements = len(self.scraped_data)
        elements_with_text = 0
        text_lengths = []
        all_texts = []
        
        # Alle Dokumente durchlaufen
        for item in self.scraped_data:
            text = item.get('text', '')
            if text and len(text.strip()) > 0:
                elements_with_text = elements_with_text + 1
                text_lengths.append(len(text))
                all_texts.append(text)
        
        # Erfolgsrate
        extraction_success_rate = elements_with_text / total_elements
        
        # Duplikatrate
        unique_texts = set(all_texts)
        duplicate_rate = 1 - (len(unique_texts) / len(all_texts))
        
        # Durchschnittliche Textlänge
        average_text_length = sum(text_lengths) / len(text_lengths)
        
        statistics = {
            'web_scraping_elements_total': total_elements,
            'web_scraping_elements_with_text': elements_with_text,
            'web_scraping_extraction_success_rate': round(extraction_success_rate, 2),
            'web_scraping_duplicate_rate_by_text': round(duplicate_rate, 2),
            'web_scraping_average_text_length': round(average_text_length, 2)
        }
        return statistics

class Spider(scrapy.Spider):

    # Scrapy benötigt Name
    name = "spider"

    def __init__(self, start_url, excluded_terms: list[str], scraped_data: list, proxy=None, *args, **kwargs) -> None:
        super(Spider, self).__init__(*args, **kwargs)
        self.start_url = start_url
        # Auf richtige Domain prüfen
        parsed_url = urlparse(start_url)
        self.allowed_domain = parsed_url.netloc
        self.excluded_terms = excluded_terms
        self.scraped_data = scraped_data
        self.proxy = proxy
        # Link Extractor extrahiert Links auf Seite
        self.link_extractor = LinkExtractor(allow_domains=['aok.de'], deny_extensions=['pdf', 'jpg', 'png', 'gif', 'css', 'js'])
    
    def start_requests(self):
        # Überschreibt Standard-Methode für Proxy-Nutzung
        if self.proxy:
            yield scrapy.Request(self.start_url, callback=self.parse, meta={'proxy': self.proxy})
        else:
            yield scrapy.Request(self.start_url, callback=self.parse)
    
    def parse(self, response):
        current_url = response.url
        
        # Prüfen, ob URL erlaubt
        if self.allowed_domain not in current_url:
            return
        
        # Auf ausgeschlossenen Begeriff prüfen
        url_lower = current_url.lower()
        should_exclude = False
        for term in self.excluded_terms:
            if term in url_lower:
                should_exclude = True
                break
        
        if should_exclude:
            print(f"Web Scraping | URL ausgeschlossen: {current_url}")
            return
        
        # Text mittels CSS-Selektoren aus Hauptinhalt extrahieren
        
        # Selektoren die Inhalt auswählen
        main_content_selectors = [
            'main ::text',
            'article ::text',
            '.content ::text',
            '#content ::text',
            '.main-content ::text',
            '#main-content ::text',
            '.article-content ::text',
        ]
        
        # Selektoren die ausgeschlossen werden (Navigation, Footer etc.)
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
        
        extracted_text = ""
        
        # Selektoren versuchen
        for selector in main_content_selectors:
            texts = response.css(selector).getall()
            if texts:
                extracted_text = ' '.join(texts)
                break
        
        # Body nutzen, falls keine Inhalt gefunden
        if not extracted_text:
            body_selector = response.css('body')
            if body_selector:
                # Unerwünschte Selektoren entfernen
                for exclude in exclude_selectors:
                    body_selector.css(exclude).remove()
                texts = response.css('body ::text').getall()
                extracted_text = ' '.join(texts)
        
        cleaned_text = self.clean_text(extracted_text)
        
        title = response.css('title::text').get()
        if title:
            title = title.strip()
        else:
            title = ""

        page_data = {
            'url': current_url,
            'title': title,
            'text': cleaned_text
        }
        
        self.scraped_data.append(page_data)
        print(f"Web Scraping | Seite verarbeitet: {current_url}")
        
        # Links auf Seite suchen und verarbeiten
        links = self.link_extractor.extract_links(response)
        
        for link in links:
            link_url = link.url
            
            # Prüfen ob die URL erlaubt ist
            if self.allowed_domain not in link_url:
                continue
            
            # Auf ausgeschlossene Begriffe prüfen
            link_lower = link_url.lower()
            link_excluded = False
            for term in self.excluded_terms:
                if term in link_lower:
                    link_excluded = True
                    break
            
            # Link folgen
            if not link_excluded:
                if self.proxy:
                    yield response.follow(link_url, callback=self.parse, meta={'proxy': self.proxy})
                else:
                    yield response.follow(link_url, callback=self.parse)
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Mehrfache Leerzeichen entfernen
        cleaned = re.sub(r'\s+', ' ', text)
        # Leerezichen am Anfang und Ende entfernen
        cleaned = cleaned.strip()
        return cleaned

def load_documents() -> list[dict]:
    file_path = config.WEB_SCRAPING_OUTPUT_PATH
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data