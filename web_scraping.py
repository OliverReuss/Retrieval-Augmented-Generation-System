from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urlparse
from parsel import Selector
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
        
        print("\nℹ️ Starte Web Scraping")
    
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

        process.start(install_signal_handlers=False)
        self.save_documents()
        print(f"\nℹ️ Web Scraping abgeschlossen. Verarbeitete Dokumente: {len(self.scraped_data)}.")
    
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
        text_lengths = []
        all_texts = []
        all_urls = []
        
        # Alle Dokumente durchlaufen
        for item in self.scraped_data:
            text = item.get('text', '')
            url = item.get('url', '')
            if url:
                all_urls.append(url)
            if text and len(text.strip()) > 0:
                text_lengths.append(len(text))
                all_texts.append(text)
        
        # Duplikatrate
        unique_texts = set(all_texts)
        duplicate_rate = 1 - (len(unique_texts) / len(all_texts)) if all_texts else 0
        
        # Durchschnittliche Textlänge
        average_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        
        # Einzigartige URLs
        unique_pages = len(set(all_urls))

        statistics = {
            'web_scraping_unique_pages': unique_pages,
            'web_scraping_elements_total': total_elements,
            'web_scraping_duplicate_rate_by_text': round(duplicate_rate, 4),
            'web_scraping_average_text_length': round(average_text_length, 4)
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
        self.link_extractor = LinkExtractor(allow_domains=[self.allowed_domain], deny_extensions=['pdf', 'jpg', 'png', 'gif', 'css', 'js'])
        # HTML-Elemente für Überschriften und Content
        self.header_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        self.content_tags = ['p', 'div', 'li', 'span', 'td']
        # Selektoren die ausgeschlossen werden (Navigation, Footer etc.)
        self.excludes_selectors = [
            'nav', 'header', 'footer', 'aside',
            '.navigation', '.nav', '.menu', '.sidebar',
            '.footer', '.header', '.cookie-banner', '.breadcrumb',
            '#nav', '#navigation', '#footer', '#header',
            'script', 'style', 'noscript'
        ]
    
    async def start(self):
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
        
        # Auf ausgeschlossenen Begriff prüfen
        url_lower = current_url.lower()
        should_exclude = False
        for term in self.excluded_terms:
            if term in url_lower:
                should_exclude = True
                break
        
        if should_exclude:
            print(f"Web Scraping | Seite ausgeschlossen: {current_url}")
            return
        
        # Seitentitel extrahieren
        page_title = response.css('title::text').get()
        if page_title:
            page_title = page_title.strip()
        else:
            page_title = ""
        
        # Strukturierte Extraktion durchführen
        sections = self.extract_structured_content(response)
        
        # Jede Sektion als eigenes Dokument speichern
        for idx, section in enumerate(sections):
            if section['text'].strip():  # Nur nicht-leere Sektionen
                document = {
                    'url': current_url,
                    'page_title': page_title,
                    'section_header': section['header'],
                    'text': section['text'],
                }
                self.scraped_data.append(document)
        
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
    
    def extract_structured_content(self, response) -> list[dict]:
        sections = []
        
        # Hauptinhalt finden
        main_content = None
        for selector in ['main', 'article', '.content', '#content', '.main-content']:
            main_content = response.css(selector).get()
            if main_content:
                break
        
        # Falls kein Hauptinhalt, Body verwenden
        if not main_content:
            main_content = response.css('body').get()
        
        if not main_content:
            return sections
        
        # Parsel Selector für Hauptinhalt erstellen
        content_selector = Selector(text=main_content)
        
        # Ausgeschlossene Elemente mit XPath entfernen
        exclude_xpath = ' | '.join([f'.//{sel}' for sel in self.excludes_selectors if not sel.startswith('.') and not sel.startswith('#')])
        
        # Alle relevanten Elemente finden
        all_tags = self.header_tags + self.content_tags
        xpath_query = ' | '.join([f'.//{tag}' for tag in all_tags])
        elements = content_selector.xpath(xpath_query)
        
        current_header = ""
        current_texts = []
        seen_texts = set()  # Set um bereits gesehene Texte zu tracken
        
        for element in elements:
            # Element-Namen ermitteln
            tag_name = element.xpath('name()').get()
            if not tag_name:
                continue
            tag_name = tag_name.lower()
            
            # Prüfen ob Element ausgeschlossen
            if self.is_excluded_element(element):
                continue
            
            # Wenn Überschrift gefunden
            if tag_name in self.header_tags:
                # Vorherige Sektion speichern
                if current_texts:
                    combined_text = ' '.join(current_texts)
                    cleaned_text = self.clean_text(combined_text)
                    cleaned_text = self.remove_json_ld(cleaned_text)
                    if cleaned_text:
                        sections.append({
                            'header': current_header,
                            'text': cleaned_text
                        })
                
                # Neue Sektion starten
                header_text = ' '.join(element.xpath('.//text()').getall())
                current_header = self.clean_text(header_text)
                current_texts = []
            
            # Content-Elemente sammeln
            elif tag_name in self.content_tags:
                # Nur direkte Text-Inhalte (keine verschachtelten Divs)
                if tag_name == 'div':
                    # Prüfen ob div weitere divs enthält
                    nested_divs = element.xpath('.//div')
                    if nested_divs:
                        continue
                
                text_content = ' '.join(element.xpath('.//text()').getall())
                text_content = self.clean_text(text_content)
                
                # Mindestlänge und Duplikat-Prüfung
                if text_content and len(text_content) > 10:
                    # Normalisierter Text für Duplikat-Erkennung
                    normalized_text = text_content.lower().strip()
                    
                    # Nur hinzufügen wenn nicht bereits gesehen
                    if normalized_text not in seen_texts:
                        # Prüfen ob der Text nicht als Substring in bereits gesehenen Texten enthalten ist
                        is_substring = False
                        for seen in seen_texts:
                            if normalized_text in seen or seen in normalized_text:
                                is_substring = True
                                break
                        
                        if not is_substring:
                            current_texts.append(text_content)
                            seen_texts.add(normalized_text)
        
        # Letzte Sektion speichern
        if current_texts:
            combined_text = ' '.join(current_texts)
            cleaned_text = self.clean_text(combined_text)
            cleaned_text = self.remove_json_ld(cleaned_text)
            if cleaned_text:
                sections.append({
                    'header': current_header,
                    'text': cleaned_text
                })
        
        # Falls keine strukturierten Sektion, Fallback auf gesamten Text
        if not sections:
            all_text = ' '.join(content_selector.xpath('.//text()').getall())
            cleaned_text = self.clean_text(all_text)
            cleaned_text = self.remove_json_ld(cleaned_text)
            if cleaned_text:
                sections.append({
                    'header': '',
                    'text': cleaned_text
                })
        
        return sections
    
    def is_excluded_element(self, element) -> bool:
        # Prüfe Vorfahren-Elemente
        for selector in self.excludes_selectors:
            if selector.startswith('.'):
                # Klassen-Selector
                class_name = selector[1:]
                ancestors_with_class = element.xpath(f'ancestor::*[contains(@class, "{class_name}")]')
                if ancestors_with_class:
                    return True
            elif selector.startswith('#'):
                # ID-Selector
                id_name = selector[1:]
                ancestors_with_id = element.xpath(f'ancestor::*[@id="{id_name}"]')
                if ancestors_with_id:
                    return True
            else:
                # Tag-Selector
                ancestors_with_tag = element.xpath(f'ancestor::{selector}')
                if ancestors_with_tag:
                    return True
        return False
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Mehrfache Leerzeichen entfernen
        cleaned = re.sub(r'\s+', ' ', text)
        # Leerzeichen am Anfang und Ende entfernen
        cleaned = cleaned.strip()
        return cleaned
    
    def remove_json_ld(self, text: str) -> str:
        if not text:
            return ""
        # JSON-LD Patterns entfernen
        pattern = r'\{["\']@context["\']:\s*["\']https?://schema\.org/?["\'][^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        cleaned = re.sub(pattern, '', text)
        # Copytight Pattern entfernen
        cleaned = re.sub(r'©\s*iStock\s*/\s*[\w\s]+', '', cleaned)
        # Leerzeichen entfernen
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

def load_documents() -> list[dict]:
    with open(config.WEB_SCRAPING_OUTPUT_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data