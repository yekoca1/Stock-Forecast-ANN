# kap_scraper.py
# Module: KAP Disclosure Scraper for Turkish Stocks with RSS & HTML fallback

# Environment:
#   - Python 3.9+
#   - Virtual environment managed via venv or conda
# Dependencies (install with pip):
#   requests
#   beautifulsoup4
#   pandas
#   pdfminer.six
#   lxml
#   feedparser
#   nltk (for stopwords)
#   langdetect (optional)

import io
import requests
import pandas as pd
import feedparser
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from datetime import datetime


class KAPScraper:
    """
    Scrapes KAP disclosures for a given stock ticker using RSS and HTML archive fallback.

    Usage:
        scraper = KAPScraper(ticker="THYAO")
        df = scraper.fetch_announcements(start_date="2024-01-01", end_date="2024-12-31")
    """
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        # RSS feed URL
        self.base_rss = f"https://www.kap.org.tr/{self.ticker}.xml"
        # Archive page URL
        self.base_archive = f"https://www.kap.org.tr/tr/Bildirim/{self.ticker}"

    def fetch_rss_items(self) -> list:
        feed = feedparser.parse(self.base_rss)
        items = []
        for entry in feed.entries:
            if not hasattr(entry, 'published'):
                continue
            items.append({
                'date': entry.published,
                'title': entry.title,
                'url': entry.link
            })
        return items

    def fetch_archive_items(self) -> list:
        """Fallback: scrape HTML archive page for announcements"""
        items = []
        r = requests.get(self.base_archive)
        soup = BeautifulSoup(r.text, 'lxml')
        # Adjust selectors based on KAP page structure
        # Example: rows in a table with class 'table'
        rows = soup.select('table.table tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 3:
                continue
            date_str = cols[0].get_text(strip=True)
            try:
                date = datetime.strptime(date_str, '%d.%m.%Y')
            except ValueError:
                continue
            link_tag = cols[1].find('a', href=True)
            if not link_tag:
                continue
            url = 'https://www.kap.org.tr' + link_tag['href']
            title = link_tag.get_text(strip=True)
            items.append({ 'date': date, 'title': title, 'url': url })
        return items

    def download_and_parse(self, url: str) -> tuple[str, str]:
        r = requests.get(url)
        ctype = r.headers.get('Content-Type', '')
        if 'application/pdf' in ctype or url.lower().endswith('.pdf'):
            text = extract_text(io.BytesIO(r.content))
            return 'pdf', text
        soup = BeautifulSoup(r.text, 'lxml')
        for tag in soup(['script','style']): tag.decompose()
        return 'html', soup.get_text(separator=' ', strip=True)

    def fetch_announcements(self, start_date=None, end_date=None) -> pd.DataFrame:
        # Collect items from RSS
        items = self.fetch_rss_items()
        # If RSS empty, fallback to HTML archive
        if not items:
            print("RSS feed empty, falling back to HTML archive scraping.")
            items = self.fetch_archive_items()
        # Normalize dates
        records = []
        for it in items:
            dt = it['date']
            if isinstance(dt, str):
                try:
                    date = pd.to_datetime(dt)
                except:
                    continue
            else:
                date = pd.to_datetime(dt)
            if start_date and date < pd.to_datetime(start_date): continue
            if end_date and date > pd.to_datetime(end_date): continue
            doc_type, text = self.download_and_parse(it['url'])
            records.append({ 'date': date, 'doc_type': doc_type,
                             'title': it['title'], 'url': it['url'], 'text': text })
        df = pd.DataFrame(records)
        if df.empty:
            print("No announcements found in the given range.")
            return df
        df = df.sort_values('date').reset_index(drop=True)
        return df


if __name__ == '__main__':
    scraper = KAPScraper("THYAO")
    df = scraper.fetch_announcements(start_date="2024-01-01", end_date="2024-12-31")
    print(df)

# Test cases (run with pytest or similar)
def test_rss_url_format():
    s = KAPScraper('THYAO')
    assert 'THYAO.xml' in s.base_rss

def test_archive_url_format():
    s = KAPScraper('THYAO')
    assert '/Bildirim/THYAO' in s.base_archive

# Note: For live HTTP tests, use mocking to simulate RSS and archive HTML.
