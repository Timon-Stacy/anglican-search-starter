import sqlite3
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("library")
import time
from urllib.parse import quote
from io import BytesIO
import subprocess
import sys, json
import re
import shutil
import argparse
import requests

from pdfminer.high_level import extract_text

DB_PATH_DEFAULT = "library.db"

SESSION = None


def check_dependencies():
    """Check for required and optional dependencies at startup."""
    global SESSION

    missing_required = []
    missing_optional = []

    # Check required Python packages
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        missing_required.append("pdfminer.six (pip install pdfminer.six)")

    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        SESSION = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        SESSION.mount("https://", HTTPAdapter(max_retries=retry))
        SESSION.headers.update({"User-Agent": "MVP-Library/0.1 (+no email)"})
    except ImportError:
        missing_required.append("requests (pip install requests)")

    # Optional OCR deps
    if shutil.which("tesseract") is None:
        missing_optional.append("tesseract (OCR will be unavailable)")

    if shutil.which("gswin64c") is None and shutil.which("gs") is None:
        missing_optional.append("ghostscript (OCR will be unavailable)")

    try:
        import ocrmypdf
    except ImportError:
        missing_optional.append("ocrmypdf (pip install ocrmypdf - OCR will be unavailable)")

    if missing_required:
        log.error("Missing required dependencies:")
        for dep in missing_required:
            log.error("  - %s", dep)
        sys.exit(1)

    if missing_optional:
        log.warning("Missing optional dependencies:")
        for dep in missing_optional:
            log.warning("  - %s", dep)
        log.warning("PDF text extraction will work, but OCR fallback will be unavailable.")


class Downloaders:
    domain = None
    name = None
    headers = {"User-Agent": "MVP-Library/0.1 (+no email)"}

    def get_id(self, url: str):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def _download_text(self, urls: list[str]):
        for url in urls:
            try:
                r = SESSION.get(url, timeout=20, headers=self.headers)
                if r.status_code == 200:
                    return r, url
            except Exception as e:
                log.warning("Failed to fetch %s: %s", url, e)
        return None, None

    def _get_json(self, url: str):
        try:
            r = SESSION.get(url, timeout=20, headers=self.headers)
            if r.status_code == 200:
                return r.json(), url
        except Exception as e:
            log.warning("Failed to fetch JSON %s: %s", url, e)
        return None, None

    @classmethod
    def match_downloader(cls, classes: list, url: str):
        for downloader in classes:
            if downloader.domain in url:
                return downloader()
        return None


class GutenbergDownloader(Downloaders):
    domain = "gutenberg.org"
    name = "gutenberg_id"

    def __init__(self):
        self.id = None
        self.book = None

    def get_id(self, url: str):
        if "/ebooks/" not in url:
            return None
        try:
            self.id = int(url.split("/ebooks/")[1].split("?")[0])
        except ValueError:
            return None
        return self.id

    def download(self):
        if self.id is None:
            return None, None

        urls = [
            f"https://www.gutenberg.org/files/{self.id}/{self.id}-0.txt",
            f"https://www.gutenberg.org/files/{self.id}/{self.id}.txt",
            f"https://www.gutenberg.org/ebooks/{self.id}.txt.utf-8",
        ]
        r, url = self._download_text(urls)
        if r and r.text.strip():
            return r.text, url
        return None, None


class InternetArchiveDownloader(Downloaders):
    domain = "archive.org"
    name = "ia_title_id"

    def __init__(self):
        self.id = None
        self.book = None

    def get_id(self, url: str):
        if "/details/" not in url:
            return None
        self.id = url.split("/details/")[1].split("/")[0]
        return self.id

    def download(self):
        if self.id is None:
            return None, None

        urls = [
            f"https://archive.org/download/{self.id}/{self.id}_djvu.txt",
            f"https://archive.org/download/{self.id}/{self.id}.txt",
        ]
        r, url = self._download_text(urls)
        if r and r.text.strip():
            return r.text, url
        return None, None


class GoogleBooksDownloader(Downloaders):
    domain = "books.google."
    name = "gb_title_id"

    def __init__(self):
        self.id = None
        self.book = None

    def get_id(self, url: str):
        patterns = [
            r"/books/edition/[^/]+/([^?\/]+)",
            r"\bid=([^&]+)",
            r"/books\?id=([^&]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                self.id = match.group(1)
                return self.id
        return None

    def download(self):
        if self.id is None:
            return None, None

        url = f"https://www.googleapis.com/books/v1/volumes/{self.id}"
        if api_key:
            url += f"?key={quote(api_key)}"

        meta, _ = self._get_json(url)
        if not meta:
            return None, None
        return extract_pdf(meta)

DOWNLOADERS = [GutenbergDownloader, InternetArchiveDownloader, GoogleBooksDownloader]

# PDF text extraction functions

def extract_text_from_pdf(pdf):
    try:
        text = extract_text(BytesIO(pdf.content))
        if text and text.strip():
            return text
    except Exception:
        log.exception("pdfminer failed")

    return None

def extract_ocr_from_pdf(pdf):
    if shutil.which("tesseract") is None:
        log.info("Tesseract not found — skipping OCR")
        return None

    # Check for ghostscript
    gs_cmd = shutil.which("gswin64c") or shutil.which("gs")
    if gs_cmd is None:
        log.info("Ghostscript not found — skipping OCR")
        return None
    
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ocrmypdf",
             "--skip-text", "--force-ocr", "-l", "eng", "-", "-"],
            input=pdf.content,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # capture stderr for debugging
            check=True,
        )
        text = extract_text(BytesIO(proc.stdout))
        if text and text.strip():
            return text
        else:
            log.warning("OCR produced no text")
    except Exception:
        log.exception("OCR failed")
    return None

def convert_pdf_to_text(pdf):

    text = extract_text_from_pdf(pdf)
    if text is not None:
        return text
    else:
        return extract_ocr_from_pdf(pdf)

def extract_pdf(meta):
    book_meta = (meta.get("accessInfo") or {}).get("pdf") or {}
    if not book_meta.get("isAvailable"):
        log.info("No downloadable PDF for this volume")
        return None, None

    download_url = book_meta.get("downloadLink")
    if not download_url:
        log.warning("PDF marked available but no downloadLink present")
        return None, None
    
    try:
        pdf = SESSION.get(download_url, timeout=60, headers=Downloaders.headers)
        log.info("Google PDF status=%s size=%s bytes", pdf.status_code, len(pdf.content))

        # Tiny PDFs are usually error/captcha pages
        if pdf.status_code != 200 or len(pdf.content) < 50000:
            log.warning("Likely CAPTCHA or error page from Google — skipping")
            return None, None
    except requests.RequestException:
        log.exception("PDF download error")
        return None, None
    
    return convert_pdf_to_text(pdf), download_url


def store_in_db(downloader: Downloaders, connection, cursor):
    title_id, user_title, author, category = downloader.book
    log.info("Downloading %s (%s)", title_id, downloader.domain)
    text, url = downloader.download()
    if text:
        cursor.execute(f"""
            INSERT INTO books ({downloader.name}, author, title, category, source_url, content)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT({downloader.name}) DO UPDATE SET
              title=excluded.title,
              category=excluded.category,
              source_url=excluded.source_url,
              content=excluded.content,
              author=excluded.author
        """, (title_id, author, user_title, category, url, text))
        connection.commit()
        log.info("Stored %s successfully", title_id)
    else:
        log.warning("Failed to store %s", title_id)

def process_input(data, connection, cursor):
    total = len(data)
    for idx, item in enumerate(data, 1):
        lower = {k.lower(): v for k, v in item.items()}

        url = lower.get("url")
        book_title = lower.get("title")
        author   = lower.get("author") or "Unknown"
        category = lower.get("category") or "Uncategorized"

        if not url or not book_title:
            continue

        downloader = Downloaders.match_downloader(DOWNLOADERS, url)
        if downloader is None:
            continue
        source_id = downloader.get_id(url)
        if source_id is None:
            continue;

        cursor.execute(f"SELECT 1 FROM books WHERE {downloader.name} = ?", (source_id,))
        if cursor.fetchone():
            log.info("Skipping %s (already in database)", source_id)
            continue

        log.info("Progress %s/%s", idx, total)
        downloader.book = (source_id, book_title, author, category)
        store_in_db(downloader, connection, cursor)
        time.sleep(1)

def init_database(db_path):
    """Initialize the database schema."""
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS books (
      id            INTEGER PRIMARY KEY,
      gutenberg_id  INTEGER UNIQUE,
      ia_title_id   TEXT UNIQUE,
      gb_title_id   TEXT UNIQUE,
      author        TEXT,
      title         TEXT,
      category      TEXT,
      source_url    TEXT,
      content       TEXT
    );
    
    CREATE UNIQUE INDEX IF NOT EXISTS idx_gutenberg
    ON books(gutenberg_id)
    WHERE gutenberg_id IS NOT NULL;
    
    CREATE UNIQUE INDEX IF NOT EXISTS idx_archive
    ON books(ia_title_id)
    WHERE ia_title_id IS NOT NULL;
    
    CREATE UNIQUE INDEX IF NOT EXISTS idx_google
    ON books(gb_title_id)
    WHERE gb_title_id IS NOT NULL;
    """)
    connection.commit()
    return connection, cursor


def main():
    """Main entry point for the book downloader."""
    global api_key
    
    check_dependencies()
    
    ap = argparse.ArgumentParser(
        description="Download books from various sources into SQLite database."
    )
    ap.add_argument("--db", default=DB_PATH_DEFAULT, 
                    help="Path to SQLite database (default: library.db)")
    ap.add_argument("--api-key", default=None, 
                    help="Optional Google Books API key")
    args = ap.parse_args()
    
    api_key = args.api_key
    
    connection, cursor = init_database(args.db)
    
    try:
        data = json.loads(sys.stdin.read())
        
        process_input(data, connection, cursor)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
    