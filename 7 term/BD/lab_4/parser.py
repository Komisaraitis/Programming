import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import sqlite3
import os
from collections import defaultdict, Counter, OrderedDict

start_urls = [
    "https://ru.wikipedia.org/wiki/Большие_данные",
    "https://ru.wikipedia.org/wiki/NoSQL",
    "https://ru.wikipedia.org/wiki/Hadoop",
    "https://ru.wikipedia.org/wiki/MapReduce",
    "https://ru.wikipedia.org/wiki/SQL",
    "https://ru.wikipedia.org/wiki/База_данных",
]


class DocumentParser:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

        self.visited = set()
        self.documents = []
        self.word_freqs = defaultdict(Counter)
        self.all_links = defaultdict(list)
        self.internal_links = defaultdict(list)

    def extract_text(self, soup):
        for tag in ["script", "style", "nav", "footer", "header", "aside"]:
            for element in soup(tag):
                element.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return " ".join(chunk for chunk in chunks if chunk)

    def extract_words(self, text):
        words = re.findall(r"\b[а-яёa-z][а-яёa-z0-9]*\b", text.lower())
        return [w for w in words if not re.match(r"^\d+$", w)]

    def get_links(self, soup, base_url):
        all_links, internal = [], []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith(("#", "javascript:")):
                continue

            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            all_links.append(normalized)
            if normalized in start_urls and normalized != base_url:
                internal.append(normalized)

        return list(set(all_links)), list(set(internal))

    def parse_document(self, url, doc_id):
        if url in self.visited:
            return None

        try:
            print(f"Парсинг: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            text = self.extract_text(soup)
            words = self.extract_words(text)
            all_links, internal = self.get_links(soup, url)

            doc = {
                "id": str(doc_id),
                "url": url,
                "word_count": len(words),
            }

            self.visited.add(url)
            self.documents.append(doc)
            self.word_freqs[str(doc_id)] = Counter(words)
            self.all_links[str(doc_id)] = all_links
            self.internal_links[str(doc_id)] = internal

            print(f"  ✓ Завершено (слов: {len(words)}, ссылок: {len(all_links)})")
            return True

        except Exception as e:
            print(f"Ошибка при парсинге {url}: {e}")
            return None

    def parse_all(self):
        for i, url in enumerate(start_urls, 1):
            if len(self.documents) >= len(start_urls):
                break
            self.parse_document(url, i)

    def create_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        tables = ["document_word", "words", "links", "documents"]
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")

        cursor.execute(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, url TEXT UNIQUE, word_count INTEGER)"
        )
        cursor.execute(
            "CREATE TABLE words (id INTEGER PRIMARY KEY, word TEXT UNIQUE NOT NULL)"
        )
        cursor.execute(
            """CREATE TABLE document_word (
            document_id TEXT, word_id INTEGER, frequency INTEGER,
            PRIMARY KEY (document_id, word_id),
            FOREIGN KEY (document_id) REFERENCES documents (id),
            FOREIGN KEY (word_id) REFERENCES words (id)
        )"""
        )
        cursor.execute(
            "CREATE TABLE links (source_id TEXT, target_url TEXT, FOREIGN KEY (source_id) REFERENCES documents (id))"
        )

        for doc in self.documents:
            cursor.execute(
                "INSERT INTO documents (id, url, word_count) VALUES (?, ?, ?)",
                (doc["id"], doc["url"], doc["word_count"]),
            )

        ordered_words = OrderedDict()
        for doc_id, counter in self.word_freqs.items():
            for word in counter:
                ordered_words[word] = True

        word_to_id = {}
        for i, word in enumerate(ordered_words.keys(), 1):
            cursor.execute("INSERT INTO words (id, word) VALUES (?, ?)", (i, word))
            word_to_id[word] = i

        doc_word_records = []
        for doc_id, counter in self.word_freqs.items():
            for word, freq in counter.items():
                if word in word_to_id:
                    doc_word_records.append((doc_id, word_to_id[word], freq))

        cursor.executemany(
            "INSERT INTO document_word (document_id, word_id, frequency) VALUES (?, ?, ?)",
            doc_word_records,
        )

        link_records = []
        for source_id, links in self.all_links.items():
            for target_url in links:
                link_records.append((source_id, target_url))

        cursor.executemany(
            "INSERT INTO links (source_id, target_url) VALUES (?, ?)", link_records
        )

        indexes = [
            ("idx_words_word", "words(word)"),
            ("idx_document_word_doc_id", "document_word(document_id)"),
            ("idx_document_word_word_id", "document_word(word_id)"),
            ("idx_document_word_frequency", "document_word(frequency)"),
            ("idx_links_source", "links(source_id)"),
            ("idx_links_target_url", "links(target_url)"),
        ]
        for name, definition in indexes:
            cursor.execute(f"CREATE INDEX {name} ON {definition}")

        conn.commit()

        doc_count = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        word_count = cursor.execute("SELECT COUNT(*) FROM words").fetchone()[0]
        links_count = cursor.execute("SELECT COUNT(*) FROM links").fetchone()[0]

        conn.close()

        print(f"\nДокументов: {doc_count}")
        print(f"Уникальных слов: {word_count}")
        print(f"Ссылок: {links_count}")
        print("\nБаза данных создана.")


def main():
    db_path = r"C:\\Users\\Бобр Даша\\Desktop\\university\\4 КУРС\\7 семестр\\BD\\лабы\\lab_4\\search.db"

    print("Парсинг и создание базы данных")

    parser = DocumentParser(db_path)
    parser.parse_all()
    parser.create_database()


if __name__ == "__main__":
    main()
