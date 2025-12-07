import sqlite3
from collections import defaultdict
import math
import time


class FullTextSearch:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        self.total_documents = self.cursor.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        self.doc_lengths = {}
        self.word_to_id = {}
        self.word_idf = {}

        self._load_data()

    def _load_data(self):
        self.cursor.execute("SELECT id, word_count FROM documents")
        for row in self.cursor.fetchall():
            self.doc_lengths[str(row["id"])] = row["word_count"]

        self.cursor.execute("SELECT id, word FROM words")
        for row in self.cursor.fetchall():
            word, word_id = row["word"], row["id"]
            self.word_to_id[word] = word_id

            self.cursor.execute(
                "SELECT COUNT(DISTINCT document_id) FROM document_word WHERE word_id = ?",
                (word_id,),
            )
            doc_freq = self.cursor.fetchone()[0]
            self.word_idf[word] = (
                math.log((self.total_documents + 1) / (doc_freq + 1)) + 1
            )

    def get_word_id(self, word):
        return self.word_to_id.get(word.lower().strip())

    def document_at_a_time_search(self, query_terms):
        start_time = time.time()

        word_ids, valid_terms = [], []
        for term in query_terms:
            if word_id := self.get_word_id(term):
                word_ids.append(word_id)
                valid_terms.append(term)

        if not word_ids:
            return [], time.time() - start_time, valid_terms

        term_postings, word_id_to_text = {}, {}
        for word_id in word_ids:
            self.cursor.execute(
                "SELECT document_id, frequency FROM document_word WHERE word_id = ? ORDER BY document_id",
                (word_id,),
            )
            if postings := self.cursor.fetchall():
                term_postings[word_id] = postings
                self.cursor.execute("SELECT word FROM words WHERE id = ?", (word_id,))
                word_id_to_text[word_id] = self.cursor.fetchone()[0]

        if not term_postings:
            return [], time.time() - start_time, valid_terms

        pointers = {word_id: 0 for word_id in term_postings.keys()}
        scores = defaultdict(float)

        while True:
            current_docs = [
                term_postings[word_id][pointers[word_id]][0]
                for word_id in pointers
                if pointers[word_id] < len(term_postings[word_id])
            ]
            if not current_docs:
                break

            current_doc = min(current_docs)
            doc_score = 0.0

            for word_id in list(pointers.keys()):
                pos = pointers[word_id]
                if pos < len(term_postings[word_id]):
                    doc_id, freq = term_postings[word_id][pos]
                    if doc_id == current_doc:
                        word_text = word_id_to_text[word_id]
                        tf = freq / self.doc_lengths.get(str(doc_id), 1)
                        doc_score += tf * self.word_idf.get(word_text, 1)
                        pointers[word_id] += 1

            if doc_score > 0:
                scores[str(current_doc)] = doc_score

        results = []
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            self.cursor.execute("SELECT url FROM documents WHERE id = ?", (doc_id,))
            if url := self.cursor.fetchone()[0]:
                results.append((doc_id, url, score))

        return results, time.time() - start_time, valid_terms

    def term_at_a_time_search(self, query_terms):
        start_time = time.time()

        valid_terms, word_ids = [], []
        for term in query_terms:
            normalized_term = term.lower().strip()
            if word_id := self.get_word_id(term):
                valid_terms.append(normalized_term)
                word_ids.append(word_id)

        if not valid_terms:
            return [], time.time() - start_time, valid_terms

        doc_scores = defaultdict(float)

        for i, term in enumerate(valid_terms):
            word_id = word_ids[i]
            idf = self.word_idf.get(term, 1)

            self.cursor.execute(
                "SELECT document_id, frequency FROM document_word WHERE word_id = ?",
                (word_id,),
            )
            for doc_id, freq in self.cursor.fetchall():
                doc_id_str = str(doc_id)
                tf = freq / self.doc_lengths.get(doc_id_str, 1)
                doc_scores[doc_id_str] += tf * idf

        results = []

        for doc_id, score in sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        ):
            self.cursor.execute("SELECT url FROM documents WHERE id = ?", (doc_id,))
            if url := self.cursor.fetchone()[0]:
                results.append((doc_id, url, score))

        return results, time.time() - start_time, valid_terms

    def search(self):
        print("Полнотекстовый поиск по документам")

        while True:
            query = input(
                "\nВведите поисковый запрос (чтоб вернуться в меню, введите 0):"
            ).strip()

            if query == "0":
                break

            if not query:
                continue

            query_terms = query.split()

            print(
                "\nВыберите алгоритм поиска:\n1. Document-at-a-time\n2. Term-at-a-time\n0. Вернуться в меню"
            )
            choice = input().strip()

            if choice == "0":
                break
            elif choice == "1":
                results, time_taken, _ = self.document_at_a_time_search(query_terms)
                self._display_results("Document-at-a-time", results, time_taken)
            elif choice == "2":
                results, time_taken, _ = self.term_at_a_time_search(query_terms)
                self._display_results("Term-at-a-time", results, time_taken)
            else:
                print("Неверный выбор")

    def _display_results(self, algorithm_name, results, time_taken):
        print(f"\n{algorithm_name}")
        print(f"Найдено документов: {len(results)}")

        if not results:
            print("По вашему запросу ничего не найдено.")
            return

        print(f"\n{'Рейтинг':<7} | {'Документ':<8} | {'Релевантность':<12} | URL")
        print("-" * 80)

        for i, (doc_id, url, score) in enumerate(results, 1):
            display_url = url[:57] + "..." if len(url) > 60 else url
            print(f"{i:<7} | {doc_id:<8} | {score:<12.6f} | {display_url}")

    def close(self):
        self.conn.close()


def main():
    db_path = r"C:\\Users\\Бобр Даша\\Desktop\\university\\4 КУРС\\7 семестр\\BD\\лабы\\lab_4\\search.db"

    search_engine = FullTextSearch(db_path)
    search_engine.search()
    search_engine.close()


if __name__ == "__main__":
    main()
