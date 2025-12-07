import sqlite3
from collections import defaultdict


class PageRankCalculator:
    def __init__(
        self, db_path, damping_factor=0.85, max_iterations=100, tolerance=1e-6
    ):

        self.db_path = db_path
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        self.link_graph = defaultdict(list)
        self.outgoing_links_count = defaultdict(int)
        self.pageranks = {}
        self.num_documents = 0

    def load_graph_from_database(self):

        self.cursor.execute("SELECT id FROM documents ORDER BY id")
        documents = [row[0] for row in self.cursor.fetchall()]
        self.num_documents = len(documents)

        if self.num_documents == 0:
            raise ValueError("В базе данных нет документов!")

        initial_rank = 1.0 / self.num_documents
        for doc_id in documents:
            self.pageranks[doc_id] = initial_rank

        self.cursor.execute("SELECT source_id, target_url FROM links")
        all_links = self.cursor.fetchall()

        self.cursor.execute("SELECT id, url FROM documents")
        url_to_id = {url: doc_id for doc_id, url in self.cursor.fetchall()}

        internal_links_count = 0
        external_links_count = 0

        for source_id, target_url in all_links:
            if target_url in url_to_id:
                target_id = url_to_id[target_url]
                if source_id != target_id:
                    self.link_graph[source_id].append(target_id)
                    internal_links_count += 1
                else:
                    external_links_count += 1

        for source_id in self.link_graph:
            self.outgoing_links_count[source_id] = len(self.link_graph[source_id])

        return True

    def map_step(self, doc_id, rank, link_graph, outgoing_links_count):

        contributions = []

        if outgoing_links_count[doc_id] > 0:
            contribution = rank / outgoing_links_count[doc_id]
            for target_id in link_graph[doc_id]:
                contributions.append((target_id, contribution))
        else:
            contribution = rank / self.num_documents
            for target_id in self.pageranks.keys():
                contributions.append((target_id, contribution))

        return contributions

    def reduce_step(self, contributions_iter):

        new_ranks = {}

        for doc_id, contributions in contributions_iter:
            total_contribution = sum(contributions)

            new_rank = (
                1 - self.damping_factor
            ) / self.num_documents + self.damping_factor * total_contribution

            new_ranks[doc_id] = new_rank

        return new_ranks

    def calculate_page_rank(self):

        self.load_graph_from_database()

        for iteration in range(self.max_iterations):
            all_contributions = []

            for doc_id, rank in self.pageranks.items():
                contributions = self.map_step(
                    doc_id, rank, self.link_graph, self.outgoing_links_count
                )
                all_contributions.extend(contributions)

            contributions_by_doc = defaultdict(list)
            for target_id, contribution in all_contributions:
                contributions_by_doc[target_id].append(contribution)

            new_pageranks = self.reduce_step(contributions_by_doc.items())

            total_change = 0.0
            for doc_id in self.pageranks:
                change = abs(new_pageranks[doc_id] - self.pageranks[doc_id])
                total_change += change

            self.pageranks = new_pageranks

            total = sum(self.pageranks.values())
            if total > 0:
                for doc_id in self.pageranks:
                    self.pageranks[doc_id] /= total

            if total_change < self.tolerance:
                break

        return self.pageranks

    def print_results(self):

        print("РРезультаты PageRank (MapReduce)")

        self.cursor.execute("SELECT id, url FROM documents")
        id_to_url = {doc_id: url for doc_id, url in self.cursor.fetchall()}

        sorted_ranks = sorted(self.pageranks.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'№':>3} | {'Документ ID':>12} | {'PageRank':>10} | URL")
        print("-" * 80)

        for i, (doc_id, rank) in enumerate(sorted_ranks, 1):
            url = id_to_url.get(doc_id, "неизвестно")
            short_url = url[:50] + "..." if len(url) > 50 else url
            print(f"{i:3d} | {doc_id:>12} | {rank:10.6f} | {short_url}")

    def close(self):
        """Закрываем соединение с базой данных"""
        self.conn.close()


def main():
    db_path = r"C:\\Users\\Бобр Даша\\Desktop\\university\\4 КУРС\\7 семестр\\BD\\лабы\\lab_4\\search.db"

    try:

        pr_calculator = PageRankCalculator(
            db_path=db_path,
            damping_factor=0.85,
            max_iterations=100,
            tolerance=1e-6,
        )

        pageranks = pr_calculator.calculate_page_rank()

        pr_calculator.print_results()

        pr_calculator.close()

        return pageranks

    except Exception as e:
        print(f"\n✗ Ошибка при вычислении PageRank: {e}")
        return None


if __name__ == "__main__":
    main()
