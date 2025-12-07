import sqlite3
from collections import defaultdict


class PregelVertex:

    def __init__(
        self, vertex_id, outgoing_edges=None, num_vertices=1, all_vertices_ids=None
    ):
        self.id = vertex_id
        self.value = 1.0 / num_vertices
        self.outgoing_edges = outgoing_edges or []
        self.incoming_messages = []
        self.active = True
        self.superstep = 0
        self.num_vertices = num_vertices
        self.all_vertices_ids = all_vertices_ids or []

    def update(self):
        self.superstep += 1

        if self.superstep == 0:
            return self._prepare_messages()

        if self.incoming_messages:
            total_contributions = sum(self.incoming_messages)

            new_value = (1 - 0.85) / self.num_vertices + 0.85 * total_contributions
            self.value = new_value

        return self._prepare_messages()

    def _prepare_messages(self):
        if self.outgoing_edges:
            contribution = self.value / len(self.outgoing_edges)
            return [(target, contribution) for target in self.outgoing_edges]
        else:
            contribution = self.value / self.num_vertices
            return [(target_id, contribution) for target_id in self.all_vertices_ids]


class PregelPageRank:

    def __init__(self, db_path):
        self.db_path = db_path
        self.vertices = {}
        self.num_vertices = 0
        self.all_vertices_ids = []
        self.previous_values = {}
        self.load_graph()

    def load_graph(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM documents ORDER BY id")
        documents = [str(row[0]) for row in cursor.fetchall()]
        self.num_vertices = len(documents)
        self.all_vertices_ids = documents[:]

        cursor.execute("SELECT id, url FROM documents")
        url_to_id = {}
        for doc_id, url in cursor.fetchall():
            url_to_id[url] = str(doc_id)

        for doc_id in documents:
            self.vertices[doc_id] = PregelVertex(
                vertex_id=doc_id,
                num_vertices=self.num_vertices,
                all_vertices_ids=self.all_vertices_ids,
            )

        cursor.execute("SELECT source_id, target_url FROM links")
        edge_map = defaultdict(list)
        for source_id, target_url in cursor.fetchall():
            source_id_str = str(source_id)
            if target_url in url_to_id and source_id_str in self.vertices:
                target_id = url_to_id[target_url]
                edge_map[source_id_str].append(target_id)

        for doc_id, vertex in self.vertices.items():
            if doc_id in edge_map:
                vertex.outgoing_edges = list(set(edge_map[doc_id]))

        conn.close()

    def run(self, max_supersteps=20):

        for superstep in range(max_supersteps):
            messages_for_next = defaultdict(list)

            active_count = 0
            for vertex in self.vertices.values():
                if vertex.active:
                    active_count += 1
                    messages = vertex.update()
                    for target_id, message in messages:
                        messages_for_next[target_id].append(message)

            if active_count == 0:
                break

            for target_id, messages in messages_for_next.items():
                if target_id in self.vertices:
                    self.vertices[target_id].incoming_messages.extend(messages)

            total = sum(v.value for v in self.vertices.values())
            if total > 0:
                for vertex in self.vertices.values():
                    vertex.value /= total

            if superstep > 0 and self._check_convergence():
                break

    def _check_convergence(self, tolerance=1e-6):
        max_change = 0.0

        for vertex in self.vertices.values():
            prev = self.previous_values.get(vertex.id, 0)
            change = abs(vertex.value - prev)
            if change > max_change:
                max_change = change

        for vertex in self.vertices.values():
            self.previous_values[vertex.id] = vertex.value

        return max_change < tolerance

    def get_results(self):
        results = {}
        for doc_id, vertex in self.vertices.items():
            results[doc_id] = vertex.value

        total = sum(results.values())
        if abs(total - 1.0) > 0.001:
            print(f"Внимание: сумма PageRank = {total:.6f} (должна быть 1.0)")
            for doc_id in results:
                results[doc_id] /= total

        return results

    def print_results(self):
        results = self.get_results()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, url FROM documents")
        id_to_url = {str(row[0]): row[1] for row in cursor.fetchall()}
        conn.close()

        print("Результаты PageRank (Pregel)")

        sorted_ranks = sorted(results.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'№':>3} | {'Документ ID':>12} | {'PageRank':>10} | URL")
        print("-" * 80)

        for i, (doc_id, rank) in enumerate(sorted_ranks, 1):
            url = id_to_url.get(doc_id, "неизвестно")
            short_url = url[:50] + "..." if len(url) > 50 else url
            print(f"{i:3d} | {doc_id:>12} | {rank:10.6f} | {short_url}")

    def close(self):
        pass


def main():
    db_path = r"C:\\Users\\Бобр Даша\\Desktop\\university\\4 КУРС\\7 семестр\\BD\\лабы\\lab_4\\search.db"

    try:
        pregel = PregelPageRank(db_path)
        pregel.run(max_supersteps=20)
        pregel.print_results()
        pregel.close()

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
