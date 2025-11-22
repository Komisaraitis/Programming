import itertools
from typing import List, Tuple, Any, Optional


def shuffler(mapped_words):

    mapped_words = sorted(mapped_words)

    buffer: List[Any] = []
    prev_word: Optional[Any] = None

    for word, value in mapped_words:
        if prev_word == word:
            buffer.append(value)
        else:
            if prev_word is not None:
                yield prev_word, buffer
            buffer = [value]
        prev_word = word

    if prev_word is not None:
        yield prev_word, buffer


class LinearRegressionMR:
    def __init__(self, data_points):
        self.data_points = data_points

    def mapper(self, data_chunk):

        n = len(data_chunk)
        sum_x = sum(x for x, y in data_chunk)
        sum_y = sum(y for x, y in data_chunk)
        sum_xx = sum(x * x for x, y in data_chunk)
        sum_xy = sum(x * y for x, y in data_chunk)

        yield "stats", (n, sum_x, sum_y, sum_xx, sum_xy)

    def reducer(self, key, values_list):

        total_n = 0
        total_sum_x = 0
        total_sum_y = 0
        total_sum_xx = 0
        total_sum_xy = 0

        for values in values_list:
            n, sum_x, sum_y, sum_xx, sum_xy = values
            total_n += n
            total_sum_x += sum_x
            total_sum_y += sum_y
            total_sum_xx += sum_xx
            total_sum_xy += sum_xy

        denominator = total_n * total_sum_xx - total_sum_x * total_sum_x
        if denominator == 0:

            a = float("inf")
            b = total_sum_y / total_n if total_n > 0 else 0
        else:
            a = (total_n * total_sum_xy - total_sum_x * total_sum_y) / denominator
            b = (total_sum_y - a * total_sum_x) / total_n

        return a, b

    def fit(self, chunk_size=2):

        chunks = [
            self.data_points[i : i + chunk_size]
            for i in range(0, len(self.data_points), chunk_size)
        ]

        all_mapped = []
        for chunk in chunks:
            for key, value in self.mapper(chunk):
                all_mapped.append((key, value))

        shuffled = list(shuffler(all_mapped))

        a, b = self.reducer(
            "stats", [value for key, values in shuffled for value in values]
        )

        return a, b

    def predict(self, x, a, b):

        if a == float("inf"):
            return float("inf")
        return a * x + b


def demo_linear_regression():

    data = [
        (2, 10.8),
        (4, 16.9),
        (6, 22.7),
        (8, 28.9),
        (10, 34.8),
        (12, 40.7),
        (14, 46.9),
        (16, 52.6),
        (18, 58.8),
        (20, 64.9),
    ]

    lr_mr = LinearRegressionMR(data)
    a, b = lr_mr.fit()

    print(f"Результаты линейной регрессии:")
    print(f"Уравнение: y = {a:.4f}x + {b:.4f}")

    test_points = [3]
    print(f"\nПрогнозирование:")
    for x in test_points:
        y_pred = lr_mr.predict(x, a, b)
        print(f"x = {x}, y_pred = {y_pred:.4f}")


if __name__ == "__main__":
    demo_linear_regression()
