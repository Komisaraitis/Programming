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


class MatrixMultiplicationMR:
    def __init__(self, matrix_A, matrix_B):
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.n = len(matrix_A)
        self.m = len(matrix_A[0]) if matrix_A else 0
        self.p = len(matrix_B[0]) if matrix_B else 0

    def mapper_A(self):
        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.p):
                    yield (i, k), ("A", j, self.matrix_A[i][j])

    def mapper_B(self):
        for j in range(self.m):
            for k in range(self.p):
                for i in range(self.n):
                    yield (i, k), ("B", j, self.matrix_B[j][k])

    def reducer(self, key, values):
        i, k = key

        A_values = {}
        B_values = {}

        for value in values:
            matrix_type, j, val = value
            if matrix_type == "A":
                A_values[j] = val
            else:
                B_values[j] = val

        result = 0
        for j in range(self.m):
            if j in A_values and j in B_values:
                result += A_values[j] * B_values[j]
        return key, result

    def multiply(self):

        all_mapped = []

        for key, value in self.mapper_A():
            all_mapped.append((key, value))

        for key, value in self.mapper_B():
            all_mapped.append((key, value))

        shuffled = shuffler(all_mapped)

        results = []
        for key, values in shuffled:
            result_key, result_value = self.reducer(key, values)
            results.append((result_key, result_value))

        result_matrix = [[0] * self.p for _ in range(self.n)]
        for (i, k), value in results:
            result_matrix[i][k] = value

        return result_matrix


def demo_matrix_multiplication():

    A = [[1, 4, 5], [2, 3, 6]]

    B = [[7, 8], [10, 9], [11, 12]]

    print("Матрица A:")
    for row in A:
        print(row)

    print("\nМатрица B :")
    for row in B:
        print(row)

    matrix_mr = MatrixMultiplicationMR(A, B)
    result = matrix_mr.multiply()

    print("\nРезультат умножения A × B:")
    for row in result:
        print(row)


if __name__ == "__main__":
    demo_matrix_multiplication()
