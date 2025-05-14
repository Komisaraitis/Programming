import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_coefficients(data):

    n = len(X)

    b1 = sum((X[i] - sum(X) / n) * (Y[i] - sum(Y) / n) for i in range(n)) / sum(
        (X[i] - sum(X) / n) ** 2 for i in range(n)
    )
    b0 = sum(Y) / n - b1 * sum(X) / n

    return b0, b1


file_path = input("Введите путь к файлу CSV: ")
data = pd.read_csv(file_path)

columns = data.columns.tolist()
print("Столбцы:\n" + "\n".join(f"{i + 1}) {col}" for i, col in enumerate(columns)))

print("Введите номер столбца для X и номер столбца для Y")

x_index, y_index = int(input()) - 1, int(input()) - 1

x_column, y_column = columns[x_index], columns[y_index]

print("Статистическая информация:")
for column in data.columns:
    print(
        f"{column}:\nКоличество: {data[column].count()}, минимум: {data[column].min()}, максимум: {data[column].max()}, среднее: {data[column].mean()}"
    )

X, Y = data[x_column].values, data[y_column].values

plt.figure(figsize=(15, 10))
plt.scatter(data[x_column], data[y_column], color="blue")
plt.xlabel(x_column)
plt.ylabel(y_column)

b0, b1 = calculate_coefficients(data)

Y_predicted = b0 + b1 * X
plt.plot(X, Y_predicted, color="green")

for i in range(len(X)):
    size = abs(Y_predicted[i] - Y[i])
    plt.fill_betweenx([Y[i], Y_predicted[i]], X[i], X[i] + size, color="red", alpha=0.3)

plt.xlabel(x_column)
plt.ylabel(y_column)
plt.show()
