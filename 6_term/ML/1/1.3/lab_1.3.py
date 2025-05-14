from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv(
    "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\6 сем\\ML\\1\\student_scores.csv"
)

# Визуализация данных
plt.scatter(df["Hours"], df["Scores"], marker="o", color="red")
plt.title("Зависимость оценок от времени обучения")
plt.xlabel("Часы обучения")
plt.ylabel("Процент баллов")
plt.grid(True)
plt.show()

# Подготовка данных
X = df[["Hours"]].to_numpy()
y = df["Scores"].to_numpy()

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Вывод параметров модели
print(f"\nСвободный член (intercept): {model.intercept_}")
print(f"Коэффициент: {model.coef_[0]}")

# Предсказания
predictions = model.predict(X_test)

# Сравнение результатов
results = pd.DataFrame({"Фактические": y_test, "Предсказанные": predictions})
print("\n", results)

# Оценка модели
mae = mean_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nMAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")
