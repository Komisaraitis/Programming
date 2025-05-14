import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)


# Собственный класс для линейной регрессии
class MyLinearModel:
    def __init__(self):
        self.bias = 0  # Свободный член (intercept)
        self.weight = 0  # Коэффициент наклона (slope)

    def train(self, features, target):
        n = len(features)
        sum_x = np.sum(features)
        sum_x_sq = np.sum(features**2)
        sum_y = np.sum(target)
        sum_xy = np.sum(features * target)

        self.weight = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x**2)
        self.bias = (sum_y - self.weight * sum_x) / n

    def make_prediction(self, features):
        return self.bias + self.weight * features


# Загрузка данных о диабете
diabetes_data = datasets.load_diabetes()
df = pd.DataFrame(data=diabetes_data.data, columns=diabetes_data.feature_names)
df["progression"] = diabetes_data.target

# Анализ корреляции признаков
print("Анализ корреляций")
print(df.corr())

# Выбор наиболее значимого признака
selected_feature = "bmi"
X_values = df[selected_feature].values
y_values = df["progression"].values

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.05)

# Подготовка данных для scikit-learn
X_train_reshaped = X_train.reshape(-1, 1)
X_test_reshaped = X_test.reshape(-1, 1)

# Обучение стандартной модели
standard_model = LinearRegression()
standard_model.fit(X_train_reshaped, y_train)

# Обучение моей модели
my_model = MyLinearModel()
my_model.train(X_train, y_train)

# Вывод параметров моделей
print("\nПараметры регрессионных моделей:")
print(
    f"Стандартная модель: intercept = {standard_model.intercept_:.4f}, coef = {standard_model.coef_[0]:.4f}"
)
print(f"Моя модель: bias = {my_model.bias:.4f}, weight = {my_model.weight:.4f}")

# Получение предсказаний
standard_predictions = standard_model.predict(X_test_reshaped)
my_predictions = my_model.make_prediction(X_test)

# Сравнение результатов
print(f'\n{"Моя модель":20}  {"Стандартная модель":20}  {"Реальные значения":20}')
for i in range(len(X_test)):
    print(
        f"{my_predictions[i]:20.4f}  {standard_predictions[i]:20.4f}  {y_test[i]:20.4f}"
    )

# Оценка качества моделей
mae_my = mean_absolute_error(y_test, my_predictions)
mae_standard = mean_absolute_error(y_test, standard_predictions)

mape_my = mean_absolute_percentage_error(y_test, my_predictions)
mape_standard = mean_absolute_percentage_error(y_test, standard_predictions)

r2_my = r2_score(my_model.make_prediction(X_train), y_train)
r2_standard = r2_score(standard_model.predict(X_train_reshaped), y_train)

print("\nМетрики качества:")
print(
    f"Средняя абсолютная ошибка (MAE): Моя = {mae_my:.4f}, Стандартная = {mae_standard:.4f}"
)
print(
    f"Относительная ошибка (MAPE): Моя = {mape_my:.4f}, Стандартная = {mape_standard:.4f}"
)
print(
    f"Коэффициент детерминации (R²): Моя = {r2_my:.4f}, Стандартная = {r2_standard:.4f}"
)

# Визуализация результатов
full_pred_standard = standard_model.predict(X_values.reshape(-1, 1))
full_pred_my = my_model.make_prediction(X_values)

# Первое окно - стандартная модель
plt.figure(1, figsize=(8, 6))
plt.scatter(X_values, y_values, color="#3498db", alpha=0.6, label="Реальные данные")
plt.plot(
    X_values,
    full_pred_standard,
    color="#e74c3c",
    linewidth=2.5,
    label="Стандартная модель",
)
plt.xlabel("Индекс массы тела", fontsize=12)
plt.ylabel("Прогрессия заболевания", fontsize=12)
plt.title("Стандартная линейная регрессия", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.3)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# Второе окно - моя модель
plt.figure(2, figsize=(8, 6))
plt.scatter(X_values, y_values, color="#3498db", alpha=0.6, label="Реальные данные")
plt.plot(X_values, full_pred_my, color="#2ecc71", linewidth=2.5, label="Моя модель")
plt.xlabel("Индекс массы тела", fontsize=12)
plt.ylabel("Прогрессия заболевания", fontsize=12)
plt.title("Моя реализация регрессии", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.3)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
