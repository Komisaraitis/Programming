import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

# ГЕНЕРАЦИЯ И ПОДГОТОВКА ДАННЫХ (ИЗ ЛАБОРАТОРНОЙ №1)


def generate_time_series(n_points=1000, has_trend=True, has_seasonality=True, seed=42):
    """Генерация временного ряда (из первой лабораторной)"""
    np.random.seed(seed)
    start_date = datetime(2020, 1, 1)
    dates = [
        (start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_points)
    ]

    noise = np.random.normal(0, 5, n_points)
    trend_component = np.zeros(n_points)
    seasonal_component = np.zeros(n_points)

    if has_trend:
        slope = 0.05
        intercept = 75
        trend_component = slope * np.arange(n_points) + intercept

    if has_seasonality:
        monthly_amplitude = 15
        seasonal_component += monthly_amplitude * np.sin(
            2 * np.pi * np.arange(n_points) / 30 + 0.7 * np.pi
        )

    values = noise + trend_component + seasonal_component
    return dates, values


dates, values = generate_time_series(seed=42)
ts_dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
ts_values = np.array(values)
n_points = len(ts_values)

print(f"Сгенерировано временных точек: {n_points}")
print(f"Диапазон дат: {dates[0]} - {dates[-1]}")
print(f"Среднее значение: {np.mean(ts_values):.2f}")
print(f"Стандартное отклонение: {np.std(ts_values):.2f}")
print(f"Минимальное значение: {np.min(ts_values):.2f}")
print(f"Максимальное значение: {np.max(ts_values):.2f}")

plt.figure(figsize=(14, 6))
plt.plot(ts_dates, ts_values, linewidth=1.2)
plt.title("Сгенерированный временной ряд", fontsize=14, fontweight="bold")
plt.xlabel("Дата", fontsize=12)
plt.ylabel("Значение", fontsize=12)
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.show()

# РАЗДЕЛЕНИЕ ДАННЫХ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ

test_size = int(0.2 * n_points)
train_size = n_points - test_size

train_data = ts_values[:train_size]
test_data = ts_values[train_size:]
train_dates = ts_dates[:train_size]
test_dates = ts_dates[train_size:]

# ПРЕДОБРАБОТКА ДАННЫХ ДЛЯ LSTM

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
test_scaled = scaler.transform(test_data.reshape(-1, 1))


def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


SEQ_LENGTH = 30
X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"\nПОДГОТОВКА ДАННЫХ ДЛЯ LSTM")
print(f"   Длина последовательности: {SEQ_LENGTH} дней")
print(f"   Размер X_train: {X_train.shape}")
print(f"   Размер y_train: {y_train.shape}")
print(f"   Размер X_test: {X_test.shape}")
print(f"   Размер y_test: {y_test.shape}")

# СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ LSTM

print(f"\nПОСТРОЕНИЕ МОДЕЛИ LSTM")

model = Sequential(
    [
        LSTM(
            units=50,
            activation="tanh",
            return_sequences=True,
            input_shape=(SEQ_LENGTH, 1),
        ),
        Dropout(0.2),
        LSTM(units=50, activation="tanh", return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, activation="tanh"),
        Dropout(0.2),
        Dense(units=1),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
)

print("   Архитектура модели:")
model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)


print(f"\nОБУЧЕНИЕ МОДЕЛИ")
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1,
)

# ПРОГНОЗИРОВАНИЕ НА ТЕСТОВОЙ ВЫБОРКЕ И ОЦЕНКА КАЧЕСТВА

print(f"\nПРОГНОЗИРОВАНИЕ НА ТЕСТОВОЙ ВЫБОРКЕ")
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler.inverse_transform(y_pred_scaled)

y_test_actual = test_data[SEQ_LENGTH:]
y_test_original = y_test_actual.reshape(-1, 1)


mae = mean_absolute_error(y_test_original, y_pred)
mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test_original, y_pred) * 100
r2 = r2_score(y_test_original, y_pred)

print("\nМЕТРИКИ КАЧЕСТВА ПРОГНОЗА (тестовая выборка):")
print(f"  MAE  (Средняя абсолютная ошибка): {mae:.4f}")
print(f"  MSE  (Средняя квадратичная ошибка): {mse:.4f}")
print(f"  RMSE (Среднеквадратичное отклонение): {rmse:.4f}")
print(f"  MAPE (Средняя абсолютная процентная ошибка): {mape:.2f}%")
print(f"  R²   (Коэффициент детерминации): {r2:.4f}")

test_dates_adj = test_dates[SEQ_LENGTH:]

plt.figure(figsize=(14, 6))

plt.plot(ts_dates, ts_values, label="Исходный временной ряд", alpha=0.7, linewidth=1)

plt.plot(test_dates_adj, y_pred, label="Прогноз LSTM", color="red", linewidth=2)

plt.fill_between(
    test_dates_adj,
    y_pred.flatten() - rmse,
    y_pred.flatten() + rmse,
    color="red",
    alpha=0.2,
    label=f"Возможная ошибка",
)

plt.title(
    "Прогноз временного ряда с помощью LSTM (тестовая выборка)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Дата", fontsize=12)
plt.ylabel("Значение", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ПРОГНОЗ НА БУДУЩИЙ ПЕРИОД (60 дней)

future_steps = 60

last_sequence_scaled = scaler.transform(test_data[-SEQ_LENGTH:].reshape(-1, 1))

future_preds_scaled = []
current_seq = last_sequence_scaled.flatten()

for step in range(future_steps):
    next_pred = model.predict(current_seq.reshape(1, SEQ_LENGTH, 1), verbose=0)
    future_preds_scaled.append(next_pred[0, 0])

    current_seq = np.roll(current_seq, -1)
    current_seq[-1] = next_pred[0, 0]

future_preds_scaled = np.array(future_preds_scaled)

future_preds = scaler.inverse_transform(future_preds_scaled.reshape(-1, 1)).flatten()

last_date = test_dates[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]

plt.figure(figsize=(14, 6))

plt.plot(
    ts_dates,
    ts_values,
    label="Исторические данные",
    alpha=0.8,
    linewidth=1.5,
)

plt.plot(
    future_dates,
    future_preds,
    label="Прогноз LSTM на 60 дней",
    color="red",
    linewidth=2,
)

plt.fill_between(
    future_dates,
    future_preds - rmse,
    future_preds + rmse,
    color="red",
    alpha=0.2,
    label=f"Возможная ошибка",
)

plt.axvline(
    x=last_date, color="black", linestyle=":", linewidth=1.5, label="Начало прогноза"
)

plt.title(
    "Прогноз временного ряда на 60 дней с помощью LSTM", fontsize=14, fontweight="bold"
)
plt.xlabel("Дата", fontsize=12)
plt.ylabel("Значение", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# СРАВНИТЕЛЬНЫЙ АНАЛИЗ И ЗАКЛЮЧЕНИЕ

print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ LSTM И SARIMA")

print("\nМЕТРИКИ КАЧЕСТВА ПРОГНОЗА:")
print(f"{'Метрика':<25} {'SARIMA':<15} {'LSTM':<15} {'Разница':<15}")
print("-" * 70)
print(f"{'MAE':<25} {3.9629:<15.4f} {mae:<15.4f} {mae-3.9629:<+15.4f}")
print(f"{'MSE':<25} {24.0565:<15.4f} {mse:<15.4f} {mse-24.0565:<+15.4f}")
print(f"{'RMSE':<25} {4.9047:<15.4f} {rmse:<15.4f} {rmse-4.9047:<+15.4f}")
print(f"{'MAPE (%)':<25} {3.30:<15.2f} {mape:<15.2f} {mape-3.30:<+15.2f}")
print(f"{'R²':<25} {0.8225:<15.4f} {r2:<15.4f} {r2-0.8225:<+15.4f}")


print(f"\nSARIMA показала немного лучшую точность по всем метрикам")

print("\nВЫБОР МОДЕЛИ ДЛЯ КОНКРЕТНОЙ ЗАДАЧИ:")
print("Для небольших рядов с четкой структурой - SARIMA")
print("Для больших рядов со сложными зависимостями - LSTM")
print("Для производственных систем с требованием интерпретации - SARIMA")
print("Для исследовательских задач с большими данными - LSTM")

print("\nВЫВОД:")
print("Хотя LSTM является мощным инструментом для сложных временных рядов,")
print("для данной конкретной задачи с ее относительно простой структурой")
print("классический метод SARIMA оказывается более эффективным решением.")
print("LSTM стоит выбирать для более сложных рядов с нелинейностями,")
print("большими объемами данных или когда нужна максимальная гибкость модели.")
