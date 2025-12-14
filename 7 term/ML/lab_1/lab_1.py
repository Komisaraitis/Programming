import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
import warnings

warnings.filterwarnings("ignore")

# ГЕНЕРАЦИЯ ВРЕМЕННОГО РЯДА

print("\nГЕНЕРАЦИЯ ВРЕМЕННОГО РЯДА\n")


def generate_time_series(n_points=1000, has_trend=True, has_seasonality=True, seed=42):
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

# КЛАССИЧЕСКИЕ СТАТИСТИЧЕСКИЕ ТЕСТЫ

print("\nКЛАССИЧЕСКИЕ СТАТИСТИЧЕСКИЕ ТЕСТЫ")


def adf_test(series):
    """Тест Дики-Фуллера на стационарность"""
    res = adfuller(series, autolag="AIC")
    return {"ADF statistic": res[0], "p-value": res[1]}


def kpss_test(series):
    """KPSS тест на стационарность"""
    res = kpss(series, nlags="auto", regression="ct")
    return {"KPSS statistic": res[0], "p-value": res[1]}


print("\n1. Тест Дики-Фуллера (ADF)")
adf_result = adf_test(ts_values)
print(f"ADF статистика: {adf_result['ADF statistic']:.6f}")
print(f"p-value: {adf_result['p-value']:.6f}")

print("\n2. Тест KPSS")
kpss_result = kpss_test(ts_values)
print(f"KPSS статистика: {kpss_result['KPSS statistic']:.6f}")
print(f"p-value: {kpss_result['p-value']:.6f}")

ts_diff_1 = np.diff(ts_values)

print("\n1. Тест Дики-Фуллера (ADF) - ряд после 1-го дифференцирования")
adf_result_diff = adf_test(ts_diff_1)
print(f"ADF статистика: {adf_result_diff['ADF statistic']:.6f}")
print(f"p-value: {adf_result_diff['p-value']:.6f}")

print("\n2. Тест KPSS - ряд после 1-го дифференцирования")
kpss_result_diff = kpss_test(ts_diff_1)
print(f"KPSS статистика: {kpss_result_diff['KPSS statistic']:.6f}")
print(f"p-value: {kpss_result_diff['p-value']:.6f}")


# РАЗДЕЛЕНИЕ РЯДА НА КОМПОНЕНТЫ

ts_series = pd.Series(ts_values, index=pd.DatetimeIndex(ts_dates))
decomposition = seasonal_decompose(ts_series, model="additive", period=30)

fig = decomposition.plot()
fig.set_size_inches(14, 9)
plt.suptitle(
    "Декомпозиция временного ряда на компоненты", fontsize=16, fontweight="bold"
)
plt.tight_layout()
plt.show()

# АНАЛИЗ АВТОКОРРЕЛЯЦИОННЫХ ФУНКЦИЙ

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Автокорреляционный анализ", fontsize=14, fontweight="bold")

plot_acf(ts_values, lags=60, ax=axes[0], title="ACF")
axes[0].grid(True, alpha=0.3)

plot_pacf(ts_values, lags=60, ax=axes[1], title="PACF")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ВЫБОР МЕТОДА МОДЕЛИРОВАНИЯ

print("\nВЫБОР МЕТОДА МОДЕЛИРОВАНИЯ И ОБОСНОВАНИЕ\n")

print("Выбранная модель - SARIMA, потому что:")
print("SARIMA создана для работы с нестационарными рядами")
print("Наличие явной сезонности → нужна сезонная модель")
print("PACF показывает значимые пики → AR компонента")
print("ACF показывает сезонные пики → MA компонента с сезонностью")

# МОДЕЛИРОВАНИЕ И ПРОГНОЗИРОВАНИЕ

print("\nМОДЕЛИРОВАНИЕ И ПРОГНОЗИРОВАНИЕ\n")

h_test = int(round(0.2 * n_points))
train_data = ts_values[:-h_test]
test_data = ts_values[-h_test:]
train_dates = ts_dates[:-h_test]
test_dates = ts_dates[-h_test:]

s = 30
d = 1
D = 1

p_vals = [0, 1, 2]
q_vals = [0, 1]
P_vals = [0, 1]
Q_vals = [0, 1]

print("Поиск оптимальной модели по AIC:\n")

best = {"aic": np.inf, "order": None, "seasonal_order": None}

for p in p_vals:
    for q in q_vals:
        for P in P_vals:
            for Q in Q_vals:
                order = (p, d, q)
                seasonal_order = (P, D, Q, s)

                try:
                    model = SARIMAX(
                        train_data,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend="n",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    results = model.fit(disp=False, maxiter=50, method="lbfgs")

                    print(f"SARIMA{order}x{seasonal_order}: AIC = {results.aic:.2f}")

                    if results.aic < best["aic"]:
                        best.update(
                            {
                                "aic": results.aic,
                                "order": order,
                                "seasonal_order": seasonal_order,
                            }
                        )

                except Exception:
                    print(f"SARIMA{order}x{seasonal_order}: ошибка при обучении")
                    continue


print("\nРезультаты подбора по AIC:")

print(f"Лучшая модель: SARIMA{best['order']}x{best['seasonal_order']}")
print(f"Лучший AIC: {best['aic']:.2f}")

best_order = best["order"]
best_seasonal = best["seasonal_order"]

model = SARIMAX(
    train_data,
    order=best_order,
    seasonal_order=best_seasonal,
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False,
)
model_fit = model.fit(disp=False, maxiter=100, method="lbfgs")

pred = model_fit.get_forecast(steps=len(test_data))
forecast = np.asarray(pred.predicted_mean)
forecast_conf_int = np.asarray(pred.conf_int(alpha=0.05))

# АНАЛИЗ МЕТРИК КАЧЕСТВА
print("\nАнализ метрик качества прогноза\n")

mae = mean_absolute_error(test_data, forecast)
mse = mean_squared_error(test_data, forecast)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(test_data, forecast)
r2 = r2_score(test_data, forecast)

print(f"  MAE  (Средняя абсолютная ошибка): {mae:.4f}")
print(f"  MSE  (Средняя квадратичная ошибка): {mse:.4f}")
print(f"  RMSE (Среднеквадратичное отклонение): {rmse:.4f}")
print(f"  MAPE (Средняя абсолютная процентная ошибка): {mape*100:.2f}%")
print(f"  R²   (Коэффициент детерминации): {r2:.4f}")

# ГРАФИК МОДЕЛИ С ПРОГНОЗОМ

plt.figure(figsize=(14, 5))
plt.plot(train_dates, train_data, linewidth=1.0, label="Обучающая выборка")
plt.plot(test_dates, test_data, linewidth=1.2, label="Тестовая выборка")
plt.plot(test_dates, forecast, linewidth=1.5, label="Прогноз")
plt.fill_between(
    test_dates,
    forecast_conf_int[:, 0],
    forecast_conf_int[:, 1],
    alpha=0.25,
    label="95% доверительный интервал",
)

plt.axvline(
    x=train_dates[-1],
    color="black",
    linestyle=":",
    linewidth=1.5,
    label="Начало прогноза",
)

plt.title(
    f"SARIMA{best_order}x{best_seasonal}: прогноз на тестовой выборке",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Дата", fontsize=12)
plt.ylabel("Значение", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 9. ПРОГНОЗ НА БУДУЩИЙ ПЕРИОД

full_model = SARIMAX(
    ts_values,
    order=best["order"],
    seasonal_order=best["seasonal_order"],
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False,
)
full_res = full_model.fit(disp=False, maxiter=100, method="lbfgs")

future_steps = 60
future = full_res.get_forecast(steps=future_steps)
future_mean = np.asarray(future.predicted_mean)
future_ci = np.asarray(future.conf_int(alpha=0.05))

last_date = ts_dates[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]

plt.figure(figsize=(14, 5))
plt.plot(ts_dates, ts_values, linewidth=1.0, label="Исторические данные")
plt.plot(future_dates, future_mean, linewidth=1.5, label="Прогноз на 60 дней")
plt.fill_between(
    future_dates,
    future_ci[:, 0],
    future_ci[:, 1],
    alpha=0.25,
    label="95% доверительный интервал",
)

plt.axvline(
    x=last_date,
    color="black",
    linestyle=":",
    linewidth=1.5,
    label="Начало прогноза",
)

plt.title(
    f"SARIMA{best['order']}x{best['seasonal_order']}: прогноз на 60 дней",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Дата", fontsize=12)
plt.ylabel("Значение", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
