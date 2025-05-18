import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, learning_curve
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("husl")
pd.set_option("display.float_format", "{:.4f}".format)


def load_data(filepath):
    """Загрузка и подготовка данных"""
    columns = [
        "CRIME",
        "HD",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RH",
        "TAX",
        "PTRATIO",
        "BL",
        "LSTAT%",
        "MEDV",
    ]
    return pd.read_csv(filepath, names=columns, sep="\s+", engine="python")


def plot_learning_curve(estimator, title, X, y, target_mae, cv=5):
    """Визуализация кривых обучения"""
    plt.figure(figsize=(12, 7))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 20),
        scoring="neg_mean_absolute_error",
    )

    train_mae = -np.mean(train_scores, axis=1)
    test_mae = -np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_mae, "o-", label="Ошибка на обучении")
    plt.plot(train_sizes, test_mae, "o-", label="Ошибка на валидации")
    plt.axhline(
        target_mae, color="r", linestyle="--", label=f"Целевое MAE = {target_mae}"
    )

    plt.title(title)
    plt.xlabel("Размер обучающей выборки")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_model(model, X_test, y_test):
    """Оценка модели и вывод метрик"""
    preds = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "MAPE": mean_absolute_percentage_error(y_test, preds),
    }


def print_metrics(name, metrics):
    """Вывод метрик модели"""
    print(
        f'{name}:\nMAE: {metrics["MAE"]:.2f}\nR2: {metrics["R2"]:.2f}\nMAPE: {metrics["MAPE"]:.2f}\n'
    )


df = load_data(r"C:\Users\Бобр Даша\Desktop\university\3 КУРС\6 сем\ML\4\housing.csv")
X = df.drop("MEDV", axis=1)
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

target_mae = 3
plot_learning_curve(
    LinearRegression(), "Линейная регрессия", X_train, y_train, target_mae
)
plot_learning_curve(
    KNeighborsRegressor(), "KNN регрессия", X_train, y_train, target_mae
)

k_values = range(1, 15)
train_errors, val_errors = [], []

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    train_errors.append(mean_absolute_error(y_train, model.predict(X_train)))
    val_errors.append(mean_absolute_error(y_test, model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_errors, "o-", label="Ошибка на обучении")
plt.plot(k_values, val_errors, "o-", label="Ошибка на валидации")
plt.axhline(target_mae, color="r", linestyle="--", label="Целевое MAE = 3")
plt.title("Зависимость MAE от количества соседей")
plt.xlabel("Количество соседей (k)")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.show()

models = {
    "Линейная регрессия": LinearRegression(),
    "Ridge регрессия": Ridge(alpha=1),
    "Lasso регрессия": Lasso(alpha=1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = evaluate_model(model, X_test, y_test)
    print_metrics(name, results[name])

weights = pd.DataFrame(
    {
        "Feature": X.columns,
        "LinearRegression": models["Линейная регрессия"].coef_,
        "Ridge": models["Ridge регрессия"].coef_,
        "Lasso": models["Lasso регрессия"].coef_,
    }
)
print("Веса:\n", weights)

"""
Выводы:
Для данного датасета более подходящей является линейная регрессия.

Линейная регрессия:
Кривые обучения и валидации находятся близко друг к другу. Отсутствие переобучения.
MAE на валидации ≈ 3.0–3.5, что соответствует целевому значению MAE = 3.
Вывод: Модель хорошо обобщает данные, но есть небольшой зазор между обучением и валидацией, что может указывать на недостаточную сложность модели (недообучение).

KNN-регрессия:
Кривые обучения и валидации расходятся при увеличении размера выборки.
MAE на обучении ниже, чем на валидации, что указывает на переобучение при малых k.
Лучшее значение k (по валидации) — около 5, но даже при этом MAE ≈ 3.5–4.0, что хуже, чем у линейной регрессии.

Вывод: KNN хуже справляется с этим датасетом, возможно, из-за нелинейности данных или неоптимального масштабирования признаков.

Целевое значение MAE = 3:
Линейная регрессия почти достигает целевого значения, KNN — нет.



Коллинеарность:
В линейной регрессии NOX имеет аномально большой отрицательный вес (–17.8), что может указывать на мультиколлинеарность.
Ridge и Lasso сглаживают веса, уменьшая влияние коллинеарности.
Lasso обнулил некоторые веса (CRIME, NOX), упростив модель.

Важные признаки:
RM (среднее число комнат) — сильнее всего влияет на цену (положительно).
LSTAT% (% населения с низким статусом) — отрицательно влияет на цену.
DIS (расстояние до рабочих центров) и PTRATIO (соотношение учеников и учителей) также значимы.

Сравнение моделей:
Линейная регрессия: подвержена переобучению.
Ridge: более устойчивая к коллинеарности.
Lasso: упрощает модель, но может снижать точность.
"""
