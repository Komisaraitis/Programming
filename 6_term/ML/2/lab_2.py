import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def load_data():
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
    return pd.read_csv(
        r"C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\6 сем\\ML\\2\\housing.csv",
        names=columns,
        sep="\s+",
        engine="python",
    )


def evaluate_model(X, Y, test_size=0.3, random_state=1):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    return {
        "MAE": mean_absolute_error(Y_test, Y_pred),
        "MAPE": mean_absolute_percentage_error(Y_test, Y_pred),
        "R2": r2_score(Y_test, Y_pred),
    }


def print_results(title, results, size_key, size_label):
    print(f"\n{title}\n")
    print(f"{size_label:^20}  {'MAE':^10}  {'MAPE':^10}  {'R2':^10}")
    for result in results:
        print(
            f"{result[size_key]:^20}  {result['MAE']:^10.4f}  "
            f"{result['MAPE']:^10.4f}  {result['R2']:^10.4f}"
        )


def analyze_sample_sizes(X, Y):
    sample_sizes = [0.2, 0.4, 0.6, 0.8, 1]
    results = []

    for size in sample_sizes:
        if size != 1:
            X_sub, _, Y_sub, _ = train_test_split(X, Y, train_size=size, random_state=1)
        else:
            X_sub, Y_sub = X, Y

        metrics = evaluate_model(X_sub, Y_sub)
        results.append({"SIZE": len(X_sub), **metrics})

    print_results(
        "Зависимость качества модели от размера выборки",
        results,
        "SIZE",
        "Размер выборки",
    )


def analyze_feature_count(data, target):
    feature_sets = [
        ["CRIME", "HD", "INDUS", "CHAS"],
        ["CRIME", "HD", "INDUS", "CHAS", "RM", "AGE", "DIS", "RH"],
        [
            "CRIME",
            "HD",
            "INDUS",
            "CHAS",
            "RM",
            "AGE",
            "DIS",
            "RH",
            "TAX",
            "PTRATIO",
            "BL",
            "LSTAT%",
        ],
        data.drop(["MEDV"], axis=1).columns,
    ]

    results = []
    for i, features in enumerate(feature_sets, 1):
        metrics = evaluate_model(data[features], target)
        results.append({"COUNT": len(features), **metrics})

    print_results(
        "Зависимость качества модели от количества признаков",
        results,
        "COUNT",
        "Количество переменных",
    )


def visualize_3d_relationship(data):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    X_vis = data[["RM", "LSTAT%"]].values
    y_vis = data["MEDV"].values

    model = LinearRegression()
    model.fit(X_vis, y_vis)

    x1 = np.linspace(X_vis[:, 0].min(), X_vis[:, 0].max(), 20)
    x2 = np.linspace(X_vis[:, 1].min(), X_vis[:, 1].max(), 20)
    x1_mesh, x2_mesh = np.meshgrid(x1, x2)

    y_mesh = model.predict(np.c_[x1_mesh.ravel(), x2_mesh.ravel()]).reshape(
        x1_mesh.shape
    )

    # Визуализация
    # 1. Точки данных
    ax.scatter(
        X_vis[:, 0],
        X_vis[:, 1],
        y_vis,
        c="hotpink",
        marker="o",
        s=40,
        label="Реальные данные",
        edgecolor="k",
        alpha=0.7,
    )

    # 2. Плоскость регрессии
    surf = ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        color="lightblue",
        alpha=0.6,
        shade=True,
        edgecolor="dodgerblue",
        linewidth=0.3,
    )

    ax.set_xlabel("RM")
    ax.set_ylabel("LSTAT%")
    ax.set_zlabel("Цена")
    ax.set_title("Зависимость цены от RM и LSTAT%")
    ax.legend()

    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.show()


# Загрузка данных
data = load_data()
X = data.drop(["MEDV"], axis=1)
Y = data[["MEDV"]]

# Анализ влияния размера выборки
analyze_sample_sizes(X, Y)

# Анализ влияния количества признаков
analyze_feature_count(data, Y)

# Вывод корреляций
print("\nКорреляция признаков с MEDV:")
print(data.corr()["MEDV"].sort_values(ascending=False))

# 3D визуализация
visualize_3d_relationship(data)

""" 
Вывод

Увеличение объема данных в наборе способно повысить точность модели. Это подтверждается увеличением значения коэффициента детерминации (R2), 
снижением средней абсолютной ошибки (MAE) и уменьшением средней относительной ошибки в процентах (MAPE). Более широкий набор данных позволяет 
модели лучше учитывать различные вариации, что приводит к более точным предсказаниям.
Аналогично, увеличение числа признаков также может улучшить точность модели, что также отображается в росте R2, 
снижении MAE и уменьшении MAPE. Большее количество переменных предоставляет больше информации для обучения, что способствует лучшему выявлению 
взаимосвязей между ними и, следовательно, повышает точность предсказаний.
"""
