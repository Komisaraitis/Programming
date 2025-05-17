from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class IrisAnalyzer:
    def __init__(self):
        self.iris = load_iris()
        self.df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        self.df["target"] = self.iris.target
        self.color_map = {0: "pink", 1: "skyblue", 2: "gold"}

    def visualize_data(self):
        """Визуализация данных Iris"""
        plt.figure(figsize=(12, 6))

        # График зависимости длины и ширины чашелистика
        plt.subplot(1, 2, 1)
        for label in self.df["target"].unique():
            subset = self.df[self.df["target"] == label]
            plt.scatter(
                subset.iloc[:, 0],
                subset.iloc[:, 1],
                color=self.color_map[label],
                label=self.iris.target_names[label],
            )
        plt.title("Зависимость длины и ширины чашелистика")
        plt.xlabel("Длина чашелистика (см)")
        plt.ylabel("Ширина чашелистика (см)")
        plt.legend()

        # График зависимости длины и ширины лепестка
        plt.subplot(1, 2, 2)
        for label in self.df["target"].unique():
            subset = self.df[self.df["target"] == label]
            plt.scatter(
                subset.iloc[:, 2],
                subset.iloc[:, 3],
                color=self.color_map[label],
                label=self.iris.target_names[label],
            )
        plt.title("Зависимость длины и ширины лепестка")
        plt.xlabel("Длина лепестка (см)")
        plt.ylabel("Ширина лепестка (см)")
        plt.legend()

        plt.tight_layout()
        plt.show()

        sns.pairplot(self.df, hue="target", palette=self.color_map)
        plt.suptitle("Парный график для набора данных Iris", y=1.02)
        plt.show()

    def train_models(self):
        """Обучение моделей на разных подмножествах данных"""
        dataset1 = self.df[self.df["target"].isin([0, 1])]
        dataset2 = self.df[self.df["target"].isin([1, 2])]

        # Модель 1: Setosa vs Versicolor
        X1 = dataset1[self.iris.feature_names]
        y1 = dataset1["target"]
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X1, y1, test_size=0.3, random_state=0
        )
        model1 = LogisticRegression(random_state=0)
        model1.fit(X_train1, y_train1)
        y_pred1 = model1.predict(X_test1)
        accuracy1 = model1.score(X_test1, y_test1)

        # Модель 2: Versicolor vs Virginica
        X2 = dataset2[self.iris.feature_names]
        y2 = dataset2["target"]
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X2, y2, test_size=0.3, random_state=0
        )
        model2 = LogisticRegression(random_state=0)
        model2.fit(X_train2, y_train2)
        y_pred2 = model2.predict(X_test2)
        accuracy2 = model2.score(X_test2, y_test2)

        print("\nРезультаты модели на датасете Iris:")
        print("Классификация Setosa vs Versicolor")
        print(f"Предсказания: {np.array2string(y_pred1, separator=', ')}")
        print(f"Точность: {accuracy1 * 100:.2f}%")

        print("\nКлассификация Versicolor vs Virginica")
        print(f"Предсказания: {np.array2string(y_pred2, separator=', ')}")
        print(f"Точность: {accuracy2 * 100:.2f}%")

        return model1, model2


class SyntheticDataAnalyzer:
    def __init__(self):
        self.X, self.y = make_classification(
            n_samples=1000,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=42,
            n_clusters_per_class=1,
        )

    def train_and_visualize(self):
        """Обучение модели и визуализация синтетических данных"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)

        print("\nРезультаты на синтетических данных:")
        print(f"Предсказания модели: {np.array2string(y_pred, separator=', ')}")
        print(f"Точность классификации: {accuracy * 100:.2f}%")

        # Визуализация
        plt.figure(figsize=(8, 6))
        colors = ["pink", "skyblue"]
        for i in range(2):
            plt.scatter(
                self.X[self.y == i, 0],
                self.X[self.y == i, 1],
                color=colors[i],
                label=f"Класс {i}",
                alpha=0.6,
                edgecolors="k",
            )
        plt.title("Синтетический набор данных")
        plt.xlabel("Признак 1")
        plt.ylabel("Признак 2")
        plt.legend()
        plt.show()

        return model


# Запуск
iris_analyzer = IrisAnalyzer()
iris_analyzer.visualize_data()
iris_analyzer.train_models()

synth_analyzer = SyntheticDataAnalyzer()
synth_analyzer.train_and_visualize()
