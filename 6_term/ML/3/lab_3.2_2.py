import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

np.random.seed(123)
df["petal length (cm)"] = df["petal length (cm)"] * (
    0.9 + np.random.rand(len(df)) * 0.2
)
df["petal width (cm)"] = df["petal width (cm)"] * (0.9 + np.random.rand(len(df)) * 0.2)

features = ["petal length (cm)", "petal width (cm)"]
X = df[features].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=123
)
model = LogisticRegression(
    multi_class="multinomial", solver="saga", C=0.5, max_iter=200
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print(f"Предсказания модели: {y_pred}\n")
print(f"Точность модели: {accuracy:.2f}")

custom_cmap = ListedColormap(["#FF6B8B", "#00B4D8", "#FFD166"])

plt.figure(figsize=(10, 6), facecolor="#f5f5f5")
scatter = plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pred,
    cmap=custom_cmap,
    edgecolor="k",
    s=70,
    alpha=0.8,
)

plt.xlabel("Длина лепестка (см)", fontsize=12)
plt.ylabel("Ширина лепестка (см)", fontsize=12)
plt.title(
    "Результаты классификации цветов Iris (модифицированные данные)",
    fontsize=14,
    pad=20,
)

legend_labels = ["setosa", "versicolor", "virginica"]
legend = plt.legend(
    handles=scatter.legend_elements()[0],
    labels=legend_labels,
    title="Классы",
    frameon=True,
    shadow=True,
    facecolor="white",
)
legend.get_title().set_fontsize(12)

plt.grid(True, linestyle="--", alpha=0.3)

plt.text(
    0.02,
    0.98,
    f"Точность: {accuracy:.0%}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8),
)

plt.tight_layout()
plt.show()
