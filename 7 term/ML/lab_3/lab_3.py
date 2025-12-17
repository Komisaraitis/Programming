import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Функции активации и потерь
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def bce(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Базовый класс для всех моделей
class BaseModel:

    def predict(self, X, threshold=0.5):
        return (self.forward(X) > threshold).astype(int).flatten()


# Искусственный нейрон с сигмоидальной функцией активации
class SigmoidNeuron(BaseModel):

    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.1
        self.bias = np.random.randn() * 0.1
        self.loss_history = []

    def forward(self, X):
        self.z = np.dot(X, self.weights) + self.bias
        return sigmoid(self.z)

    def backward(self, X, y_true, learning_rate):
        y_pred = self.forward(X)

        error = y_pred.flatten() - y_true
        self.weights -= learning_rate * (X.T @ error) / len(X)
        self.bias -= learning_rate * np.sum(error) / len(X)

        loss = np.mean(bce(y_true, y_pred.flatten()))
        self.loss_history.append(loss)

        return loss


# Многослойная нейронная сеть с двумя скрытыми слоями
class NeuralNetwork(BaseModel):

    def __init__(self, input_size, hidden_size=10):
        self.layers = []
        sizes = [input_size, hidden_size, hidden_size, 1]

        for i in range(3):
            in_size, out_size = sizes[i], sizes[i + 1]
            limit = np.sqrt(6 / (in_size + out_size))
            self.layers.append(
                {
                    "W": np.random.uniform(-limit, limit, (in_size, out_size)),
                    "b": np.zeros(out_size),
                }
            )

        self.loss_history = []

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for layer in self.layers:
            z = np.dot(self.activations[-1], layer["W"]) + layer["b"]
            a = sigmoid(z)
            self.activations.append(a)
            self.z_values.append(z)

        return self.activations[-1]

    def backward(self, X, y_true, learning_rate):

        y_pred = self.forward(X)
        m = len(X)

        y_true = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true

        dZ = y_pred - y_true

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            dW = np.dot(self.activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0) / m

            layer["W"] -= learning_rate * dW
            layer["b"] -= learning_rate * db

            if i > 0:
                dA = np.dot(dZ, layer["W"].T)
                dZ = dA * sigmoid_derivative(self.z_values[i - 1])

        loss = np.mean(bce(y_true.flatten(), y_pred.flatten()))
        self.loss_history.append(loss)

        return loss


# Подготовка данных Iris для бинарной классификации
def prepare_iris_data():
    iris = load_iris()

    mask = iris.target != 0
    X = iris.data[mask][:, 2:4]
    y = (iris.target[mask] == 2).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    return X_train, X_test, y_train, y_test, iris


# Обучение модели
def train_model(model, X_train, y_train, epochs=1000, learning_rate=0.1, verbose=True):
    for epoch in range(epochs):
        loss = model.backward(X_train, y_train, learning_rate)

        if verbose and (epoch % 1000 == 0 or epoch == epochs - 1):
            accuracy = accuracy_score(y_train, model.predict(X_train))
            print(f"Эпоха {epoch:4d} | Loss: {loss:.6f} | Accuracy: {accuracy:.4f}")


# Оценка модели
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=1),
        "Recall": recall_score(y_test, y_pred, zero_division=1),
        "F1": f1_score(y_test, y_pred, zero_division=1),
    }

    print(f"Точность (Accuracy):  {metrics['Accuracy']:.4f}")
    print(f"Точность (Precision): {metrics['Precision']:.4f}")
    print(f"Полнота (Recall):     {metrics['Recall']:.4f}")
    print(f"F1-мера:              {metrics['F1']:.4f}")
    print(f"\nМатрица ошибок:\n{confusion_matrix(y_test, y_pred)}")

    return metrics


# Отрисовка разделяющей линии
def plot_decision_boundary(model, X, y, title, feature_names=None):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z_prob = model.forward(grid).reshape(xx.shape)
    Z = (Z_prob > 0.5).astype(int)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.contour(xx, yy, Z_prob, levels=[0.5], colors=["black"], linewidths=2)

    for class_val, color, label in [
        (0, "blue", "versicolor (0)"),
        (1, "red", "virginica (1)"),
    ]:
        mask = y == class_val
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            c=color,
            edgecolors="k",
            s=50,
            alpha=0.8,
            label=label,
        )

    if feature_names and len(feature_names) >= 2:
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    else:
        plt.xlabel("Petal length")
        plt.ylabel("Petal width")

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main():
    # Подготовка данных
    iris = load_iris()
    X_train, X_test, y_train, y_test, iris = prepare_iris_data()

    # Обучение одного нейрона
    print("\nОбучение одного нейрона")
    neuron = SigmoidNeuron(input_size=2)
    train_model(neuron, X_train, y_train, epochs=5000, learning_rate=0.01, verbose=True)

    # Оценка одного нейрона
    print("\nОценка одного нейрона")
    metrics_neuron = evaluate_model(neuron, X_test, y_test)
    plot_decision_boundary(
        neuron, X_test, y_test, "Один нейрон", iris.feature_names[2:4]
    )

    # Обучение нейронной сети
    print("\nОбучение нейронной сети (2 слоя по 10 нейронов)")
    network = NeuralNetwork(input_size=2, hidden_size=10)
    train_model(
        network, X_train, y_train, epochs=5000, learning_rate=0.05, verbose=True
    )

    # Оценка нейронной сети
    print("\nОценка нейронной сети")
    metrics_network = evaluate_model(network, X_test, y_test)
    plot_decision_boundary(
        network, X_test, y_test, "Нейронная сеть (2x10)", iris.feature_names[2:4]
    )

    # Сравнение моделей
    print("\nСравнение метрик классификации")
    print(f"{'Метрика':<15} {'Нейрон':<10} {'Сеть':<10} {'Изменение':<10}")
    print("-" * 45)

    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        change = metrics_network[metric] - metrics_neuron[metric]
        print(
            f"{metric:<15} {metrics_neuron[metric]:<10.4f} "
            f"{metrics_network[metric]:<10.4f} {change:+.4f}"
        )


if __name__ == "__main__":
    main()


""""
Обучение одного нейрона
Эпоха    0 | Loss: 0.827626 | Accuracy: 0.5000
Эпоха 1000 | Loss: 0.596438 | Accuracy: 0.7429
Эпоха 2000 | Loss: 0.546001 | Accuracy: 0.8429
Эпоха 3000 | Loss: 0.505236 | Accuracy: 0.9143
Эпоха 4000 | Loss: 0.471799 | Accuracy: 0.9286
Эпоха 4999 | Loss: 0.443995 | Accuracy: 0.9286

Оценка одного нейрона
Точность (Accuracy):  0.9000
Точность (Precision): 0.8750
Полнота (Recall):     0.9333
F1-мера:              0.9032

Матрица ошибок:
[[13  2]
 [ 1 14]]

Обучение нейронной сети (2 слоя по 10 нейронов)
Эпоха    0 | Loss: 0.807482 | Accuracy: 0.5000
Эпоха 1000 | Loss: 0.663694 | Accuracy: 0.9429
Эпоха 2000 | Loss: 0.555802 | Accuracy: 0.9429
Эпоха 3000 | Loss: 0.302134 | Accuracy: 0.9571
Эпоха 4000 | Loss: 0.186924 | Accuracy: 0.9571
Эпоха 4999 | Loss: 0.150912 | Accuracy: 0.9429

Оценка нейронной сети
Точность (Accuracy):  0.9333
Точность (Precision): 0.9333
Полнота (Recall):     0.9333
F1-мера:              0.9333

Матрица ошибок:
[[14  1]
 [ 1 14]]

Сравнение метрик классификации
Метрика         Нейрон     Сеть       Изменение
---------------------------------------------
Accuracy        0.9000     0.9333     +0.0333
Precision       0.8750     0.9333     +0.0583
Recall          0.9333     0.9333     +0.0000
F1              0.9032     0.9333     +0.0301


ВЫВОД:

Нейронная сеть с двумя скрытыми слоями по 10 нейронов показала лучшие результаты по сравнению с одиночным нейроном: 
общая точность классификации увеличилась с 90% до 93.3%, а F1-мера — с 90.32% до 93.33%. Сеть
допустила по одной ошибке на каждый класс, тогда как одиночный нейрон совершил две ошибки на классе versicolor и одну на virginica. 
Несмотря на более длительное обучение, нейронная сеть достигла значительно меньшего значения функции потерь (0.15 против 0.44), что свидетельствует о лучшей оптимизации.

"""
