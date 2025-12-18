import numpy as np
import warnings
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ

mnist = fetch_openml("mnist_784", version=1, parser="auto")
X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype("int64")

X_train, X_test = X[:60000].values, X[60000:].values
y_train, y_test = y[:60000].values, y[60000:].values

# MLP

mlp_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size=128,
                learning_rate_init=0.001,
                max_iter=20,
                random_state=42,
                verbose=True,
                early_stopping=True,
                validation_fraction=0.1,
            ),
        ),
    ]
)

print("Обучение MLP\n")
mlp_pipeline.fit(X_train, y_train)

y_pred_mlp = mlp_pipeline.predict(X_test)

print("\nМетрики качества MLP:\n")

print(f"Accuracy: {accuracy_score(y_test, y_pred_mlp):.4f}")
print(f"F1-micro: {f1_score(y_test, y_pred_mlp, average='micro'):.4f}")
print(f"F1-macro: {f1_score(y_test, y_pred_mlp, average='macro'):.4f}")

# CNN


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

X_train_tensor = torch.tensor(X_train).reshape(-1, 1, 28, 28).float()
X_test_tensor = torch.tensor(X_test).reshape(-1, 1, 28, 28).float()
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

X_train_tensor = (X_train_tensor - MNIST_MEAN) / MNIST_STD
X_test_tensor = (X_test_tensor - MNIST_MEAN) / MNIST_STD

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nОбучение CNN\n")

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

    train_loss /= len(train_loader)
    test_acc = 100.0 * test_correct / test_total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Accuracy: {test_acc:.2f}%"
    )

print("\nМетрики качества CNN:\n")

model.eval()
y_pred_cnn = []
y_true_cnn = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        y_pred_cnn.extend(predicted.cpu().numpy())
        y_true_cnn.extend(target.cpu().numpy())

cnn_acc = accuracy_score(y_true_cnn, y_pred_cnn)
print(f"Accuracy: {cnn_acc:.4f}")
print(f"F1-micro: {f1_score(y_true_cnn, y_pred_cnn, average='micro'):.4f}")
print(f"F1-macro: {f1_score(y_true_cnn, y_pred_cnn, average='macro'):.4f}")

# СРАВНЕНИЕ И ВЫВОДЫ

mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
mlp_f1_micro = f1_score(y_test, y_pred_mlp, average="micro")
mlp_f1_macro = f1_score(y_test, y_pred_mlp, average="macro")

cnn_accuracy = accuracy_score(y_true_cnn, y_pred_cnn)
cnn_f1_micro = f1_score(y_true_cnn, y_pred_cnn, average="micro")
cnn_f1_macro = f1_score(y_true_cnn, y_pred_cnn, average="macro")

print("\nСравнительная таблица:\n")

print(f"{'Метрика':<15} {'MLP':<10} {'CNN':<10} {'Разница':<10}")
print("-" * 50)
print(
    f"{'Accuracy':<15} {mlp_accuracy:.4f}     {cnn_accuracy:.4f}     {cnn_accuracy-mlp_accuracy:+.4f}"
)
print(
    f"{'F1-micro':<15} {mlp_f1_micro:.4f}     {cnn_f1_micro:.4f}     {cnn_f1_micro-mlp_f1_micro:+.4f}"
)
print(
    f"{'F1-macro':<15} {mlp_f1_macro:.4f}     {cnn_f1_macro:.4f}     {cnn_f1_macro-mlp_f1_macro:+.4f}"
)

print(
    """
Выводы:    

Обе модели демонстрируют высокую точность, что подтверждает их эффективность для задачи классификации MNIST.
CNN показывает лучшую обобщающую способность благодаря архитектуре, специально разработанной для работы с изображениями.
Для задач классификации изображений CNN предпочтительнее MLP. 
MLP, хотя и показывает высокую точность, обрабатывает изображения как плоские векторы, теряя информацию о взаимном расположении пикселей.
"""
)
