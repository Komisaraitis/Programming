import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def load_and_preprocess_data(filepath):
    """Загрузка и предварительная обработка данных"""
    df = pd.read_csv(filepath)
    df_clean = df.dropna().drop(["Name", "Cabin", "Ticket", "PassengerId"], axis=1)

    df_clean["Sex"] = df_clean["Sex"].map({"male": 0, "female": 1})
    df_clean["Embarked"] = df_clean["Embarked"].map({"S": 1, "C": 2, "Q": 3})

    lost_percent = (1 - df_clean.size / df.size) * 100
    print(f"Процент потерянных данных: {lost_percent} %\n\n")

    return df_clean


def evaluate_model(model, X_test, y_test, y_pred, y_scores, model_name):
    """Оценка модели и визуализация метрик"""
    accuracy = model.score(X_test, y_test)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{model_name}\nТочность модели: {accuracy*100}")
    print(f"Метрика Recall: {recall:.2f}")
    print(f"Метрика Precision: {precision:.2f}")
    print(f"Метрика F1: {f1:.2f}")

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    plt.title("Матрица ошибок")

    plt.subplot(1, 3, 2)
    prec, rec, _ = precision_recall_curve(y_test, y_scores)
    plt.plot(prec, rec)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision-Recall кривая")

    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")

    plt.suptitle(model_name)
    plt.tight_layout()
    plt.show()


df = load_and_preprocess_data(
    r"C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\6 сем\\ML\\3\\Titanic.csv"
)

X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

log_reg = LogisticRegression(random_state=0, max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
y_scores_log = log_reg.predict_proba(X_test)[:, 1]

evaluate_model(
    log_reg, X_test, y_test, y_pred_log, y_scores_log, "Логистическая регрессия\n"
)

svm = SVC(kernel="linear", probability=True, random_state=0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
y_scores_svm = svm.predict_proba(X_test)[:, 1]

evaluate_model(
    svm, X_test, y_test, y_pred_svm, y_scores_svm, "\nМетод опорных векторов\n"
)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_scores_knn = knn.predict_proba(X_test)[:, 1]

evaluate_model(
    knn, X_test, y_test, y_pred_knn, y_scores_knn, "\nМетод ближайших соседей\n"
)

"""
Вывод: Логистическая регрессия показала лучший баланс метрик — высокую точность, сбалансированные Recall и Precision, 
а также максимальный F1. Метод опорных векторов имеет чуть ниже точность, но лучшую Precision.
Метод ближайших соседей дал худшие результаты с низкой точностью, высоким Recall 
и низкой Precision, склонен к переобучению.
"""
