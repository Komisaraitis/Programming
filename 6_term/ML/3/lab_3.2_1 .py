import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def load_and_clean_data(filepath):
    """Загрузка данных и первичная очистка"""
    df = pd.read_csv(filepath)

    df_clean = df.dropna()

    lost_percent = (1 - df_clean.size / df.size) * 100
    print(f"Процент потерянных данных: {lost_percent} %\n")

    df_clean = df_clean.drop(columns=["Name", "Cabin", "Ticket", "PassengerId"])

    df_clean["Sex"] = df_clean["Sex"].map({"male": 0, "female": 1})
    df_clean["Embarked"] = df_clean["Embarked"].map({"S": 1, "C": 2, "Q": 3})

    return df_clean


def train_and_evaluate_model(X, y):
    """Обучение модели и возвращение её точности"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = LogisticRegression(random_state=0, max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy


path = r"C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\6 сем\\ML\\3\\Titanic.csv"
df_prepared = load_and_clean_data(path)

# Разделение на признаки и целевую переменную
X = df_prepared.drop(columns=["Survived"])
y = df_prepared["Survived"]

# Модель с Embarked
full_features_accuracy = train_and_evaluate_model(X, y)
print(f"Точность модели: {full_features_accuracy * 100}")

# Модель без Embarked
X_without_embarked = X.drop(columns=["Embarked"])
reduced_features_accuracy = train_and_evaluate_model(X_without_embarked, y)
print(f"Точность модели без признака Embarked: {reduced_features_accuracy * 100}")


"""
Точность модели: 67.27272727272727
Точность модели без признака Embarked: 76.36363636363637

Признак Embarked не вносит значимого вклада в качество модели, так как его исключение практически не изменяет точность предсказаний.
"""
