import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz


diabetes_data = pd.read_csv(
    "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\6 сем\\ML\\5\\diabetes.csv"
)
features = diabetes_data.drop("Outcome", axis=1)
target = diabetes_data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=0
)
# Логистическая регрессия
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)

# Решающее дерево
base_tree = tree.DecisionTreeClassifier()
base_tree.fit(X_train, y_train)
tree_pred = base_tree.predict(X_test)

# Вычисление метрик
metrics_data = {
    "Логистическая регрессия": {
        "Accuracy": round(metrics.accuracy_score(y_test, log_reg_pred), 4),
        "Precision": round(metrics.precision_score(y_test, log_reg_pred), 4),
        "F1": round(metrics.f1_score(y_test, log_reg_pred), 4),
    },
    "Решающее дерево": {
        "Accuracy": round(metrics.accuracy_score(y_test, tree_pred), 4),
        "Precision": round(metrics.precision_score(y_test, tree_pred), 4),
        "F1": round(metrics.f1_score(y_test, tree_pred), 4),
    },
}

# Вывод результатов сравнения
for model, scores in metrics_data.items():
    print(f"{model}:\n")
    for metric, value in scores.items():
        print(f"{metric}: {value}")
    print()

# Поиск оптимальной глубины дерева
depths = range(1, 30)
f1_scores = []

for depth in depths:
    current_tree = tree.DecisionTreeClassifier(max_depth=depth)
    current_tree.fit(X_train, y_train)
    f1_scores.append(metrics.f1_score(y_test, current_tree.predict(X_test)))

optimalModelTree = tree.DecisionTreeClassifier(max_depth=5)
optimalModelTree.fit(X_train, y_train)

dot_data = tree.export_graphviz(optimalModelTree, out_file=None)  
graph = graphviz.Source(dot_data)  
#graph.render("tree_graph")

# Визуализация зависимости F1 от глубины
plt.figure(figsize=(10, 5))
plt.plot(depths, f1_scores, marker="o", linestyle="--", color="royalblue")
plt.xlabel("Глубина дерева", fontsize=12)
plt.ylabel("F1-score", fontsize=12)
plt.title("Оптимизация глубины дерева", pad=20, fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Создание и визуализация оптимального дерева
optimal_tree = tree.DecisionTreeClassifier(max_depth=5)
optimal_tree.fit(X_train, y_train)

# Визуализация дерева
tree_plot = tree.export_graphviz(
    optimal_tree,
    out_file=None,
    feature_names=features.columns,
    filled=True,
    rounded=True,
)
graphviz.Source(tree_plot)

# Важность признаков
plt.figure(figsize=(12, 6))
plt.barh(features.columns, optimal_tree.feature_importances_, color="darkcyan")
plt.title("Важность признаков в оптимальной модели", pad=20)
plt.xlabel("Коэффициент важности")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# Кривые Precision-Recall и ROC
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

metrics.PrecisionRecallDisplay.from_estimator(optimal_tree, X_test, y_test, ax=ax1)
ax1.set_title("Precision-Recall Curve")

metrics.RocCurveDisplay.from_estimator(optimal_tree, X_test, y_test, ax=ax2)
ax2.set_title("ROC Curve")

plt.tight_layout()
plt.show()

# Исследование влияния количества признаков
max_features_range = range(1, len(features.columns) + 1)
f1_scores = []

for n_features in max_features_range:
    model = tree.DecisionTreeClassifier(max_depth=5, max_features=n_features)
    model.fit(X_train, y_train)
    f1_scores.append(metrics.f1_score(y_test, model.predict(X_test)))

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.plot(max_features_range, f1_scores, marker="s", color="indianred")
plt.xlabel("Количество используемых признаков", fontsize=12)
plt.ylabel("F1-score", fontsize=12)
plt.title("Влияние параметра max_features на качество модели", pad=20)
plt.xticks(max_features_range)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

"""
В ходе анализа были рассмотрены три ключевые метрики оценки качества моделей:

1. Accuracy (точность) - показывает общую долю верных прогнозов модели
2. Precision (прецизионность) - отражает способность алгоритма точно идентифицировать случаи положительного класса
3. F1-score (F-мера) - гармоническое среднее между precision и recall, учитывающее оба показателя

Сравнительный анализ показал, что логистическая регрессия продемонстрировала более высокие результаты по всем метрикам по сравнению с моделью решающего дерева. 
Это свидетельствует о ее большей эффективности для данного набора данных.

Для дальнейшего исследования была выбрана F1-мера, поскольку:
1. Она обеспечивает комплексную оценку качества классификации, учитывая как полноту (recall), так и точность (precision) предсказаний
2. В отличие от accuracy, F1-score более корректно оценивает модели при работе с несбалансированными классами, что особенно актуально для медицинских данных
3. Данная метрика позволяет избежать перекосов в оценке, когда одна из характеристик (precision или recall) доминирует в ущерб другой

Такой подход к выбору метрики позволяет получить более объективную и сбалансированную оценку качества работы алгоритмов классификации.


Оптимальная глубина дерева - 5.
"""
