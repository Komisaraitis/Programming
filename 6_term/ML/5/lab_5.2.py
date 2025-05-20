import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from xgboost import XGBClassifier


plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.grid"] = True


data = pd.read_csv(
    "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\6 сем\\ML\\5\\diabetes.csv"
)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Функция для оценки модели
def evaluate_model(model_params, model_type="rf"):
    """Оценивает производительность модели"""
    start_time = time.time()

    if model_type == "rf":
        model = RandomForestClassifier(**model_params)
    else:
        model = XGBClassifier(**model_params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    exec_time = time.time() - start_time

    return f1, exec_time


#Анализ зависимости от глубины дерева
print("\n1. Анализ оптимальной глубины дерева")
depth_results = {"depth": [], "f1": []}

for depth in range(1, 21):
    f1, _ = evaluate_model({"max_depth": depth})
    depth_results["depth"].append(depth)
    depth_results["f1"].append(f1)

plt.figure()
plt.plot(depth_results["depth"], depth_results["f1"], "b-o", linewidth=2)
plt.xlabel("Глубина дерева")
plt.ylabel("F1-score")
plt.title("Зависимость F1-score от глубины дерева")
plt.xticks(range(1, 21, 2))
plt.show()

optimal_depth = depth_results["depth"][
    depth_results["f1"].index(max(depth_results["f1"]))
]
print(f"Оптимальная глубина: {optimal_depth} (F1 = {max(depth_results['f1']):.4f})")

#Анализ зависимости от количества признаков
print("\n2. Анализ оптимального количества признаков")
feature_results = {"n_features": [], "f1": []}

for n_features in range(1, X.shape[1] + 1):
    f1, _ = evaluate_model({"max_features": n_features})
    feature_results["n_features"].append(n_features)
    feature_results["f1"].append(f1)

plt.figure()
plt.plot(feature_results["n_features"], feature_results["f1"], "g-s", linewidth=2)
plt.xlabel("Количество признаков")
plt.ylabel("F1-score")
plt.title("Зависимость F1-score от количества признаков")
plt.xticks(range(1, X.shape[1] + 1))
plt.show()

optimal_features = feature_results["n_features"][
    feature_results["f1"].index(max(feature_results["f1"]))
]
print(
    f"Оптимальное количество признаков: {optimal_features} (F1 = {max(feature_results['f1']):.4f})"
)

#Анализ зависимости от количества деревьев
print("\n3. Анализ оптимального количества деревьев")
tree_results = {"n_trees": [], "f1": [], "time": []}

for n_trees in range(50, 101, 10):
    f1, exec_time = evaluate_model({"n_estimators": n_trees})
    tree_results["n_trees"].append(n_trees)
    tree_results["f1"].append(f1)
    tree_results["time"].append(exec_time)

fig, ax1 = plt.subplots()
ax1.plot(tree_results["n_trees"], tree_results["f1"], "b-o", label="F1-score")
ax1.set_xlabel("Количество деревьев")
ax1.set_ylabel("F1-score", color="b")
ax1.tick_params("y", colors="b")

ax2 = ax1.twinx()
ax2.plot(tree_results["n_trees"], tree_results["time"], "r--s", label="Время обучения")
ax2.set_ylabel("Время (сек)", color="r")
ax2.tick_params("y", colors="r")

plt.title("Зависимость F1-score и времени обучения от количества деревьев")
fig.legend(loc="upper right")
plt.show()

optimal_trees = tree_results["n_trees"][
    tree_results["f1"].index(max(tree_results["f1"]))
]
print(
    f"Оптимальное количество деревьев: {optimal_trees} (F1 = {max(tree_results['f1']):.4f})"
)

#Оценка XGBoost с оптимальными параметрами
print("\n4. Оценка модели XGBoost")
xgb_params = {
    "reg_alpha": 0.6,
    "reg_lambda": 0.3,
    "max_depth": 6,
    "subsample": 0.5,
    "n_estimators": 8,
    "random_state": 0,
}

xgb_f1, xgb_time = evaluate_model(xgb_params, model_type="xgb")

# Сравнение с оптимальной моделью RandomForest
rf_params = {
    "max_depth": optimal_depth,
    "max_features": optimal_features,
    "n_estimators": optimal_trees,
    "random_state": 0,
}

rf_f1, rf_time = evaluate_model(rf_params)

print("\nСравнение моделей:")
print(f"{'Модель':<15} {'F1-score':<10} {'Время обучения':<15}")
print(f"{'RandomForest':<15} {rf_f1:.4f}     {rf_time:.4f} сек")
print(f"{'XGBoost':<15} {xgb_f1:.4f}     {xgb_time:.4f} сек")

"""
ВЫВОД:
    1. Анализ зависимости F1-score от глубины дерева показывает, что максимальное значение достигается при глубине 9. Это является оптимальной глубиной.
    2. Исследование влияния количества признаков выявило, что использование 5 признаков дает наилучший результат(~0.66).
    3. При анализе зависимости от количества деревьев:
       - Наблюдается линейный рост времени обучения
       - Максимальный F1-score (0.62) достигается при 90 деревьях
       - Оптимальный компромисс между качеством и временем - 80 деревьев. За чуть меньшее время обучения F1-score изменяется очень мало.
    
    XGBoost с подобранными параметрами показал F1-score 0.6619 при времени обучения 0.05 сек,
    что превосходит результаты Random Forest.
    
    Параметры XGBoost подбирались методом последовательного перебора:
    - reg_alpha = 0.6
    - reg_lambda = 0.3
    - max_depth = 6
    - subsample = 0.5
    - n_estimators = 8
"""
