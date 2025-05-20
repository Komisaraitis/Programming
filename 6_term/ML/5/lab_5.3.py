import pandas as pd
import numpy as np
import time

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

from scipy.stats import uniform

# Загрузка данных
data = pd.read_csv(
    "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\6 сем\\ML\\5\\diabetes.csv"
)
X, y = data.drop("Outcome", axis=1), data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ====== Базовая модель ======
initial_model = XGBClassifier(
    reg_alpha=0.6, reg_lambda=0.3, max_depth=6, subsample=0.5, n_estimators=8
)

start_time = time.time()
initial_model.fit(X_train, y_train)
training_time = time.time() - start_time
y_pred = initial_model.predict(X_test)

# ====== Задание 1: Random Search ======

# Пространство гиперпараметров
search_space = {
    "n_estimators": np.arange(5, 100, 5),
    "max_depth": np.arange(3, 10),
    "learning_rate": uniform(0, 1),
    "subsample": uniform(0, 1),
    "colsample_bytree": uniform(0, 1),
    "reg_alpha": uniform(0, 1),
    "reg_lambda": uniform(0, 1),
    "gamma": uniform(0, 1),
}

base_model = XGBClassifier(random_state=0)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=search_space,
    n_iter=50,
    scoring="f1",
    cv=5,
    verbose=1,
    random_state=0,
    n_jobs=-1,
)

start_time = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

print("Лучшие параметры:", random_search.best_params_)
print("Лучший F1-score:", random_search.best_score_)
print(f"Время выполнения: {random_time:.2f} секунд")

# ====== Задание 2: TPE ======

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Пространство поиска для Hyperopt
tpe_space = {
    "n_estimators": hp.choice("n_estimators", list(range(5, 100, 5))),
    "max_depth": hp.choice("max_depth", list(range(3, 10))),
    "learning_rate": hp.uniform("learning_rate", 0, 1.0),
    "subsample": hp.uniform("subsample", 0, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0, 1.0),
    "reg_alpha": hp.uniform("reg_alpha", 0, 1.0),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1.0),
    "gamma": hp.uniform("gamma", 0, 0.5),
}


# Целевая функция для оптимизации
def objective(params):
    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        gamma=params["gamma"],
        random_state=0,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    return {"loss": -f1, "status": STATUS_OK}


# Запуск TPE
trials = Trials()
rng = np.random.default_rng(0)

start_time = time.time()
best_params = fmin(
    fn=objective,
    space=tpe_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials,
    rstate=rng,
)
tpe_time = time.time() - start_time

best_f1 = -trials.best_trial["result"]["loss"]

print("Лучшие параметры:", best_params)
print(f"Лучший F1-score: {best_f1:.4f}")
print(f"Время выполнения: {tpe_time:.2f} секунд")


"""
Вывод:
Алгоритм TPE показал более высокую эффективность по сравнению с алгоритмом Random Search как по времени выполнения, так и по качеству модели.
Время выполнения TPE составило 2.69 секунды, тогда как Random Search занял значительно больше — 6.45 секунд.
Кроме того, наилучшее значение F1-score при использовании TPE составило 0.6667, что выше, чем результат 0.6472, полученный при Random Search.
Таким образом, TPE оказался более предпочтительным подходом для подбора гиперпараметров в данной задаче.
"""
