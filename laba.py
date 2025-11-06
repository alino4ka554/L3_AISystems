import pandas as pd

file_path = "data.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=['Unnamed: 32', 'id'])
# эквивалентно:
# df = df.loc[:, df.columns[10:]]
df.info()
df.describe()
import matplotlib.pyplot as plt

df.hist(bins=10, figsize=(20, 20))
plt.suptitle("Гистограммы распределений признаков и целевой переменной")
plt.show()
from sklearn.preprocessing import StandardScaler, MinMaxScaler

map_diagnosis = {'M':1, 'B':0}

df['diagnosis'] = df['diagnosis'].map(map_diagnosis)

scaler = StandardScaler()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
# scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# print(y_test.count())

metrics_dict = {'Model': [], 'Accuracy': [], 'Recall': [], 'Precision': [], 'F1': [], 'ROC_AUC': []}


def build_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['B', 'M'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    
    if model_name == "DecisionTree":
        plt.figure(figsize=(24, 12))
        plot_tree(
            model_tree,
            filled=True,
            feature_names=X.columns,
            class_names=['Malignant', 'Benign'],
            fontsize=10
        )
        plt.show()
        
    metrics_dict['Model'].append(model_name)
    metrics_dict['Accuracy'].append(accuracy)
    metrics_dict['Recall'].append(recall)
    metrics_dict['Precision'].append(precision)
    metrics_dict['F1'].append(f1)
    metrics_dict['ROC_AUC'].append(roc_auc)
model_gaussianNB = GaussianNB()
build_model(model_gaussianNB, "GaussianNB")
smoothing = [0, 1, 2, 3, 4, 5]
for sm in smoothing:
    print("var_smoothing  =", sm)
    model = GaussianNB(var_smoothing=sm)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
model_tree = DecisionTreeClassifier()
build_model(model_tree, "DecisionTree")
criterion = ['entropy', 'gini']

for cr in criterion:
    for i in range (1,5):
        model = DecisionTreeClassifier(
            criterion=cr, 
            max_depth=i, 
            random_state=42
        )
        print(f"criterion: {cr}, max_depth = {i}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print("Accuracy: ", accuracy_score(y_test, y_pred))
model_lda = LinearDiscriminantAnalysis()
build_model(model_lda, "LDA")
solvers = ['svd', 'lsqr', 'eigen']
shrinkages = [None, 'auto', 0.5] 

for s in solvers:
    for sh in shrinkages:
        if s == 'svd' and sh is not None:
            continue
        model = LinearDiscriminantAnalysis(solver=s, shrinkage=sh)
        print(f"solver: {s}, shrinkage={sh}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print()

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model_svc = SVC()

build_model(model_svc, "SVC")

c = [0.5, 1, 2]
kernels = ['rbf', 'linear', 'poly', 'sigmoid']
gamma = ['auto', 'scale']
for i in c:
    for k in kernels:
        for g in gamma:
            if k != 'linear':
                model = SVC(C=i, kernel=k, gamma=g)
                print(f"C = {i}, kernel: {k}, gamma: {g}")
            else: 
                model = SVC(C=i, kernel=k, gamma=g)
                print(f"C = {i}, kernel: {k}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(confusion_matrix(y_test, y_pred))
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print()

model_kneighbors = KNeighborsClassifier()
build_model(model_kneighbors, "KNeighbors")

weights = ['uniform', 'distance']
algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
neighbors = [5, 10, 15]
for n in neighbors:
    for w in weights:
        for a in algorithms:
            model = KNeighborsClassifier(n_neighbors=n, weights=w, algorithm=a)
            print(f"n_neighbors: {n}, weights: {w}, algorithm: {a}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(confusion_matrix(y_test, y_pred))
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print()

import matplotlib.pyplot as plt

# Преобразуем словарь в DataFrame, если ещё не сделали
metrics_df = pd.DataFrame(metrics_dict)

# Список метрик для отображения
metric_names = ['Accuracy', 'Recall', 'Precision', 'F1', 'ROC_AUC']

plt.figure(figsize=(18, 10))

for i, metric in enumerate(metric_names, 1):
    plt.subplot(2, 3, i)
    plt.bar(metrics_df['Model'], metrics_df[metric])
    plt.title(metric)
    plt.ylim(0, 1)  # Метрики от 0 до 1
    # Подписи над столбцами
    for j, val in enumerate(metrics_df[metric]):
        plt.text(j, val + 0.02, f'{val:.2f}', ha='center', va='bottom')
    plt.ylabel(metric)

plt.tight_layout()
plt.show()

# Полный скрипт: обучение, эксперименты, TensorBoard, визуализация
import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# ---------- 0. ПАРАМЕТРЫ / НАСТРОЙКИ ----------
RANDOM_STATE = 42
BASE_LOG_DIR = "logs_tensorboard"   # базовая папка для логов (меняй при необходимости)

# ---------- 1. ПОДГОТОВКА ДАННЫХ ----------
# Ожидается, что df уже загружен в окружение и содержит столбец 'diagnosis'
# Если у тебя датафрейм называется по-другому, замени имя.
try:
    df  # если df есть в namespace
except NameError:
    raise RuntimeError("В окружении не найден DataFrame `df`. Помести сюда DataFrame с колонкой 'diagnosis'.")

# Признаки и целевая
X = df.drop(columns=['diagnosis'])
y = df['diagnosis'].map({'B':0, 'M':1}) if df['diagnosis'].dtype == object else df['diagnosis']

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Стандартизация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Приведение к numpy float32 для TF
X_train_np = np.asarray(X_train, dtype=np.float32)
X_test_np  = np.asarray(X_test, dtype=np.float32)
y_train_np = np.asarray(y_train, dtype=np.float32)
y_test_np  = np.asarray(y_test, dtype=np.float32)

# ---------- 2. УТИЛИТЫ: очистка логов, генерация имени запуска ----------
def prepare_log_dir(base=BASE_LOG_DIR):
    # Удалим старую папку только если она является файлом (или по желанию всю директорию)
    if os.path.exists(base):
        # если хочешь сохранять старые логи — закомментируй следующий блок
        try:
            shutil.rmtree(base)
        except Exception:
            # если вдруг ошибка, удаляем файл
            if os.path.isfile(base):
                os.remove(base)
    os.makedirs(base, exist_ok=True)
    return base

def run_log_dir(name):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(BASE_LOG_DIR, f"{name}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path

prepare_log_dir()

# ---------- 3. Фабрика модели ----------
def build_model(input_shape,
                architecture='small',
                activation='relu',
                dropout=0.2,
                l2=0.0,
                learning_rate=1e-3):
    """
    architecture: 'small' (2 dense), 'medium' (3 dense), 'large' (4 dense)
    activation: 'relu', 'tanh', 'elu' и т.д.
    dropout: dropout rate
    l2: коэффициент L2 регуляризации
    """
    inp = layers.Input(shape=(input_shape,))
    x = inp

    # базовые архитектуры
    if architecture == 'small':
        units = [32, 16]
    elif architecture == 'medium':
        units = [64, 32, 16]
    elif architecture == 'large':
        units = [128, 64, 32, 16]
    else:
        raise ValueError("architecture must be 'small','medium' or 'large'")

    for u in units:
        x = layers.Dense(u, activation=activation,
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# ---------- 4. Эксперименты: сетка гиперпараметров ----------
# Подбери наборы, которые хочешь протестировать — ниже пример небольшой сетки.
param_grid = [
    {'architecture':'small',  'activation':'relu', 'dropout':0.2, 'l2':1e-4, 'lr':1e-3, 'batch_size':32, 'epochs':50, 'name':'small_relu'},
    {'architecture':'small',  'activation':'tanh', 'dropout':0.2, 'l2':1e-4, 'lr':1e-3, 'batch_size':32, 'epochs':50, 'name':'small_tanh'},
    {'architecture':'medium', 'activation':'relu', 'dropout':0.3, 'l2':1e-4, 'lr':1e-3, 'batch_size':32, 'epochs':80, 'name':'medium_relu'},
    {'architecture':'medium', 'activation':'relu', 'dropout':0.2, 'l2':1e-4, 'lr':5e-4, 'batch_size':16, 'epochs':80, 'name':'medium_lr5e-4'},
    {'architecture':'large',  'activation':'relu', 'dropout':0.3, 'l2':1e-4, 'lr':1e-3, 'batch_size':16, 'epochs':100, 'name':'large_relu'},
]

results = []  # сюда будут складываться итоговые метрики и параметры
histories = {}  # сохраним history для каждого прогона для визуализации

# ---------- 5. Цикл запусков ----------
for params in param_grid:
    run_name = params.get('name', 'run')
    logdir = run_log_dir(run_name)
    print("\n" + "="*60)
    print(f"Запуск: {run_name}")
    print("Параметры:", params)
    print("Логи в:", logdir)

    # модель
    model = build_model(input_shape=X_train_np.shape[1],
                        architecture=params['architecture'],
                        activation=params['activation'],
                        dropout=params['dropout'],
                        l2=params['l2'],
                        learning_rate=params['lr'])

    # callbacks
    tb_cb = callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    ckpt_path = os.path.join(logdir, "best_model.h5")
    mc_cb = callbacks.ModelCheckpoint(ckpt_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1)
    es_cb = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=12, restore_best_weights=True, verbose=1)

    # fit
    history = model.fit(
        X_train_np, y_train_np,
        validation_split=0.2,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=[tb_cb, mc_cb, es_cb],
        verbose=2
    )

    # оценка на тесте
    preds = model.predict(X_test_np).ravel()
    y_pred_label = (preds >= 0.5).astype(int)
    acc = accuracy_score(y_test_np, y_pred_label)
    auc = roc_auc_score(y_test_np, preds)
    cm = confusion_matrix(y_test_np, y_pred_label)

    print(f"Test Accuracy: {acc:.4f}, Test AUC: {auc:.4f}")
    print("Confusion matrix:\n", cm)

    # сохраняем результаты
    results.append({
        'run': run_name,
        'params': params,
        'test_accuracy': acc,
        'test_auc': auc,
        'confusion_matrix': cm
    })

    histories[run_name] = history.history

# ---------- 6. Таблица результатов ----------
results_df = pd.DataFrame([{
    'run': r['run'],
    'architecture': r['params']['architecture'],
    'activation': r['params']['activation'],
    'dropout': r['params']['dropout'],
    'l2': r['params']['l2'],
    'lr': r['params']['lr'],
    'batch_size': r['params']['batch_size'],
    'epochs': r['params']['epochs'],
    'test_accuracy': r['test_accuracy'],
    'test_auc': r['test_auc']
} for r in results])

results_df = results_df.sort_values(by='test_auc', ascending=False).reset_index(drop=True)
print("\nИтоги экспериментов (отсортированы по test_auc):")
print(results_df)

# ---------- 7. Визуализация: графики обучения для каждого прогона ----------
def plot_history(history, run_name):
    # ожидаем словарь history, содержащий 'loss','val_loss','auc','val_auc' (у нас есть auc)
    fig, axes = plt.subplots(1, 2, figsize=(14,4))
    # loss
    axes[0].plot(history.get('loss', []), label='train_loss')
    axes[0].plot(history.get('val_loss', []), label='val_loss')
    axes[0].set_title(f'Loss: {run_name}')
    axes[0].legend()
    # auc
    if 'auc' in history:
        axes[1].plot(history.get('auc', []), label='train_auc')
        axes[1].plot(history.get('val_auc', []), label='val_auc')
        axes[1].set_title(f'AUC: {run_name}')
        axes[1].legend()
    plt.show()

# Покажем кривые для каждого прогона
for rn, hist in histories.items():
    plot_history(hist, rn)

# ---------- 8. Cравнительный график (test_auc) ----------
plt.figure(figsize=(8,5))
plt.bar(results_df['run'], results_df['test_auc'])
for i, v in enumerate(results_df['test_auc']):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center')
plt.ylim(0,1)
plt.title("Сравнение run по test AUC")
plt.xticks(rotation=45)
plt.show()

# ---------- 9. Как запустить TensorBoard (локально) ----------
print("\nЗапуск TensorBoard в терминале (копируй и запускай в shell):")
print(f"tensorboard --logdir {BASE_LOG_DIR} --port 6006")
print("Затем открой в браузере: http://localhost:6006")

# ---------- 10. Сохранение результата в CSV ----------
results_df.to_csv("tf_experiment_results.csv", index=False)
print("Результаты сохранены в tf_experiment_results.csv")
