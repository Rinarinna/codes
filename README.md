**Экзамен по курсу "Машинное обучение"**

Выполнил: *Тюменцева Арина Владиславовна Б22-701*

Филиал: *НИЯУ МИФИ*
---
# 1. Исходные данные
# 1.1 Загрузка библиотек и данных

*Загрузить данные в соответствии с вариантом задания*
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# добавили вывод сразу нескольких строк под одной ячейкой
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
X.shape
y.shape
### 1.2 Описание исходных данных

*Привести описание исходных данных, описание и типы признаков (вещественные, целочисленные, категориальные и т.д.), объём выборки, особенности данных. Сформулировать решаемую задачу, определить тип задачи (регрессия / классификация), указать входные и выходные переменные.*
Рассмотрим данные которые будем использовать для обучения.
print("\nИнформация о данных:")
print(X.info())
Всего имеется 569 элементов в выборке. Все данные вещественного типа. В данных отсутствуют пропущенные значения.
Анализируемый датасет `load_breast_cancer` содержит 30 признаков, которые описывают различные характеристики клеток опухоли.

Эти признаки можно разделить на три группы: **средние значения (mean)**, **стандартные ошибки (standard error, se)** и **"наихудшие" (worst)** значения.

Описание каждого признака из данных в разрезе групп:

**Средние значения (mean)**
`mean radius`: Средний радиус клетки.

`mean texture`: Средняя текстура клетки (изменение градации серого в изображении).

`mean perimeter`: Средний периметр клетки.

`mean area`: Средняя площадь клетки.

`mean smoothness`: Средняя гладкость клетки (локальное изменение длины радиуса).

`mean compactness`: Средняя компактность клетки (периметр² / площадь - 1.0).

`mean concavity`: Средняя вогнутость клетки (степень вогнутости контура).

`mean concave points`: Среднее количество вогнутых участков клетки.

`mean symmetry`: Средняя симметрия клетки.

`mean fractal dimension`: Средняя фрактальная размерность клетки.

**Стандартные ошибки (standard error, se)**
`radius error`: Стандартная ошибка радиуса клетки.

`texture error`: Стандартная ошибка текстуры клетки.

`perimeter error`: Стандартная ошибка периметра клетки.

`area error`: Стандартная ошибка площади клетки.

`smoothness error`: Стандартная ошибка гладкости клетки.

`compactness error`: Стандартная ошибка компактности клетки.

`concavity error`: Стандартная ошибка вогнутости клетки.

`concave points error`: Стандартная ошибка количества вогнутых участков клетки.

`symmetry error`: Стандартная ошибка симметрии клетки.

`fractal dimension error`: Стандартная ошибка фрактальной размерности клетки.

**"Наихудшие" значения (worst)**
`worst radius`: Максимальный радиус клетки.

`worst texture`: Максимальная текстура клетки.

`worst perimeter`: Максимальный периметр клетки.

`worst area`: Максимальная площадь клетки.

`worst smoothness`: Максимальная гладкость клетки.

`worst compactness`: Максимальная компактность клетки.

`worst concavity`: Максимальная вогнутость клетки.

`worst concave points`: Максимальное количество вогнутых участков клетки.

`worst symmetry`: Максимальная симметрия клетки.

`worst fractal dimension`: Максимальная фрактальная размерность клетки.
print("Описание переменных:")
print(X.describe())
print("\nПервые 5 строк данных:")
print(X.head())
Изучим целевую переменную `target`, которую будем предсказывать.
print("\nРаспределение целевой переменной:")
print(y.value_counts())
Целевая переменная `target` в датасете `load_breast_cancer` является бинарной и представляет собой классификацию опухоли на доброкачественную (benign) или злокачественную (malignant), где:

`0` обозначает доброкачественную опухоль (benign).

`1` обозначает злокачественную опухоль (malignant).
### 1.3 Выборочные характеристики
*Рассчитать основные выборочные характеристики (среднее, дисперсию, среднеквадратическое отклонение, медиану и т.д.), привести объемы выборок в каждом классе (для задач классификации)*
mean_values = X.mean()
variance_values = X.var()
std_dev_values = X.std()
median_values = X.median()

print("\nСредние значения:\n", mean_values)
print("\nДисперсия:\n", variance_values)
print("\nСреднеквадратическое отклонение:\n", std_dev_values)
print("\nМедиана:\n", median_values)
Всего для расчёта описательных статистик у нас имелось 569 наблюдений в данных с заполненнными значениями.
### 1.4 Исследование распределений признаков и откликов

*Построить гистограммы распределения и диаграммы Box-and-Whisker (для отдельных признаков при большом их числе), сделать выводы о характере распределений признаков (для задач классификации - в классах), наличии выбросов и т.п.*
# Histograms
X.hist(figsize=(20, 15))
_ = plt.suptitle('Гистограммы распределения признаков')
plt.show()
# Box-and-Whisker plots
X.plot(kind='box', subplots=True, layout=(5,6), figsize=(20,15), sharex=False, sharey=False)
_ = plt.suptitle('Диаграммы Box-and-Whisker для признаков')
plt.show()
### 1.5 Корреляционный анализ данных

*Визуализировать диаграммы рассеяния и корреляционную матрицу признаков, сделать выводы*
# Correlation matrix
corr_matrix = X.corr()
_ = plt.figure(figsize=(15, 10))
_ = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
_ = plt.title('Корреляционная матрица признаков')
plt.show()
# Pairplot for scatter plots
sns.pairplot(X)
_ = plt.suptitle('Диаграммы рассеяния признаков')
plt.show()
### 1.6 Выводы

*Сделать выводы по результатам предварительного визуального анализа исходных данных*
print("\nВыводы:")
print("Данные имеют разнообразные распределения, наблюдаются корреляции между некоторыми признаками.")
print("Признаки, такие как mean radius и mean area, имеют высокую корреляцию.")
print("Некоторые признаки имеют выбросы, что видно на диаграммах Box-and-Whisker.")
Можно еще посмотреть в разрезе классов (0 и 1) как у нас меняеются размеры клетко и другие характеристики и сравнить их между собой.
---
# 2. Предобработка данных
### 2.1 Очистка данных

*а) Обнаружение и устранение дубликатов*\
*б) Обнаружение и устранение выбросов*\
*в) Устранение/восстановление пропущенных значений*
# Detect duplicates
duplicates = X.duplicated().sum()
print(f"\nКоличество дубликатов: {duplicates}")
# Handle outliers (example: remove if beyond 3 standard deviations)
X_cleaned = X[(np.abs(X - X.mean()) <= (3*X.std())).all(axis=1)]
X_cleaned.shape
X.shape
#Осталось 495 наблюдений вместо 569 после удаления выбросов из данных.
#И с учётом, что у нас стало меньше наблюдений для обучения, то надо также удалить метки классов для тех объектов, которые удалили из выборки.
y_cleaned = y[X_cleaned.index]
y_cleaned.shape
#В данных нет пропущенных значений (как в исходных, так и в отфильтрованных, поэтому заполнять пропущенные значения не нужно).

#Но если была бы необходимость заполнить пропущенные значения, то реализовали бы это следующим кодом:

`X_filled = X_cleaned.fillna(X_cleaned.mean())`
X_cleaned.info()
### 2.2 Разбиение данных на обучающую и тестовую выборки

*Разбить данные на обучающую и тестовую выборки в отношении 70/30*
SEED = 42
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.3, random_state=SEED)
print(X_train_cleaned.shape, X_test_cleaned.shape, y_train_cleaned.shape, y_test_cleaned.shape, sep='\n')
Здесь разделяем оригинальный датасет с выбросами.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, sep='\n')
Также при работе с моделями машинного обучения для получения более надежных оценок метрик можно использовать кросс-валидацию.
# # Инициализация StratifiedKFold
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Обучение модели и оценка производительности
# accuracies = []

# for train_index, test_index in skf.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracies.append(accuracy)

# print(f'Mean Accuracy: {np.mean(accuracies)}')
### 2.3 Преобразование данных
*Описать используемые способы преобразования входных и выходных переменных, привести обоснования выбранных способов преобразования, применить преобразования к обучающей и тестовой выборкам*
X_cleaned.describe()
После просмотра описательных статистик по каждому признаку видим, что некоторые признаки в разных масштабах, что может сказаться на качестве и производительности итоговой модели.

Поэтому произведем масштабирование признаков с использованием алгоритма `StandardScaler`.
# 2.3 Преобразование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Делаем преобразование на данных без выбросов.
clean_scaler = StandardScaler()
X_train_scaled_сlean = scaler.fit_transform(X_train_cleaned)
X_test_scaled_clean = scaler.transform(X_test_cleaned)
X_train_scaled
X_train_scaled_сlean
---
# 3. Построение и исследование модели машинного обучения
### 3.1 Обучение модели

*Выбрать модель и алгоритм машинного обучения для решения поставленной задачи, привести обоснование выбора, обучить модель на обучающей выборке*
В качестве базового решения (baseline) используем модель `LogisticRegression`.
model = LogisticRegression(random_state=SEED)
model.fit(X_train_scaled, y_train)
model_cleaned_data = LogisticRegression()
model_cleaned_data.fit(X_train_scaled_сlean, y_train_cleaned)
### 3.2 Оценка качества модели

**Для задач регрессии:**
* *построить диаграммы рассеяния в пространстве «выход модели – желаемый выход» на данных обучающей и тестовой выборок*
* *построить линейные регрессии выхода модели на желаемый выход*
* *рассчитать коэффициенты детерминации линейных регрессионных моделей для обучающей и тестовой выборок*
* *построить гистограммы распределения ошибок модели.*

**Для задач классификации:**
* *построить матрицы ошибок (confusion matrix) классификатора и рассчитать показатели качества классификации (чувствительность, специфичность, точность, F-мера, каппа Коэна) на обучающей и тестовой выборках.*
y_pred = model.predict(X_test_scaled)
print("\nМатрица ошибок на оригинальных данных:")
print(confusion_matrix(y_test, y_pred))

print("\nОтчет по классификации на оригинальных данных:")
print(classification_report(y_test, y_pred))
y_pred_cleaned = model.predict(X_test_scaled_clean)
print("\nМатрица ошибок на данных где удалили выбросы:")
print(confusion_matrix(y_test_cleaned, y_pred_cleaned))

print("\nОтчет по классификации где удалили выбросы:")
print(classification_report(y_test_cleaned, y_pred_cleaned))


### 3.3 Исследование модели и алгоритма обучения

*Провести экспериментальные исследования модели, построить графики зависимости ошибки модели от ее архитектурных параметров и гиперпараметров алгоритма обучения, построить ROC-кривые, оценить степень важности признаков и пр.*
Построение ROC-кривой.
y_pred_proba = model.predict_proba(X_test_scaled)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
_ = plt.plot(fpr, tpr)
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
plt.show()

print("\nAUC Score:", roc_auc_score(y_test, y_pred_proba))
model.coef_
Оценка степени важности признаков.
coef = model.coef_[0]
feature_names = data.feature_names

# Сортировка признаков по важности
sorted_idx = coef.argsort()

# Построение графика важности признаков
_ = plt.barh(range(len(sorted_idx)), coef[sorted_idx], align='center')
_ = plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
_ = plt.xlabel('Feature Importance')
_ = plt.title('Feature Importances')
plt.show()
Графики зависимости ошибки модели от гиперпараметров.
# Графики зависимости ошибки модели от гиперпараметров
C_range = [0.001, 0.01, 0.1, 1, 10, 100]

for C in C_range:
    lr_model = LogisticRegression(C=C, random_state=SEED, max_iter=1000)
    _ = lr_model.fit(X_train_scaled, y_train)
    y_pred = lr_model.predict(X_test_scaled)
    print(f"\nМатрица ошибок для С = {C}:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nОтчет по классификации для С = {C}:")
    print(classification_report(y_test, y_pred))
### 3.4 Улучшение решения

*Предложить возможное улучшение точности решения задачи (выбрать другой тип модели, алгоритм или критерий обучения, сформулировать рекомендации по возможным способам повышения точности модели), обучить модель и сравнить показатели точности с рассчитанными в п.3.2*
Попробоуем в качестве улучшения решения модели `SVM` и модель из класса решающих деревьев `RandomForestClassifier`.
svm_model = SVC(probability=True)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_svm_proba = svm_model.predict_proba(X_test_scaled)[:,1]

print("\nМатрица ошибок для SVM:")
print(confusion_matrix(y_test, y_pred_svm))

print("\nОтчет по классификации для SVM:")
print(classification_report(y_test, y_pred_svm))

print("\nAUC Score для SVM:", roc_auc_score(y_test, y_pred_svm_proba))
rf_model = RandomForestClassifier(random_state=SEED)
rf_model.fit(X_train_scaled, y_train)


y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nМатрица ошибок для RandomForestClassifier:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nОтчет по классификации для RandomForestClassifier:")
print(classification_report(y_test, y_pred_rf))

print("\nAUC Score для RandomForestClassifier:", roc_auc_score(y_test, y_pred_rf_proba))
Можно еще добавить подбор параметров по сетке.
# Определение сетки параметров для GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1, 10]
}

# Создание и обучение модели SVM с использованием GridSearchCV
svm_model = SVC(probability=True)
grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='roc_auc')
grid_search.fit(X_train_scaled, y_train)

# Вывод лучших параметров
print("Лучшие параметры:", grid_search.best_params_)

# Предсказание классов на тестовой выборке с использованием лучшей модели
best_svm_model = grid_search.best_estimator_
y_pred_svm = best_svm_model.predict(X_test_scaled)

# Предсказание вероятностей для положительного класса на тестовой выборке
y_pred_svm_proba = best_svm_model.predict_proba(X_test_scaled)[:, 1]

# Вывод матрицы ошибок для SVM
print("\nМатрица ошибок для SVM:")
print(confusion_matrix(y_test, y_pred_svm))

# Вывод отчета по классификации для SVM
print("\nОтчет по классификации для SVM:")
print(classification_report(y_test, y_pred_svm))

# Вывод AUC Score для SVM
print("\nAUC Score для SVM:", roc_auc_score(y_test, y_pred_svm_proba))
### 3.5 Выводы

*Сделать выводы по результатам проведенных исследований*
print("\nВыводы:")
print("Модель логистической регрессии показала хороший результат.")
print("SVM модель также демонстрирует высокое качество, но может быть более чувствительна к параметрам.")
print("Удаление выбросов привело к ухудшению качества модели, поэтому мы решили их оставить, что в итоге улучшило результаты. Это показывает, что выбросы могут содержать важную информацию для предсказания. \n")
print("Настройка параметров SVM модели позволила минимизировать количество ошибок, что является ключевым моментом задачи, особенно в контексте предотвращения предсказания злокачественных опухолей как доброкачественных.")
