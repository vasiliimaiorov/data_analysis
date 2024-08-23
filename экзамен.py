import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import zscore
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


data = pd.read_csv("clustering_1_3008.csv")

# 1.1  входные и выходные переменные
input_vars = data.drop("income", axis=1)
output_var = data["income"]

# Пропущенные значения
missing_values = data.isnull().sum()

# 1.3 Визуализация данных
# Для количественных признаков
numerical_features = data.select_dtypes(include=['float64'])
sns.pairplot(numerical_features)
plt.show()

# Для категориальных признаков
categorical_features = data.select_dtypes(include=['object']).columns
for col in categorical_features:
    sns.countplot(x=col, data=data)
    plt.show()

# 1.4 Для количественных переменных
for col in numerical_features.columns:
    sns.boxplot(x=col, data=data)
    plt.show()
    print(f"Статистика для {col}:\n{data[col].describe()}\n")

quantitative_vars = ['age', 'income']
# Размах
range_values = data[quantitative_vars].max() - data[quantitative_vars].min()
print(f"Размах:\n{range_values}\n")

# Число уникальных значений
unique_counts = data[quantitative_vars].nunique()
print(f"Число уникальных значений:\n{unique_counts}")

# Часть 2: Построение базовой модели машинного обучения

# 2.1 Устранение строк с пропущенными значениями
data = data.dropna()

# 2.2 OneHot-кодирование номинальных признаков
categorical_features = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# 2.4 Визуализация корреляционной матрицы признаков
correlation_matrix = data_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

print("Корреляционная матрица признаков:\n", correlation_matrix)

# 2.5 Разбиение выборки на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(
    data_encoded, data['income'], test_size=0.3, random_state=42
)

# 2.6. Масштабирование данных перед применением PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2.6.1. Факторизация обучающих данных с использованием метода главных компонент
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# 2.7. Построение диаграммы рассеяния для двух главных компонент
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.title('Диаграмма рассеяния для двух главных компонент')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.show()

# Базовая модель машинного обучения (логистическая регрессия в данном случае)
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 2.8. Проведение кластеризации данных в пространстве двух главных компонентов на 3 кластера
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_train_pca)

# Расчет коэффициента силуэта
silhouette_avg = silhouette_score(X_train_pca, clusters)
print(f"Средний коэффициент силуэта: {silhouette_avg}")

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# Дополнительные метрики качества кластеризации
davies_bouldin = davies_bouldin_score(X_train_pca, clusters)
calinski_harabasz = calinski_harabasz_score(X_train_pca, clusters)

print(f"Davies-Bouldin Index: {davies_bouldin}")
print(f"Calinski-Harabasz Index: {calinski_harabasz}")

# Визуализация результатов кластеризации
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=clusters, cmap='viridis')
plt.title('Результаты кластеризации в пространстве двух главных компонент')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.show()

# Оценка степени важности признаков
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': coefficients})
feature_importance['AbsoluteCoefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values(by='AbsoluteCoefficient', ascending=False)

# Визуализация степени важности признаков
plt.figure(figsize=(12, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Степень важности признаков')
plt.show()

# Импьютация пропущенных значений медианой
X_train_imputed = X_train.copy()
X_train_imputed[numerical_features.columns] = X_train_imputed[numerical_features.columns].apply(lambda x: x.fillna(x.median()))

# Проверка, что пропущенных значений больше нет
missing_values_imputed = X_train_imputed.isnull().sum()

# 3.2.# Определение и удаление выбросов с использованием z-оценки
z_scores = np.abs(zscore(X_train_imputed[numerical_features.columns]))
outliers_mask = (z_scores < 3).all(axis=1) # Порог z-оценки 3

X_train_no_outliers_zscore = X_train_imputed[outliers_mask]

# 3.3. Преобразование переменных
# Масштабирование числовых переменных перед применением PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_no_outliers_zscore)

# Факторизация обучающих данных с использованием метода главных компонент
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# 3.4. Отбор признаков
# Вычисление корреляции признаков с целевой переменной
correlations = data_encoded.corrwith(pd.Series(data['income'], name='income'))
# Отбор признаков с корреляцией выше порогового значения
selected_features = correlations[abs(correlations) > 0.1].index
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 3.5. Выбор модели и алгоритма машинного обучения
# Используем логистическую регрессию для базовой модели
model_base = LogisticRegression(random_state=42)
model_base.fit(X_train, y_train)

# Для модифицированной модели используем, например, метод опорных векторов (SVM)
model_modified = SVC(random_state=42)
model_modified.fit(X_train_selected, y_train)

# 3.6. Обучение моделей и рассчет показателей качества
# Для базовой модели
y_pred_base = model_base.predict(X_test)

# Для модифицированной модели
y_pred_modified = model_modified.predict(X_test_selected)

# 3.8. Расчет метрик качества на тестовой выборке для базовой модели
precision_base_test = precision_score(y_test, y_pred_base, average='weighted')
recall_base_test = recall_score(y_test, y_pred_base, average='weighted')
f1_base_test = f1_score(y_test, y_pred_base, average='weighted')

# Вывод метрик для базовой модели
print("Базовая молель:")
print(f"Precision: {precision_base_test}")
print(f"Recall: {recall_base_test}")
print(f"F1-Score: {f1_base_test}")

# Вывод метрик для модифицированной модели
precision_modified = precision_score(y_test, y_pred_modified, average='weighted')
recall_modified = recall_score(y_test, y_pred_modified, average='weighted')
f1_modified = f1_score(y_test, y_pred_modified, average='weighted')

print("Модифицированная модель:")
print(f"Precision: {precision_modified}")
print(f"Recall: {recall_modified}")
print(f"F1-Score: {f1_modified}")

if f1_modified > f1_base_test:
    print("Модифицированная модель показывает лучшие результаты.")
else:
    print("Базовая модель остается более точной.")
