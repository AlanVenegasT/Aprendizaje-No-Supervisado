import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
wine = load_wine()
X = wine.data
y = wine.target
features = wine.feature_names

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reducción de dimensionalidad con PCA (opcional)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Visualización de los datos originales y transformados
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='Set1')
plt.title("Datos con PCA (entrenamiento)")

# Aplicación de algoritmos de clustering
# 1. K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_train_pca)

# 2. DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_train_pca)

# 3. BIRCH
birch = Birch(n_clusters=3)
birch_labels = birch.fit_predict(X_train_pca)

# Evaluación de la calidad del clustering
kmeans_silhouette = silhouette_score(X_train_pca, kmeans_labels)
dbscan_silhouette = silhouette_score(X_train_pca, dbscan_labels)
birch_silhouette = silhouette_score(X_train_pca, birch_labels)

# Imprimir las evaluaciones
print(f"K-Means Silhouette Score: {kmeans_silhouette:.2f}")
print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.2f}")
print(f"BIRCH Silhouette Score: {birch_silhouette:.2f}")

# Visualización de los clusters
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=kmeans_labels, palette='Set1')
plt.title("Clusters con K-Means")

plt.show()
