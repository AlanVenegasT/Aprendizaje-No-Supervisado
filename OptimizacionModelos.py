# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Cargar el conjunto de datos y dividir en entrenamiento y prueba
data = datasets.load_breast_cancer()  # Conjunto de datos de clasificación binaria
X = data.data
y = data.target

# Dividimos los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Crear el modelo SVM con el kernel lineal
svm_model = SVC(kernel='linear')  # Usamos el kernel lineal inicialmente
svm_model.fit(X_train, y_train)    # Entrenamos el modelo

# 3. Evaluar el modelo
y_pred = svm_model.predict(X_test)

# Calcular las métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Mostrar los resultados
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 4. Ajustar hiperparámetros con GridSearchCV para SVM con kernel radial (rbf)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_search.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros
print("Mejores parámetros encontrados por GridSearchCV:", grid_search.best_params_)

# 5. Evaluar el mejor modelo
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Calcular las métricas de evaluación para el mejor modelo
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

# Mostrar los resultados del mejor modelo
print(f"Mejor Accuracy: {accuracy_best:.4f}")
print(f"Mejor Precision: {precision_best:.4f}")
print(f"Mejor Recall: {recall_best:.4f}")
print(f"Mejor F1 Score: {f1_best:.4f}")
