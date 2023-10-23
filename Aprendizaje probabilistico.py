import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Elemento 1: Modelado de Incertidumbre
# Generamos datos de ejemplo (dos clases) con cierta incertidumbre.
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Elemento 4: Aprendizaje Automático Probabilístico
# Creamos un modelo de clasificación bayesiana ingenua (Naive Bayes).
model = GaussianNB()

# Elemento 5: Toma de Decisiones Probabilística
# Dividimos los datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entrenamos el modelo
model.fit(X_train, y_train)

# Realizamos predicciones
y_pred = model.predict(X_test)

# Elemento 6: Robustez y Adaptabilidad (Opcional)
# Elemento 7: Control y Toma de Decisiones (Opcional)

# Calculamos la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Elemento 8: Redes Neuronales Probabilísticas (Opcional)

# Imprimimos la precisión
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
