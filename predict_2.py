import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Funções ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return a2

# --- Carregando o dataset Iris da UCI ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
colunas = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(url, header=None, names=colunas).dropna()

# Prepara X e y
X = df.iloc[:, :4].values
classe_map = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
y_true = df["class"].map(classe_map).values

# --- Carregando scaler e pesos ---
scaler = joblib.load("scaler_iris.pkl")
X_norm = scaler.transform(X)

pesos = np.load("pesos_iris.npz")
W1 = pesos["W1"]
b1 = pesos["b1"]
W2 = pesos["W2"]
b2 = pesos["b2"]

# --- Predição para os 150 dados ---
output = forward_pass(X_norm, W1, b1, W2, b2)
y_pred = np.argmax(output, axis=1)

# --- Avaliação completa ---
print("✅ Acurácia total:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
print("\n✅ Classification Report:")
print(classification_report(y_true, y_pred, target_names=classe_map.keys()))

print("\n✅ Matriz de confusão:")
print(confusion_matrix(y_true, y_pred))
