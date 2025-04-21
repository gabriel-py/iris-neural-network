import joblib
import numpy as np

from activations import sigmoid

def classify_flower(flower_data):
    """
    Recebe os dados de uma flor da íris (array de 4 valores) e retorna a classe prevista (0, 1 ou 2).
    """

    if isinstance(flower_data, list):
        flower_data = np.array(flower_data)

    if flower_data.ndim == 1:
        flower_data = flower_data.reshape(1, -1)

    # Carrega pesos e scaler treinados
    data = np.load("pesos_iris.npz")
    W1, b1 = data["W1"], data["b1"]
    W2, b2 = data["W2"], data["b2"]
    scaler = joblib.load("scaler_iris.pkl")

    # Pré-processa a flor com o mesmo scaler usado no treinamento
    X_scaled = scaler.transform(flower_data)

    # Forward pass
    z1 = np.dot(X_scaled, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Retorna a classe com maior probabilidade
    predicted_class = int(np.argmax(a2, axis=1)[0])
    return predicted_class


if __name__ == "__main__":
    example_flower = [5.1, 3.5, 1.4, 0.2]  # Esperado: Classe 0 (Setosa)

    result = classify_flower(example_flower)

    print(f"Classe prevista: {result} (0 = Setosa, 1 = Versicolor, 2 = Virginica)")
