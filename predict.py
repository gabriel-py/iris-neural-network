import numpy as np
import joblib

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return a2

def predict_nova_flor(features):
    """
    Faz a predição da classe de uma nova flor Iris.
    
    Parâmetros:
    - features: lista ou array com 4 valores [sepal_length, sepal_width, petal_length, petal_width]

    Retorno:
    - Índice da classe prevista (0, 1 ou 2)
    - Lista com as probabilidades de cada classe
    """

    # Garantir que seja array 2D
    flor = np.array(features).reshape(1, -1)

    # Carregar scaler e pesos
    scaler = joblib.load("scaler_iris.pkl")  # precisa ter sido salvo anteriormente
    pesos = np.load("pesos_iris.npz")

    W1 = pesos["W1"]
    b1 = pesos["b1"]
    W2 = pesos["W2"]
    b2 = pesos["b2"]

    # Normalizar
    flor_norm = scaler.transform(flor)

    # Forward pass
    output = forward_pass(flor_norm, W1, b1, W2, b2)
    classe = np.argmax(output)
    probabilidades = output.flatten().round(4).tolist()

    return classe, probabilidades
