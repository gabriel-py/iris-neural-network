import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# --- Funções auxiliares ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def one_hot_encode(y):
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit_transform(y.reshape(-1, 1))

# --- Carregando e preparando dados ---
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_encoded = one_hot_encode(y)

# --- Cross-validation (k=5) ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

# Para salvar os melhores pesos
melhor_acc = 0
melhor_W1 = melhor_W2 = melhor_b1 = melhor_b2 = None

for fold, (train_index, test_index) in enumerate(kf.split(X_scaled)):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    np.random.seed(42)
    input_size = 4
    hidden_size = 6
    output_size = 3

    W1 = np.random.rand(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.rand(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    learning_rate = 0.1
    epochs = 3000

    for epoch in range(epochs):
        z1 = np.dot(X_train, W1) + b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        error = y_train - a2
        loss = np.mean(np.square(error))

        d_a2 = error * sigmoid_deriv(a2)
        d_W2 = np.dot(a1.T, d_a2)
        d_b2 = np.sum(d_a2, axis=0, keepdims=True)

        d_a1 = np.dot(d_a2, W2.T) * sigmoid_deriv(a1)
        d_W1 = np.dot(X_train.T, d_a1)
        d_b1 = np.sum(d_a1, axis=0, keepdims=True)

        W2 += learning_rate * d_W2
        b2 += learning_rate * d_b2
        W1 += learning_rate * d_W1
        b1 += learning_rate * d_b1

    # Avaliação
    z1_test = np.dot(X_test, W1) + b1
    a1_test = sigmoid(z1_test)
    z2_test = np.dot(a1_test, W2) + b2
    a2_test = sigmoid(z2_test)

    y_pred = np.argmax(a2_test, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    accuracies.append(acc)
    print(f"Fold {fold+1} - Acurácia: {acc:.4f}")

    # Verifica se é o melhor modelo até agora
    if acc > melhor_acc:
        melhor_acc = acc
        melhor_W1, melhor_b1 = W1.copy(), b1.copy()
        melhor_W2, melhor_b2 = W2.copy(), b2.copy()

# --- Resultado final ---
print("\nAcurácias por fold:", np.round(accuracies, 4))
print("Acurácia média final:", round(np.mean(accuracies)*100, 2), "%")
print("Melhor acurácia registrada:", round(melhor_acc * 100, 2), "%")

# --- Salvando os melhores pesos ---
np.savez("pesos_iris.npz", W1=melhor_W1, b1=melhor_b1, W2=melhor_W2, b2=melhor_b2)
print("\nPesos salvos em 'pesos_iris.npz'")
joblib.dump(scaler, "scaler_iris.pkl")
