import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from activations import sigmoid, sigmoid_deriv

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=3000, seed=42, k_folds=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.k_folds = k_folds

        self.best_accuracy = 0
        self.best_weights = None
        self.accuracies = []

        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False)

    def initialize_weights(self):
        np.random.seed(self.seed)
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.rand(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        error = y - self.a2
        d_a2 = error * sigmoid_deriv(self.a2)
        d_W2 = np.dot(self.a1.T, d_a2)
        d_b2 = np.sum(d_a2, axis=0, keepdims=True)

        d_a1 = np.dot(d_a2, self.W2.T) * sigmoid_deriv(self.a1)
        d_W1 = np.dot(X.T, d_a1)
        d_b1 = np.sum(d_a1, axis=0, keepdims=True)

        self.W2 += self.learning_rate * d_W2
        self.b2 += self.learning_rate * d_b2
        self.W1 += self.learning_rate * d_W1
        self.b1 += self.learning_rate * d_b1

    def train_epoch(self, X, y):
        self.forward(X)
        self.backward(X, y)

    def train(self, X, y):
        for _ in range(self.epochs):
            self.train_epoch(X, y)

    def predict(self, X):
        a2 = self.forward(X)
        return np.argmax(a2, axis=1)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        return accuracy_score(y_true, y_pred)

    def cross_validate(self, X, y):
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.initialize_weights()
            self.train(X_train, y_train)
            acc = self.evaluate_model(X_test, y_test)
            self.accuracies.append(acc)

            print(f"Fold {fold+1} - Acurácia: {acc:.4f}")

            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_weights = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

        self.save_model()

    def hold_out_split(self, X, y, train_size=100):
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        self.initialize_weights()
        self.train(X_train, y_train)
        acc = self.evaluate_model(X_test, y_test)

        print(f"\nHold-out - Acurácia: {acc:.4f}")
        self.accuracies = [acc]
        self.best_accuracy = acc
        self.best_weights = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

        self.save_model()

    def print_results(self):
        print("\nAcurácias por fold:", np.round(self.accuracies, 4))
        print("Acurácia média final:", round(np.mean(self.accuracies) * 100, 2), "%")
        print("Melhor acurácia registrada:", round(self.best_accuracy * 100, 2), "%")

    def save_model(self):
        pass
