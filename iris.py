import numpy as np
import joblib
from sklearn.datasets import load_iris

from neural_network import NeuralNetwork


class IrisClassifier(NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(input_size=4, hidden_size=2, output_size=3, **kwargs)

    def load_data(self):
        iris = load_iris()
        X, y = iris.data, iris.target

        # Embaralha os dados
        np.random.seed(self.seed)
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

        # Escala e codifica
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1))
        return X_scaled, y_encoded

    def save_model(self):
        if self.best_weights:
            W1, b1, W2, b2 = self.best_weights
            np.savez("pesos_iris.npz", W1=W1, b1=b1, W2=W2, b2=b2)
            joblib.dump(self.scaler, "scaler_iris.pkl")
            print("\nMelhores pesos salvos em 'pesos_iris.npz'")
            print("Scaler salvo em 'scaler_iris.pkl'")

    def run_hold_out(self):
        X, y = self.load_data()
        self.hold_out_split(X, y)
        self.print_results()


if __name__ == "__main__":
    iris_model = IrisClassifier(epochs=3000, learning_rate=0.1)
    iris_model.run_hold_out()
