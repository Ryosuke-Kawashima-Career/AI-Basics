import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def keras_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """Keras reference implementation of Logistic Regression."""
    model = keras.Sequential([Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])
    model.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_test, y_test))
    return model

class CustomLogisticRegression:
    """Custom Logistic Regression implementation using Gradient Descent from scratch."""

    def __init__(self, feature_count: int, learning_rate: float = 0.5, epochs: int = 500, batch_size: int = 32):
        self.lr = learning_rate
        self.epochs = epochs
        # 2D weight matrix shape: (feature_count, 1)
        self.W = np.zeros((feature_count, 1))
        self.b = 0.0
        self.batch_size = batch_size

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable vectorized Sigmoid activation."""
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: computes linear combination and applies Sigmoid activation.
        
        Shapes:
            x: (batch_size, feature_count)
            self.W: (feature_count, 1)
            self.b: scalar float
            Output h: (batch_size, 1)
        """
        z = np.dot(x, self.W) + self.b
        return self.sigmoid(z)

    def loss_function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes Binary Cross-Entropy Loss with log safety clipping."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        entropy = y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
        return -float(np.mean(entropy))

    def backward(self, X_batch: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Backward pass: computes gradients dW and db and updates parameters.
        
        Shapes:
            X_batch: (m, feature_count)
            y_pred - y_true: (m, 1)
            dW: (feature_count, 1) = X_batch.T (feature_count, m) dot (y_pred - y_true) (m, 1) / m
            db: scalar = sum(y_pred - y_true) / m
        """
        m = X_batch.shape[0]
        dW = (1.0 / m) * np.dot(X_batch.T, (y_pred - y_true))
        db = (1.0 / m) * np.sum(y_pred - y_true)

        # Gradient Descent parameter update
        self.W -= self.lr * dW
        self.b -= self.lr * db

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the model using Mini-batch Gradient Descent."""
        # Ensure 2D column vector shape for targets
        y = y.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        n_samples = X_train.shape[0]
        batches = (n_samples + self.batch_size - 1) // self.batch_size

        print("--- Training Custom Logistic Regression ---")
        for epoch in range(self.epochs):
            for batch in range(batches):
                batch_idx = batch * self.batch_size
                X_batch = X_train[batch_idx : batch_idx + self.batch_size]
                y_batch = y_train[batch_idx : batch_idx + self.batch_size]

                # Forward pass
                y_pred_batch = self.forward(X_batch)
                # Backward pass & parameter update
                self.backward(X_batch, y_batch, y_pred_batch)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                y_pred_train = self.forward(X_train)
                loss_train = self.loss_function(y_train, y_pred_train)
                y_pred_test = self.forward(X_test)
                loss_test = self.loss_function(y_test, y_pred_test)
                print(f"Epoch {epoch+1:3d}/{self.epochs} - Train Loss: {loss_train:.4f}, Test Loss: {loss_test:.4f}")

def main():
    # Load dataset dynamically based on current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'insurance_data.csv')
    df = pd.read_csv(csv_path)

    # Feature engineering / scaling: scale 'age' by 100 for fast gradient descent convergence
    X = np.array(df[['age', 'affordibility']].values, dtype=np.float64)
    X[:, 0] = X[:, 0] / 100.0  # Scale age feature to [0, 1] range
    y = np.array(df['bought_insurance'].values, dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Custom Logistic Regression from scratch
    model_manual = CustomLogisticRegression(feature_count=X.shape[1], learning_rate=0.5, epochs=500, batch_size=32)
    model_manual.fit(X, y)

    print("\n--- Final Parameters ---")
    print(f"Custom Model Weights (W):\n{model_manual.W.ravel()}")
    print(f"Custom Model Bias (b):    {model_manual.b:.4f}")

    # 2. Keras Model for comparison
    model_keras = keras_logistic_regression(X_train, y_train, X_test, y_test)
    keras_weights, keras_bias = model_keras.get_weights()
    print(f"\nKeras Model Weights (W):   {keras_weights.ravel()}")
    print(f"Keras Model Bias (b):      {keras_bias[0]:.4f}")

if __name__ == '__main__':
    main()

    
