import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def batch_gradient_descent(X: np.ndarray, y_true: np.ndarray, epochs: int = 500, learning_rate: float = 0.01):
    """Batch Gradient Descent (BGD): Updates weights using the entire dataset in each epoch."""
    n_samples, n_features = X.shape
    w = np.ones(shape=(n_features, 1))
    b = 0.0
    cost_history = []

    for epoch in range(epochs):
        y_pred = np.dot(X, w) + b
        cost = np.mean(np.square(y_true - y_pred))

        # Compute gradients over all n_samples
        w_grad = -(2 / n_samples) * np.dot(X.T, (y_true - y_pred))
        b_grad = -(2 / n_samples) * np.sum(y_true - y_pred)

        # Update weights and bias
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad

        cost_history.append(cost)

    return w, b, cost, cost_history

def stochastic_gradient_descent(X: np.ndarray, y_true: np.ndarray, epochs: int = 500, learning_rate: float = 0.01):
    """Stochastic Gradient Descent (SGD): Updates weights using 1 random sample at a time."""
    n_samples, n_features = X.shape
    w = np.ones(shape=(n_features, 1))
    b = 0.0
    cost_history = []

    for epoch in range(epochs):
        # Pick 1 random sample per iteration
        random_index = np.random.randint(0, n_samples)
        sample_x = X[random_index:random_index+1]  # shape: (1, n_features)
        sample_y = y_true[random_index:random_index+1]  # shape: (1, 1)

        y_pred = np.dot(sample_x, w) + b
        cost = np.mean(np.square(sample_y - y_pred))

        # Compute gradient for single sample
        w_grad = -(2) * np.dot(sample_x.T, (sample_y - y_pred))
        b_grad = -(2) * np.sum(sample_y - y_pred)

        w -= learning_rate * w_grad
        b -= learning_rate * b_grad

        cost_history.append(cost)

    return w, b, cost, cost_history

def mini_batch_gradient_descent(X: np.ndarray, y_true: np.ndarray, epochs: int = 500, batch_size: int = 5, learning_rate: float = 0.01):
    """Mini-Batch Gradient Descent (MBGD): Updates weights using mini-batches of size `batch_size`."""
    n_samples, n_features = X.shape
    w = np.ones(shape=(n_features, 1))
    b = 0.0
    cost_history = []

    n_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        # Shuffle dataset at start of each epoch
        random_indices = np.random.permutation(n_samples)
        X_shuffled = X[random_indices]
        y_shuffled = y_true[random_indices]

        for b_idx in range(n_batches):
            start_i = b_idx * batch_size
            end_i = start_i + batch_size

            X_batch = X_shuffled[start_i:end_i]
            y_batch = y_shuffled[start_i:end_i]
            current_m = X_batch.shape[0]

            y_pred = np.dot(X_batch, w) + b
            cost = np.mean(np.square(y_batch - y_pred))

            # Compute gradients over current mini-batch
            w_grad = -(2 / current_m) * np.dot(X_batch.T, (y_batch - y_pred))
            b_grad = -(2 / current_m) * np.sum(y_batch - y_pred)

            w -= learning_rate * w_grad
            b -= learning_rate * b_grad

        cost_history.append(cost)

    return w, b, cost, cost_history

if __name__ == '__main__':
    # Load dataset dynamically based on current script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'homeprices_banglore.csv')
    df = pd.read_csv(csv_path)

    # Preprocessing & Scaling
    sx = MinMaxScaler()
    sy = MinMaxScaler()

    X_scaled = sx.fit_transform(df[['area', 'bedrooms']].values)
    y_scaled = sy.fit_transform(df['price'].values.reshape(-1, 1))

    print("--- Gradient Descent Comparison (Home Prices Bangalore) ---")
    w_bgd, b_bgd, cost_bgd, _ = batch_gradient_descent(X_scaled, y_scaled, epochs=500, learning_rate=0.01)
    print(f"[BGD]  Final Weights: {w_bgd.ravel()}, Bias: {b_bgd:.4f}, Final MSE Loss: {cost_bgd:.6f}")

    w_sgd, b_sgd, cost_sgd, _ = stochastic_gradient_descent(X_scaled, y_scaled, epochs=500, learning_rate=0.01)
    print(f"[SGD]  Final Weights: {w_sgd.ravel()}, Bias: {b_sgd:.4f}, Final MSE Loss: {cost_sgd:.6f}")

    w_mbgd, b_mbgd, cost_mbgd, _ = mini_batch_gradient_descent(X_scaled, y_scaled, epochs=500, batch_size=5, learning_rate=0.01)
    print(f"[MBGD] Final Weights: {w_mbgd.ravel()}, Bias: {b_mbgd:.4f}, Final MSE Loss: {cost_mbgd:.6f}")
