import numpy as np
def gradient_descent(X, y, lr=0.01, epochs=1000) -> tuple[int, int]:
    """Calculates weights and biases of w*x + b
    Args:
        X(pd.DataFrame): Explanatory Variable
        y(pd.DataFrame): Dependent Variable
    Returns:
        weight, bias
    """
    W = 0.0
    b = 0
    assert len(X) == len(y)
    n = len(X)
    forward_func = lambda : W * X + b
    # y_pred = W * X + b
    loss_func = lambda y_pred: np.mean((y_pred - y) * (y_pred - y))
    dL_dw = lambda diff: (2 / n) * np.sum(diff * X)
    dL_db = lambda diff: (2 / n) * np.sum(diff)

    for epoch in range(epochs):
        y_pred = forward_func()
        loss = loss_func(y_pred)
        diff = y_pred - y
        print("MSE: " + str(loss))
        W -= lr * dL_dw(diff)
        b -= lr * dL_db(diff)
    return (W, b)
gradient_descent(X, y)
