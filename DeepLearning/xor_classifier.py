import numpy as np
np.random.seed(34)

class XorClassifier:
    def __init__(self):
        # (Input Dimensions = 2, Output Dimensions = 1)
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(2, 1)).astype(np.float32)
        self.b = np.zeros(shape=(1,)).astype('float32')
        self.epochs = 1000
        self.learning_rate = 1.0
        # x: (4, 2) W: (2, 1), b = (1, ) => (4, 1)
        self.forward = lambda x: self.sigmoid(np.matmul(x, self.W) + self.b)

    def sigmoid(self, x: np.ndarray):
        """Calculates 1 / (1 + exp(-x)) = exp(x) / (exp(x) + 1)
        implmemented couter overflow measures
        """
        numerator = np.exp(np.minimum(0, x))
        denominator = 1 + np.exp(-np.abs(x))
        return numerator / denominator
    
    def np_log(x: np.ndarray):
        return np.log(np.clip(x, 1e-6, 1e6))

    def load_data(self) -> np.ndarray:
        """Returns Xor Data
            x_train: (batch = 4, embed=2)
            y_train: (batch = 4,)
            x_valid: (batch = 4, embed=2)
            y_valid: (batch = 4,)
        """
        ## (batch = 4, embed=2)
        x_train = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        ## (batch = 4,)
        y_train = np.array([[1], [1], [0], [1]])
        x_valid, y_valid = x_train, y_train
        return x_train, y_train, x_valid, y_valid

    
    def train(self):
        """Updates Weights and Biases
        """
        x_train, y_train, x_valid, y_valid = self.load_data()
        batch_size = x_train.shape[0]

        for epoch in range(self.epochs):
            # === forward prediction === 
            y_pred = self.forward(x_train)
            # y: (batch_size, 1) = (4, 1)
            # === evaluation ===
            loss = - np.mean(y_train * self.np_log(y_pred) + (1 - y_train) * self.np_log((1 - y_pred)))
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')
            delta = y_pred - y_train
            # === backward optimization ===
            # W: (input dimensions, output_dimensions)
            delta_W = np.matmul(x_train.T, delta) / batch_size
            delta_b = np.matmul(np.ones(shape=(batch_size, )), delta) / batch_size
            self.W -= self.learning_rate * delta_W
            self.b -= self.learning_rate * delta_b
        print("Validation of Xor (Losses):\n")
        y_bias = self.validate(y_valid, self.forward(x_valid))
        for batch in range(batch_size):
            print(f"data: {x_valid[batch]}")
            print(f"pred: {self.forward(x_valid[batch])}")
            print(f"loss: {y_bias}")
    
    def validate(self, y_ture, y_pred) -> np.array:
        """Compare the predictions and Xor answers
        """
        loss = -np.mean(y_ture * self.np_log(y_pred) + (1 - y_ture) * self.np_log((1 - y_pred)))
        return loss
