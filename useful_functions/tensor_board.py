import datetime
import os
 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
# %load_ext tensorboard
# %tensorboard --logdir logs
class LogisticMNISTClassifierWithObservation:
    """Logistic Regression Classifier for MNIST
 
    Shapes:
        x_train: (batch=60000, embed=28*28)
        y_train: (batch=60000, classes=10)
        x.T @ delta = (embed=28*28, batch) @ (batch, classes=10) = (embed=28*28, classes=10)
        W: (embed=28*28, classes=10)
        b: (classes=10,)
        delta: (batch, classes)
    """
    # the number of pixels
    INPUT_DIM = 28 * 28
    # the number of the categories
    NUM_CLASSES = 10
    def __init__(self, lr: float = 1.0, epochs: int = 100, log_dir: str = None):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(self.INPUT_DIM, self.NUM_CLASSES)).astype(np.float32)
        self.b = np.zeros(shape=(self.NUM_CLASSES,)).astype(np.float32)
        self.lr = lr
        self.epochs = epochs
        log_dir = log_dir or os.path.join(
            "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.train_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
        self.valid_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'valid'))
        print(f"TensorBoard logs -> {log_dir}  (run: tensorboard --logdir {os.path.dirname(log_dir)})")

    def forward(self, x: np.ndarray):
        """Computes prediction by Softmax(x * w + b)
        """
        return self.softmax(np.matmul(x, self.W) + self.b)

    def softmax(self, x: np.ndarray, axis=1):
        """Returns the value of softmax with the implementation of overflow measures
        Args:
            x(np.ndarray): (batch_size, flattened_pixels)
        Returns:
            np.ndarray: (batch_size, 28*28)
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        numerator = np.exp(x - x_max)
        denomenator = np.sum(numerator, axis=axis, keepdims=True)
        return numerator / denomenator

    def np_log(self, x: np.ndarray):
        return np.log(np.clip(x, 1e-6, 1e6))


    def load_MNIST_data(self) -> np.ndarray:
        """Returns MNIST Data via keras
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], -1) / 255.
        x_test = x_test.reshape(x_test.shape[0], -1) / 255.
        # Converts the data into one hot vector
        y_train = np.eye(self.NUM_CLASSES)[y_train.astype('int32').flatten()]
        y_test = np.eye(self.NUM_CLASSES)[y_test.astype('int32').flatten()]
        return x_train, y_train, x_test, y_test

    def train(self):
        """Updates Weights and Biases for each epoch
        """
        x_train, y_train, _, _ = self.load_MNIST_data()
        batch_size = x_train.shape[0]
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        for epoch in range(self.epochs):
            y_pred = self.forward(x_train)
            loss = self.loss_function(y_pred, y_train)
            acc = self.accuracy(y_pred, y_train)
            # Checks the validity of training
            with self.train_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch)
                tf.summary.scalar('accuracy', acc, step=epoch)
            if epoch % 10 == 0:
                print(f"Training: Epoch: {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}")
            # This alters the model parameters
            delta = y_pred - y_train
            self.optimization(x_train, delta)
        
            # Validation
            y_val_pred = self.forward(x_valid)
            val_loss = self.loss_function(y_val_pred, y_valid)
            val_acc = self.accuracy(y_val_pred, y_valid)
 
            with self.valid_writer.as_default():
                tf.summary.scalar("loss", val_loss, step=epoch)
                tf.summary.scalar("accuracy", val_acc, step=epoch)
 
            if epoch % 10 == 0:
                print(f"Validation: Epoch: {epoch}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        self.train_writer.flush()
        self.valid_writer.flush()

    def loss_function(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculates cross entropy
        Args:
            y_pred(ndarray): prediction of the model
            y_true(ndarray): ground truth
        Returns:
            float: loss
        """
        # y: (batchsize, number of categories)
        y_pred = np.clip(y_pred, 1e-6, 1.0 - 1e-6)
        loss = - y_true * self.np_log(y_pred) - (1 - y_true) * self.np_log((1 - y_pred))
        loss = np.sum(loss, axis=1)
        return np.mean(loss, axis=0)
    
    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculates accuracy
        Args:
            y_pred(ndarray): prediction of the model
            y_true(ndarray): ground truth
        Returns:
            float: accuracy
        """
        # y: (batchsize, actegories)
        y_pred_onehot = np.argmax(y_pred, axis=1)
        y_true_onehot = np.argmax(y_true, axis=1)
        return np.mean(y_pred_onehot == y_true_onehot)

    def optimization(self, x, delta):
        """Updates Weights and Biases
        """
        batch_size = x.shape[0]
        dW = np.matmul(x.T, delta) / batch_size
        db = np.matmul(np.ones(shape=(x.shape[0], )), delta) / batch_size
        self.W -= self.lr * dW
        self.b -= self.lr * db

    def test(self) -> np.ndarray:
        """Backward propagation
        Changes the weights and biases of self.W and self.b
        """
        _, _, x_test, y_test = self.load_MNIST_data()
        y_pred = self.forward(x_test)
        loss = self.loss(y_pred, y_test)
        print(f"Test: Loss: {loss}")
        return y_pred