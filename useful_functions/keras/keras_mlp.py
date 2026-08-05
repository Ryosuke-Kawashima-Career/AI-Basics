# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def get_model(num_layers) -> keras.Sequential:
    """Returns a multi-layer perceptron
    Input Data:
        Shape: (Batch, 28, 28, 1)
    Output:
        Shape: (Batch, 10)
    Args:
        num_layers: Number of hidden layers
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
    ])
    for _ in range(num_layers):
        model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model

def train_model(model, x, y, batch_size=32, epochs=5):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    with tf.device("/device:CPU:0"):
        history = model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)
    return history

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    print("X_train.shape =", X_train.shape)

    # Build and train models
    num_layers = 2
    model = get_model(num_layers)
    model.summary()
    history = train_model(model, X_train, y_train)

    # Plot accuracy vs epoch
    plt.plot(history.history['accuracy'])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.show()

if __name__ == '__main__':
    main()
