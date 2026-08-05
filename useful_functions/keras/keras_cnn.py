import tensorflow as tf
from tensorflow.keras import layers, models

def build_mnist_cnn_keras() -> tf.keras.Model:
    return models.Sequential([
        # Input: [Batch, Height=28, Width=28, Channel=1]
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(units=10, activation='softmax')
    ], name="mnist_cnn_keras")

# Training
cnn_keras = build_mnist_cnn_keras()

# Keras models use compile() instead of a separate Trainer class
cnn_keras.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Reshape data for Keras (N, H, W, C)
X_train_keras = X_train_processed.reshape(-1, 28, 28, 1)

# Train using Keras fit method
cnn_keras.fit(X_train_keras, y_train, epochs=10, batch_size=64, validation_split=0.2)
