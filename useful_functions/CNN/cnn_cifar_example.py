import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks

# CIFAR-10 Class Names for interpretable classification reports
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def preprocess(X_train, y_train, X_test, y_test):
    """
    Normalize pixel values from [0, 255] to [0.0, 1.0] and flatten target labels.
    Input images: 32x32x3 RGB
    """
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Flatten (N, 1) target labels to 1D arrays (N,)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    return X_train, y_train, X_test, y_test

def make_model() -> models.Sequential:
    """
    Constructs an improved VGG-style CNN for CIFAR-10 with:
    1. Data Augmentation (RandomFlip, RandomTranslation) to prevent overfitting.
    2. Paired 3x3 Conv blocks + BatchNormalization for rich feature extraction.
    3. Progressive Dropout (0.2 -> 0.3 -> 0.4 -> 0.5) for robust regularization.
    """
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=(32, 32, 3)),

        # --- Data Augmentation Block (active only during training) ---
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomRotation(0.05),

        # --- Block 1: 32 Filters ---
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        # --- Block 2: 64 Filters ---
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),

        # --- Block 3: 128 Filters ---
        layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.4),

        # --- Dense Classification Head ---
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    return model

def plot_results(history):
    """Plots training and validation accuracy/loss side-by-side."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training & Validation Accuracy')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training & Validation Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    # 1. Load CIFAR-10 Data
    print("--- Loading CIFAR-10 Dataset ---")
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    
    # 2. Preprocess
    X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # 3. Build & Compile Model
    model = make_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 4. Training Callbacks
    callback_list = [
        # Reduce learning rate when validation loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
        # Stop early if validation performance stops improving
        callbacks.EarlyStopping(
            monitor='val_loss', patience=6, restore_best_weights=True, verbose=1
        )
    ]

    # 5. Train Model
    print("\n--- Training Model ---")
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callback_list
    )

    # 6. Evaluate Model on Test Set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc*100:.2f}%")

    # 7. Predictions & Discrete Thresholding for Classification Report
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # 8. Plot Accuracy & Loss Curves
    plot_results(history)

if __name__ == '__main__':
    main()
