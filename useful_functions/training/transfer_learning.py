"""
Transfer Learning for Flower Classification using MobileNetV2 (TensorFlow / Keras)

This script demonstrates:
1. Feature Extraction Transfer Learning using a pre-trained ImageNet backbone (MobileNetV2).
2. Proper layer freezing (base_model.trainable = False) to preserve pre-trained weights.
3. Preprocessing with tf.keras.applications.mobilenet_v2.preprocess_input (scaling to [-1, 1]).
4. Modern Head Architecture: GlobalAveragePooling2D + Dropout + Linear Logits Dense.
5. In-model data augmentation for robust regularization.
6. Complete training loop (model.fit) with EarlyStopping and ReduceLROnPlateau.
7. Proper evaluation using sklearn.metrics.classification_report with named classes.
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 5

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


# ---------------------------------------------------------
# 2. Data Loading & Preprocessing
# ---------------------------------------------------------
def load_and_preprocess_data(
    data_dir: Path, 
    class_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, resize, convert color space, and split flower dataset."""
    flower_images_dict = {
        cls: list(data_dir.glob(f"{cls}/*.jpg")) 
        for cls in class_names
    }
    label_dict = {cls: idx for idx, cls in enumerate(class_names)}

    X = []
    y = []

    print(f"[*] Ingesting images for classes: {class_names}")
    for label_name, paths in flower_images_dict.items():
        label_id = label_dict[label_name]
        for img_path in paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            # Convert OpenCV default BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Standardize resolution for MobileNetV2
            img = cv2.resize(img, IMAGE_SIZE)
            X.append(img)
            y.append(label_id)

    # Convert to float32 NumPy arrays (raw pixel values [0, 255])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"[*] Loaded {len(X)} total images with shape {X.shape[1:]}")

    # Stratified 80/20 train-test split prevents class imbalance across sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 3. Model Architecture (Transfer Learning)
# ---------------------------------------------------------
def make_transfer_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE, 
    num_classes: int = NUM_CLASSES
) -> tf.keras.Model:
    """Constructs a transfer learning model using a frozen MobileNetV2 backbone."""
    
    # 1. Instantiate Pretrained Backbone (ImageNet weights, no top dense layers)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # 2. Freeze the backbone: crucial to protect pre-trained visual representations
    base_model.trainable = False

    # 3. Assemble Sequential Model
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Online Data Augmentation (active only during training)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),

        # MobileNetV2 specific preprocessing: scales [0, 255] pixels to [-1, 1]
        layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),

        # Frozen Pre-trained Feature Extractor
        base_model,

        # Global Spatial Pooling: condenses 7x7x1280 feature maps into a 1280-dim vector
        layers.GlobalAveragePooling2D(),

        # Regularized Classification Head
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        # Output raw logits for numerical stability with SparseCategoricalCrossentropy
        layers.Dense(num_classes)
    ])

    return model


# ---------------------------------------------------------
# 4. Visualization Helpers
# ---------------------------------------------------------
def visualize_sample_grid(images: np.ndarray, labels: np.ndarray, class_names: list[str]):
    """Displays a clean 1x5 grid of sample images with their corresponding class labels."""
    plt.figure(figsize=(12, 3))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].astype(np.uint8))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_training_curves(history: tf.keras.callbacks.History):
    """Plot training vs. validation accuracy and loss."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", marker="o")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="s")
    plt.title("Transfer Learning Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", marker="o")
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="s")
    plt.title("Transfer Learning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 5. Main Execution Pipeline
# ---------------------------------------------------------
def main():
    # Download dataset
    raw_path = tf.keras.utils.get_file(
        "flower_photos", 
        origin=DATASET_URL, 
        cache_dir="./data", 
        untar=True
    )
    data_dir = Path(raw_path)

    # Define verified class names matching subfolder names
    class_names = ["roses", "daisies", "sunflowers", "dandelions", "tulips"]

    # Preprocess dataset
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_dir, class_names)

    # Visualize sample images
    visualize_sample_grid(X_train, y_train, class_names)

    # Build and inspect model
    model = make_transfer_model(input_shape=INPUT_SHAPE, num_classes=len(class_names))
    model.summary()

    # Compile with Adam and Logits-based Cross-Entropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    # Training Callbacks
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    # Train model (Feature Extraction)
    print("\n[*] Training transfer learning model (Backbone Frozen)...")
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=training_callbacks
    )

    # Evaluate on Test Set
    print("\n[*] Evaluating on Test Set...")
    y_pred_logits = model.predict(X_test)
    y_pred = np.argmax(y_pred_logits, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Plot metrics
    plot_training_curves(history)


if __name__ == "__main__":
    main()
