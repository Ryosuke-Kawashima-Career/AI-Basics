# Import Libraries
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding


def make_model(vocab_size: int, embed_size: int) -> Sequential:
    """Builds and returns a basic binary text classification neural network."""
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embed_size,
            name="embedding",
        ),
        Flatten(),
        Dense(1, activation="sigmoid"),
    ])
    return model


def main():
    # Dataset
    train_reviews = [
        'nice food',
        'amazing restaurant',
        'too good',
        'just loved it!',
        'will go again',
        'horrible food',
        'never go there',
        'poor service',
        'poor quality',
        'needs improvement'
    ]
    train_sentiments = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # Preprocessing Hyperparameters
    vocab_size = 30
    max_length = 4
    embed_size = 5

    # 1. Encoding (Hashing words into integers [1, vocab_size - 1])
    encoded_train = [one_hot(d, vocab_size) for d in train_reviews]

    # 2. Padding (Equalizing sequence length to max_length with zeros)
    X_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
    y_train = train_sentiments

    # 3. Model Creation & Training
    model = make_model(vocab_size, embed_size)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, verbose=0)

    # 4. Evaluation on Training Data
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    print(f"--- Training Set Evaluation ---")
    print(f"Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%\n")

    # 5. Inference / Prediction on New Test Reviews
    test_reviews = ['great restaurant', 'horrible food', 'poor quality']
    encoded_test = [one_hot(d, vocab_size) for d in test_reviews]
    X_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')

    probabilities = model.predict(X_test, verbose=0)

    print("--- Inference on New Reviews ---")
    for review, prob in zip(test_reviews, probabilities):
        label = "Positive" if prob[0] >= 0.5 else "Negative"
        print(f"Review: '{review:<20}' -> Prediction: {label} ({prob[0]:.4f})")

    # 6. Inspect Learned Word Embeddings
    embedding_weights = model.get_layer('embedding').get_weights()[0]
    print(f"\nEmbedding Table Shape: {embedding_weights.shape} (vocab_size x embed_size)")


if __name__ == '__main__':
    main()

