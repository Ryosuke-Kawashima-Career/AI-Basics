# Phase 3: Tokenizer Guide (トークナイザーガイド)

To feed English text into a neural network, we must first convert it into numbers. This is done by a Tokenizer (トークナイザー), which consists of two main components:
1. **Encoder (エンコーダー)**: Converts text (strings) into a sequence of integer IDs.
2. **Decoder (デコーダー)**: Converts integer IDs back into human-readable text.

Here, we explore the two primary methods we can use for our Minimum Viable Transformer: **Byte-level mapping** and **Word-level dictionary mapping**.

---

## 🔢 Method 1: Byte-level Mapping (バイトレベル・マッピング)

Every character in computer memory is stored as a byte (ASCII or UTF-8). By mapping characters directly to their byte values, we get a simple, robust character-level (文字レベル) tokenizer with a fixed vocabulary size of `256`.

### How it works:
- **Input Text**: `"hello"`
- **Encoder**: Converts each character to its byte value:
  - `'h'` $\to$ `104`
  - `'e'` $\to$ `101`
  - `'l'` $\to$ `108`
  - `'l'` $\to$ `108`
  - `'o'` $\to$ `111`
  - **Output shape**: `[S]` where sequence length $S = 5$. Value: `[104, 101, 108, 108, 111]`
- **Decoder**: Collects these byte values and builds a UTF-8 string.

### Rust Code Example (Byte-level):
```rust
pub struct ByteTokenizer;

impl ByteTokenizer {
    /// Encodes a string into a sequence of byte-based Token IDs.
    /// Shape transition: String -> [S] (1D tensor of length S)
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.as_bytes().iter().map(|&b| b as usize).collect()
    }

    /// Decodes a slice of Token IDs back into a String.
    /// Shape transition: [S] -> String
    pub fn decode(&self, ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids.iter().map(|&id| id as u8).collect();
        String::from_utf8(bytes).unwrap_or_else(|_| String::from(""))
    }
}
```

---

## 📖 Method 2: Word-level Dictionary Mapping (単語レベルの辞書マッピング)

Instead of individual characters, we can split text into full English words. We build a Vocabulary (語彙) dictionary from a corpus and map each unique word to a unique ID. We also reserve Special Tokens (特殊トークン):
- `<PAD>` (ID `0`): Used to pad sequences to a uniform length.
- `<UNK>` (ID `1`): Represents out-of-vocabulary (語彙外) unknown words.
- `<EOS>` (ID `2`): Indicates the end of a sequence.

### How it works:
If our vocabulary is `{"<PAD>": 0, "<UNK>": 1, "<EOS>": 2, "hello": 3, "world": 4}`:
- **Input Text**: `"hello rust"`
- **Encoder**: 
  - `"hello"` is in vocabulary $\to$ ID `3`
  - `"rust"` is NOT in vocabulary $\to$ maps to `<UNK>` $\to$ ID `1`
  - **Output shape**: `[S]` where sequence length $S = 2$. Value: `[3, 1]`
- **Decoder**: Maps `[3, 1]` back to `"hello <UNK>"`.

### Rust Code Example (Word-level):
```rust
use std::collections::HashMap;

pub struct WordTokenizer {
    vocab: HashMap<String, usize>,
    id_to_word: Vec<String>,
}

impl WordTokenizer {
    /// Builds a vocabulary map from a training text corpus.
    pub fn new(corpus: &str) -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_word = Vec::new();

        // Register special tokens
        let special_tokens = vec!["<PAD>", "<UNK>", "<EOS>"];
        for token in special_tokens {
            vocab.insert(token.to_string(), id_to_word.len());
            id_to_word.push(token.to_string());
        }

        // Tokenize and collect unique words
        for word in corpus.split_whitespace() {
            let clean = word.to_lowercase().replace(&['.', ',', '!', '?'][..], "");
            if !vocab.contains_key(&clean) {
                vocab.insert(clean.clone(), id_to_word.len());
                id_to_word.push(clean);
            }
        }

        Self { vocab, id_to_word }
    }

    /// Encodes text into a sequence of Word IDs.
    /// Shape: String -> [S] (1D tensor of length S)
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                let clean = word.to_lowercase().replace(&['.', ',', '!', '?'][..], "");
                *self.vocab.get(&clean).unwrap_or(&1) // Default to <UNK> (ID 1)
            })
            .collect()
    }

    /// Decodes IDs back to text space-separated.
    /// Shape: [S] -> String
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| {
                if id < self.id_to_word.len() {
                    self.id_to_word[id].as_str()
                } else {
                    "<UNK>"
                }
            })
            .collect::<Vec<&str>>()
            .join(" ")
    }
}
```

---

## 📊 Tensor Shapes & Embedding Lookup

When tokens are encoded, they form a 1D index array:
\[
\text{token\_ids} \in \mathbb{R}^S
\]
Here is how this indices array of shape `[S]` interacts with the Embedding Weight Matrix (埋め込み重み行列) of shape `[V, D]`, where $V$ is `vocab_size` and $D$ is `d_model`:

```
Input IDs: [3, 4] (Shape: [S=2])
             |
             v
Embedding Lookup (Weight Matrix of Shape [V, D])
   ID 3 -> Look up row 3 -> [w3_1, w3_2, ..., w3_D]
   ID 4 -> Look up row 4 -> [w4_1, w4_2, ..., w4_D]
             |
             v
Output Embeddings (Shape: [S=2, D])
```

The output of the embedding layer is a 2D matrix of shape `[S, D]`, which is ready for Multi-Head Attention!

---

## 🎯 Understanding Check (理解度チェック)

Let's test your understanding of tokenizers and embeddings:

1. **Question 1**: If you build a `WordTokenizer` using the corpus `"I love Rust."`, what will be the encoded ID sequence for the input sentence `"I love Python."`? (Assume `<PAD>` is 0, `<UNK>` is 1, and `<EOS>` is 2).
2. **Question 2**: If the sequence length $S = 6$, and `d_model = 64`, what is the shape of the output matrix after retrieving token embeddings? How many elements does this embedding matrix contain?

*(Write down your predictions and try to visualize the lookup process!)*
