# Phase 5: Embeddings & Positional Encoding Guide

In this phase, we implement the gateway of our Transformer. Since neural networks cannot process raw token IDs directly, we must project them into a vector space. Additionally, because Transformers process all tokens simultaneously (in parallel), they have no built-in notion of word order. We resolve this by adding positional information.

This involves two components:
1. **Token Embedding (トークン埋め込み)**: Maps discrete token IDs to continuous dense vectors.
2. **Positional Encoding (位置エンコーディング)**: Adds wave-like patterns representing the absolute positions of tokens.

---

## 🎨 1. Token Embedding

The Token Embedding (トークン埋め込み) is a lookup table. For a vocabulary size $V$ (`vocab_size`) and model dimension $D$ (`d_model`), we maintain a weight matrix (重み行列) of shape `[V, D]`.

### How the lookup is performed:
Given a sequence of token IDs of shape `[S]` (where $S$ is the Sequence Length):
*   For each token ID $t \in [0, V-1]$, we retrieve the $t$-th row of the weight matrix.
*   The resulting output is a 2D tensor of shape `[S, D]`.

### Rust Code (Embedding):
```rust
use ndarray::{Array2, ArrayView1};
use rand::distributions::Uniform;
use rand::prelude::*;

pub struct Embedding {
    /// Weight matrix containing embedding vectors.
    /// Shape: [vocab_size, d_model]
    pub weight: Array2<f32>,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        let mut rng = thread_rng();
        // Initialize weights with small random values
        let dist = Uniform::new(-0.1, 0.1);
        let weight = Array2::from_shape_fn((vocab_size, d_model), |_| dist.sample(&mut rng));

        Self { weight }
    }

    /// Performs embedding lookup.
    /// Input: token_ids of shape [S]
    /// Output: dense embeddings of shape [S, d_model]
    pub fn forward(&self, token_ids: &[usize]) -> Array2<f32> {
        let seq_len = token_ids.len();
        let d_model = self.weight.ncols();
        let mut output = Array2::zeros((seq_len, d_model));

        for (i, &token_id) in token_ids.iter().enumerate() {
            // Retrieve row corresponding to token_id
            let embed_vector = self.weight.row(token_id);
            // Copy it to the output matrix row
            output.row_mut(i).assign(&embed_vector);
        }

        output
    }
}
```

---

## 🌊 2. Positional Encoding

Because self-attention contains no convolutional or recurrent structures, the model sees a sequence as an unordered set of words (bag of words). To feed sequence order into the model, we add a static Positional Encoding (位置エンコーディング) matrix of shape `[max_seq_len, d_model]` directly to our token embeddings.

We use sine (正弦) and cosine (余弦) waves of different frequencies:
\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/D}}\right)
\]
\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/D}}\right)
\]

Where:
- $pos$: The position index in the sequence ($0 \le pos < \text{max\_seq\_len}$).
- $i$: The channel index ($0 \le i < D / 2$).
- $D$: The model dimension `d_model`.

### Shape Transformations:
1. **Embedding Matrix ($X_{embed}$)**: Shape `[S, D]` (retrieved from token IDs).
2. **Positional Encoding Slice ($PE_{slice}$)**: Slice of shape `[S, D]` taken from the pre-calculated `[max_seq_len, D]` PE matrix.
3. **Combined Output ($X_{final}$)**: Element-wise addition of embeddings and positional encodings:
   \[
   X_{final} = X_{embed} + PE_{slice}
   \]
   Shape: `[S, D] + [S, D] \to [S, D]`.

### Rust Code (Positional Encoding):
```rust
pub struct PositionalEncoding {
    /// Pre-calculated positional encoding table.
    /// Shape: [max_seq_len, d_model]
    pub pe: Array2<f32>,
}

impl PositionalEncoding {
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let mut pe = Array2::zeros((max_seq_len, d_model));

        for pos in 0..max_seq_len {
            for i in 0..(d_model / 2) {
                // Calculate scale factor: 10000^(2i/D)
                let div_term = 10000.0f32.powf((2 * i) as f32 / d_model as f32);
                
                // Apply sine to even indices and cosine to odd indices
                pe[[pos, 2 * i]] = ((pos as f32) / div_term).sin();
                pe[[pos, 2 * i + 1]] = ((pos as f32) / div_term).cos();
            }
        }

        Self { pe }
    }

    /// Adds positional encoding values to the input embedding matrix.
    /// Input: token embeddings of shape [S, d_model]
    /// Output: combined representations of shape [S, d_model]
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.nrows();
        let d_model = x.ncols();
        let mut output = x.clone();

        // Add the slice pe[0..seq_len, 0..d_model] to the input tensor
        for i in 0..seq_len {
            for j in 0..d_model {
                output[[i, j]] += self.pe[[i, j]];
            }
        }

        output
    }
}
```

---

## 🎯 Understanding Check (理解度チェック)

Let's verify your understanding:

1. **Question 1**: If sequence length $S = 5$ and `d_model = 64`, what is the shape of the output matrix after adding the Token Embedding and the Positional Encoding?
2. **Question 2**: In our `PositionalEncoding` formula, why do we divide the position by a scale factor like $10000^{2i/D}$ instead of just using $\sin(pos)$ for all dimensions? How does this frequency scaling help the model represent absolute and relative positions?

*(Write down your responses and try to visualize the waves in your mind!)*
