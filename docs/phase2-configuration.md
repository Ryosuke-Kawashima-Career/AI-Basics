# Phase 2: Configuration Guide (設定ガイド)

In this phase, we define the hyperparameters (ハイパーパラメータ) that govern our Minimum Viable Transformer (最小限の実用可能なトランスフォーマー). These configurations control the sizes of all weight matrices, vectors, and layers in our network.

---

## 🛠️ Hyperparameters in a Transformer

A decoder-only (デコーダーのみ) Transformer requires a few essential dimensions to establish its model capacity and input constraints:

1. **`vocab_size` (語彙数)**: The size of the vocabulary. For character-level (文字レベル) models, this is often `256` (representing ASCII/UTF-8 bytes). For word-level (単語レベル) models, it is the number of unique words in our corpus.
2. **`d_model` (モデルの次元数)**: The feature dimension (特徴量次元) of each token embedding vector. Every input token is represented as a dense vector of shape `[d_model]`.
3. **`n_heads` (アテンションヘッド数)**: The number of attention heads. In Multi-Head Self-Attention (マルチヘッド自己注意機構), the token dimension `d_model` is divided into `n_heads` sub-spaces.
4. **`n_layers` (レイヤー数)**: The number of stacked Transformer Blocks (トランスフォーマーブロック) that make up the network.
5. **`d_ff` (フィードフォワード中間次元数)**: The size of the hidden layer in the Feed-Forward Network (順伝播型ネットワーク). Usually set to $4 \times d\_model$.
6. **`max_seq_len` (最大シーケンス長)**: The maximum context window size (コンテキストウィンドウサイズ).

---

## 📐 Vector and Matrix Dimensions

A critical constraint in Multi-Head Attention (マルチヘッドアテンション) is that the model dimension `d_model` ($D$) must be cleanly divisible by the number of heads `n_heads` ($H$). 

Let $D_k$ be the head dimension (`d_head`):
\[
D_k = \frac{D}{H}
\]
Each head processes a vector slice of shape `[S, D_k]`, where $S$ is the Sequence Length (シーケンス長). If $D$ is not divisible by $H$, the vectors cannot be split evenly across heads.

### Dimension Relationships Table:
| Layer | Parameter / Tensor | Expected Shape |
| :--- | :--- | :--- |
| **Token Embedding** | Weight Matrix | `[vocab_size, d_model]` |
| **Positional Encoding** | Static Tensor | `[max_seq_len, d_model]` |
| **Attention Projections** | Query/Key/Value Weights | `[d_model, d_model]` |
| **Attention Output** | Output Projection Weights | `[d_model, d_model]` |
| **Feed-Forward 1** | First Linear Weight | `[d_model, d_ff]` |
| **Feed-Forward 2** | Second Linear Weight | `[d_ff, d_model]` |
| **Output Logits Head** | Language Modeling Weight | `[d_model, vocab_size]` |

---

## 💻 Rust Implementation Example

Below is the code template for `src/config.rs`. We define a configuration structure and include runtime validation to prevent invalid dimension splits.

```rust
// file: src/config.rs

/// Configuration parameters for the Minimum Viable Transformer.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Vocabulary size (語彙数) - number of unique tokens.
    pub vocab_size: usize,
    /// Model dimension (モデルの次元数) - size of hidden states.
    pub d_model: usize,
    /// Number of attention heads (アテンションヘッド数).
    pub n_heads: usize,
    /// Dimension of the intermediate feed-forward network layer (フィードフォワード中間次元数).
    pub d_ff: usize,
    /// Number of stacked Transformer blocks (レイヤー数).
    pub n_layers: usize,
    /// Maximum sequence length / context window size (最大シーケンス長).
    pub max_seq_len: usize,
}

impl Config {
    /// Creates a new Configuration with validation checks.
    ///
    /// # Panics
    /// Panics if `d_model` is not divisible by `n_heads`.
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        max_seq_len: usize,
    ) -> Self {
        // Assert that the hidden dimension can be evenly divided among attention heads
        assert!(
            d_model % n_heads == 0,
            "d_model ({}) must be divisible by n_heads ({}) to allow clean dimension splitting!",
            d_model,
            n_heads
        );

        Self {
            vocab_size,
            d_model,
            n_heads,
            d_ff,
            n_layers,
            max_seq_len,
        }
    }

    /// Computes the dimension of each attention head (ヘッドごとの次元数).
    ///
    /// Shape calculation: `d_head = d_model / n_heads`
    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Helper to get a tiny model configuration for debugging and testing.
    pub fn tiny_test() -> Self {
        Self::new(
            256, // vocab_size (ASCII character set)
            64,  // d_model (64 hidden features)
            4,   // n_heads (4 heads of size 16)
            256, // d_ff (4 * d_model)
            2,   // n_layers (2 stacked blocks)
            32,  // max_seq_len (32 tokens maximum)
        )
    }
}
```

---

## 🎯 Understanding Check (理解度チェック)

Let's verify your understanding of how Transformer configurations dictate tensor shapes. Try to answer the following questions:

1. **Question 1**: If you set `d_model = 128` and `n_heads = 6`, will the configuration compile and pass the validation check in `Config::new`? Why or why not?
2. **Question 2**: If the sequence length $S = 8$ and we use the `tiny_test` configuration (`d_model = 64`, `n_heads = 4`), what is the shape of the Query matrix ($Q$) for each head *before* attention weight computation, and what is the shape *after* stacking all heads back together?

*(Spend a moment to think about these before looking at the implementation of Multi-Head Self-Attention!)*
