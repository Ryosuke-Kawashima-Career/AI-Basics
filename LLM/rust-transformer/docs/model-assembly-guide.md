# Phase 7: Model Assembly Guide (モデル構築ガイド)

In this phase, we connect all individual components—Embedding, Positional Encoding, Multi-Head Self-Attention, LayerNorm, and Feed-Forward Networks—into a unified Decoder-only Transformer architecture.

The assembly happens in two hierarchy steps:
1. **`TransformerBlock` (トランスフォーマーブロック)**: Groups Attention, Feed-Forward, LayerNorms, and Residual Connections.
2. **`Transformer` Model (トランスフォーマー全体系)**: Stacks multiple blocks and attaches the Language Modeling Head (LM Head) to project hidden representations into vocabulary logits.

---

## 🏗️ 1. Transformer Block & Residual Connections

A `TransformerBlock` uses **Pre-LayerNorm (事前レイヤー正規化)** architecture, which is the standard design in modern LLMs (e.g. GPT-3, LLaMA) for numerical stability.

### Mathematical Formulation:
Given hidden state tensor $X$ of shape `[S, D]`:

1.  **Self-Attention Sub-layer with Residual Connection (残差接続)**:
    \[
    X_1 = X + \text{Attention}(\text{LayerNorm}_1(X))
    \]
    *   $\text{LayerNorm}_1(X)$ shape: `[S, D]`
    *   $\text{Attention}(\dots)$ shape: `[S, D]`
    *   $X_1$ shape: `[S, D]` (element-wise addition with $X$)

2.  **Feed-Forward Sub-layer with Residual Connection**:
    \[
    X_2 = X_1 + \text{FeedForward}(\text{LayerNorm}_2(X_1))
    \]
    *   $\text{LayerNorm}_2(X_1)$ shape: `[S, D]`
    *   $\text{FeedForward}(\dots)$ shape: `[S, D]`
    *   $X_2$ shape: `[S, D]` (element-wise addition with $X_1$)

---

## 🏛️ 2. Complete Model Architecture & Logits Projection

The overall `Transformer` pipeline transforms raw input token IDs into vocabulary prediction Logits (ロジット).

### Step-by-Step Shape Flow:
1.  **Input Token IDs**: `[S]` (1D array of token integers).
2.  **Token Embedding + Positional Encoding**: Output shape `[S, D]`.
3.  **Stacked Transformer Blocks**: Passes through $N$ blocks (`n_layers`). Shape remains `[S, D]`.
4.  **Final LayerNorm**: Standardizes final representations. Shape `[S, D]`.
5.  **Language Modeling Head (LM Head / 出力投影層)**:
    Projects model dimension $D$ to vocabulary size $V$ using weight matrix of shape `[D, V]`:
    \[
    \text{Logits} = X_{norm} \cdot W_{lm\_head}
    \]
    *   Output shape: `[S, D] \times [D, V] \to [S, V]`.
    *   Each row $i \in [0, S-1]$ contains $V$ unnormalized log-probability scores for predicting the next token!

---

## 💻 Rust Reference Implementation

```rust
use ndarray::{Array2, Axis};
use rand::distributions::Uniform;
use rand::prelude::*;
use crate::config::Config;
use crate::layers::{Embedding, PositionalEncoding, LayerNorm, FeedForward, MultiHeadAttention};

/// A single Transformer Block combining Attention, FFN, and Residual Connections.
pub struct TransformerBlock {
    ln1: LayerNorm,
    attn: MultiHeadAttention,
    ln2: LayerNorm,
    ffn: FeedForward,
}

impl TransformerBlock {
    pub fn new(config: &Config) -> Self {
        Self {
            ln1: LayerNorm::new(config.d_model, 1e-5),
            attn: MultiHeadAttention::new(config.d_model, config.n_heads),
            ln2: LayerNorm::new(config.d_model, 1e-5),
            ffn: FeedForward::new(config.d_model, config.d_ff),
        }
    }

    /// Forward pass through the Transformer Block.
    /// Shape: [S, D] -> [S, D]
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // 1. Sub-layer 1: Pre-LN Attention + Residual Connection
        let x_norm1 = self.ln1.forward(x);
        let attn_out = self.attn.forward(&x_norm1);
        let x1 = x + &attn_out; // Residual addition

        // 2. Sub-layer 2: Pre-LN FFN + Residual Connection
        let x_norm2 = self.ln2.forward(&x1);
        let ffn_out = self.ffn.forward(&x_norm2);
        let x2 = &x1 + &ffn_out; // Residual addition

        x2
    }
}

/// The complete Decoder-only Transformer Model.
pub struct Transformer {
    embedding: Embedding,
    pos_encoding: PositionalEncoding,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Array2<f32>, // Shape: [D, V]
}

impl Transformer {
    pub fn new(config: &Config) -> Self {
        let embedding = Embedding::new(config.vocab_size, config.d_model);
        let pos_encoding = PositionalEncoding::new(config.max_seq_len, config.d_model);
        
        let blocks = (0..config.n_layers)
            .map(|_| TransformerBlock::new(config))
            .collect();
            
        let ln_f = LayerNorm::new(config.d_model, 1e-5);
        
        // Initialize LM Head weight of shape [d_model, vocab_size]
        let mut rng = thread_rng();
        let dist = Uniform::new(-0.1, 0.1);
        let lm_head = Array2::from_shape_fn((config.d_model, config.vocab_size), |_| dist.sample(&mut rng));

        Self {
            embedding,
            pos_encoding,
            blocks,
            ln_f,
            lm_head,
        }
    }

    /// End-to-End Forward Pass.
    /// Input: token_ids slice of shape [S]
    /// Output: Logits matrix of shape [S, V]
    pub fn forward(&self, token_ids: &[usize]) -> Array2<f32> {
        // Step 1: Embeddings [S] -> [S, D]
        let embeds = self.embedding.forward(token_ids);
        
        // Step 2: Add Positional Encodings [S, D] -> [S, D]
        let mut x = self.pos_encoding.forward(&embeds);
        
        // Step 3: Pass through N stacked Transformer Blocks [S, D] -> [S, D]
        for block in &self.blocks {
            x = block.forward(&x);
        }
        
        // Step 4: Final LayerNorm [S, D] -> [S, D]
        let x_norm = self.ln_f.forward(&x);
        
        // Step 5: LM Head Projection [S, D] x [D, V] -> [S, V]
        x_norm.dot(&self.lm_head)
    }
}
```
