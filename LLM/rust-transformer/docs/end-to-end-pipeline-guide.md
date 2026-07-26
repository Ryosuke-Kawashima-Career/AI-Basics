# Phase 8: End-to-End Pipeline & Execution Guide (全体パイプラインガイド)

This document explains step-by-step how our Minimum Viable Transformer processes raw English text input tokens into output predictions and loss metrics.

---

## 🔄 End-to-End Dataflow Diagram

```
Input Text: "hell"
     |
     v
[ByteTokenizer]  -->  Token IDs: [104, 101, 108, 108]  (Shape: [S=4])
                             |
                             v
                  [Embedding Lookup (Weight [V, D])]
                             |
                   Token Embeddings [S=4, D=64]
                             |
                             + <-- [Positional Encoding Slice [S=4, D=64]]
                             |
                    Combined Inputs [S=4, D=64]
                             |
                             v
           +-----------------------------------+
           |    Transformer Block Layer 1      |
           |  - Pre-LN Multi-Head Attention    |  --> Residual Addition: X + Attn(LN(X))
           |  - Pre-LN Feed-Forward Network    |  --> Residual Addition: X1 + FFN(LN(X1))
           +-----------------------------------+
                             |
                   Hidden States [S=4, D=64]
                             |
                             v
           +-----------------------------------+
           |    Transformer Block Layer 2      |
           |  - Pre-LN Multi-Head Attention    |
           |  - Pre-LN Feed-Forward Network    |
           +-----------------------------------+
                             |
                   Hidden States [S=4, D=64]
                             |
                             v
                 [Final LayerNorm [S=4, D=64]]
                             |
                             v
             [LM Head Projection (Weight [D, V])]
                             |
                             v
                  Output Logits [S=4, V=256]
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
          [Argmax per Row]       [Softmax per Row]
                 |                       |
      Predicted Next Tokens    Cross-Entropy Loss
    Pos 0: 'h' -> pred ID       Loss = -ln(P(target))
    Pos 1: 'e' -> pred ID
    Pos 2: 'l' -> pred ID
    Pos 3: 'l' -> pred ID
```

---

## 📝 Step-by-Step Algorithm Explanation

### Step 1: Configuration (設定の初期化)
We initialize the hyperparameter configuration specifying:
- $V = 256$ (Vocabulary size: ASCII byte IDs `0..255`)
- $D = 64$ (Model dimension `dim_model`)
- $H = 4$ (Attention heads `num_heads`, each of size $D_k = 16$)
- $D_{ff} = 256$ (Intermediate FFN dimension `dim_ffn`)
- $N = 2$ (Number of stacked Transformer blocks `num_layers`)

### Step 2: Tokenization & Encoding (トークン化)
Given input string `"hell"`:
- The `ByteTokenizer` converts each character to its UTF-8 byte integer representation:
  - `'h'` $\to$ `104`
  - `'e'` $\to$ `101`
  - `'l'` $\to$ `108`
  - `'l'` $\to$ `108`
- Resulting input token IDs: `[104, 101, 108, 108]` of 1D shape `[S=4]`.

Target string for next-token prediction is `"ello"` $\to$ Target token IDs: `[101, 108, 108, 111]`.

### Step 3: Embedding & Positional Encoding Addition
- **Embedding Lookup**: Looks up row `104`, `101`, `108`, `108` in weight matrix of shape `[256, 64]`, producing dense tensor of shape `[4, 64]`.
- **Positional Encoding Slice**: Takes rows `0..4` from sinusoidal matrix of shape `[32, 64]`.
- **Element-wise Addition**: $X = X_{embed} + PE_{slice}$. Shape remains `[4, 64]`.

### Step 4: Stacked Transformer Blocks Processing
The tensor passes through $N=2$ Transformer blocks:
- **Sub-layer 1 (Causal Attention)**: Normalizes $X$, splits $D=64$ into $H=4$ heads of size $D_k=16$, applies causal mask (因果マスク), computes attention weighted sum, projects output, and adds original $X$ back:
  \[
  X_1 = X + \text{MultiHeadAttention}(\text{LayerNorm}_1(X))
  \]
  Shape remains `[4, 64]`.
- **Sub-layer 2 (Feed-Forward)**: Normalizes $X_1$, expands $D=64 \to D_{ff}=256$, applies GELU non-linear activation, compresses $256 \to 64$, and adds $X_1$ back:
  \[
  X_2 = X_1 + \text{FeedForward}(\text{LayerNorm}_2(X_1))
  \]
  Shape remains `[4, 64]`.

After passing through both blocks, final LayerNorm standardizes features: $X_{norm} = \text{LayerNorm}_{final}(X_{layer2})$.

### Step 5: LM Head Projection (Logits Matrix)
Multiply $X_{norm}$ by `lm_head` weight matrix of shape `[64, 256]`:
\[
\text{Logits} = X_{norm} \cdot W_{lm\_head}
\]
Shape: `[4, 64] \times [64, 256] \to \mathbf{[4, 256]}$.

### Step 6: Prediction & Loss Evaluation
For each token position $i \in [0, 3]$:
1. **Next Token Prediction**: Find the ID with highest logit score:
   \[
   \text{Predicted ID}_i = \text{Argmax}_{j \in [0, 255]}(\text{Logits}[i, j])
   \]
2. **Cross-Entropy Loss**: Pass row $i$ through Softmax to get probability distribution $P_i$. The loss for target token ID $t_i$ is:
   \[
   \text{Loss}_i = -\ln(P_i(t_i))
   \]
   Average Loss across sequence length $S=4$:
   \[
   \text{Loss}_{avg} = \frac{1}{4} \sum_{i=0}^3 \text{Loss}_i
   \]

---

## 💻 Rust Code Example (`src/main.rs`)

```rust
mod config;
mod layers;
mod model;
mod ops;
mod tokenizer;

use config::Config;
use model::Transformer;
use ops::softmax;
use tokenizer::ByteTokenizer;

fn main() {
    let config = Config::new(256, 64, 4, 256, 2, 0.0, 32);
    let tokenizer = ByteTokenizer;

    let input_text = "hell";
    let target_text = "ello";

    let input_ids = tokenizer.encode(input_text);
    let target_ids = tokenizer.encode(target_text);
    let seq_len = input_ids.len();

    let transformer = Transformer::new(&config);

    // Forward Pass: [S] -> [S, V]
    let logits = transformer.forward(&input_ids);

    println!("Input Token IDs [S={}]: {:?}", seq_len, input_ids);
    println!("Logits Shape: [S={}, V={}]", logits.nrows(), logits.ncols());

    // Next Token Predictions
    for i in 0..seq_len {
        let row_logits = logits.row(i);
        let mut max_id = 0;
        let mut max_score = f32::NEG_INFINITY;
        for (id, &score) in row_logits.iter().enumerate() {
            if score > max_score {
                max_score = score;
                max_id = id;
            }
        }
        println!("Position {}: Input='{}' -> Predicted ID={}", i, input_text.chars().nth(i).unwrap(), max_id);
    }
}
```
