# Phase 6: Multi-Head Self-Attention Guide

Multi-Head Self-Attention (マルチヘッド自己注意機構) is the core engine of the Transformer. It allows the model to look at different parts of the input sequence to gather contextual clues. In a decoder-only model (like GPT), we must also enforce a **Causal Mask (因果マスク)** to prevent the model from looking at future tokens during training and generation.

---

## 📐 Step-by-Step Tensor Shapes Walkthrough

Let the dimensions be:
- $S$: Sequence Length (シーケンス長)
- $D$: Model Dimension (モデルの次元数) (`d_model`)
- $H$: Number of Heads (ヘッド数) (`n_heads`)
- $D_k$: Head Dimension (ヘッドの次元数) = $D / H$ (`d_head`)

Here is how the shapes transform during the forward pass:

```
                  Input: X [S, D]
                         |
      +------------------+------------------+
      |                  |                  |
   Linear Q           Linear K           Linear V
   Weight: [D, D]     Weight: [D, D]     Weight: [D, D]
      |                  |                  |
   Q [S, D]           K [S, D]           V [S, D]
      |                  |                  |
   Split & Transpose  Split & Transpose  Split & Transpose
   [H, S, D_k]        [H, S, D_k]        [H, S, D_k]
      |                  |                  |
      +--------+---------+                  |
               |                            |
          Dot Product                       |
      Q * K^T -> [H, S, S]                  |
               |                            |
         Scale (1/sqrt(D_k))                |
               |                            |
       Apply Causal Mask                    |
               |                            |
            Softmax                         |
       Weights: [H, S, S]                   |
               |                            |
               +-------------+--------------+
                             |
                        Dot Product
                  Weights * V -> [H, S, D_k]
                             |
                     Transpose & Merge
                         [S, D]
                             |
                     Linear Output O
                      Weight: [D, D]
                             |
                     Output: Y [S, D]
```

### 1. Projections (Query, Key, Value)
We project our input $X$ of shape `[S, D]` into three spaces using weight matrices $W_q, W_k, W_v$ of shape `[D, D]`:
*   $Q = X W_q$ (Shape: `[S, D]`)
*   $K = X W_k$ (Shape: `[S, D]`)
*   $V = X W_v$ (Shape: `[S, D]`)

### 2. Multi-Head Splitting
We split each projection into $H$ heads.
*   Reshape from `[S, D]` to `[S, H, D_k]`.
*   Transpose (permute axes) to `[H, S, D_k]` so we can perform matrix multiplication on each head independently.

### 3. Scaled Dot-Product Attention
For each head $h$, we calculate the compatibility scores between all queries and keys:
*   $\text{Scores} = Q_h K_h^T$ (Shape: `[S, D_k] \times [D_k, S] \to [S, S]$)
*   Scale by dividing by $\sqrt{D_k}$ to prevent very large dot products (which push softmax gradients to zero):
    \[
    \text{Scaled Scores} = \frac{Q_h K_h^T}{\sqrt{D_k}}
    \]

### 4. Causal Masking
To ensure the token at position $i$ cannot look at a token at position $j > i$ (the future), we add a Causal Mask (因果マスク) $M$ of shape `[S, S]`:
\[
M_{i,j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}
\]
When added to the scores, the $-\infty$ values become $0.0$ after applying the Softmax (ソフトマックス) function:
\[
A = \text{Softmax}(\text{Scaled Scores} + M) \quad \text{(Shape: [S, S])}
\]

### 5. Weighted Sum
Multiply the attention weights $A$ by the values $V_h$:
*   $\text{Output}_h = A V_h$ (Shape: `[S, S] \times [S, D_k] \to [S, D_k]$)

### 6. Concatenation and Output Projection
*   Combine all heads back from shape `[H, S, D_k]` to `[S, H, D_k]` and reshape to `[S, D]`.
*   Project through the output weight matrix $W_o$ of shape `[D, D]`:
    \[
    Y = \text{Output} \cdot W_o \quad \text{(Shape: [S, D])}
    \]

---

## 💻 Rust Code Example

Below is the code template for `MultiHeadAttention` to be implemented inside `src/layers.rs`.

```rust
use ndarray::{Array2, Array3, Axis};
use rand::distributions::Uniform;
use rand::prelude::*;

pub struct MultiHeadAttention {
    n_heads: usize,
    d_head: usize,
    wq: Array2<f32>, // Shape: [D, D]
    wk: Array2<f32>, // Shape: [D, D]
    wv: Array2<f32>, // Shape: [D, D]
    wo: Array2<f32>, // Shape: [D, D]
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert!(d_model % n_heads == 0);
        let d_head = d_model / n_heads;
        let mut rng = thread_rng();
        let dist = Uniform::new(-0.1, 0.1);

        let wq = Array2::from_shape_fn((d_model, d_model), |_| dist.sample(&mut rng));
        let wk = Array2::from_shape_fn((d_model, d_model), |_| dist.sample(&mut rng));
        let wv = Array2::from_shape_fn((d_model, d_model), |_| dist.sample(&mut rng));
        let wo = Array2::from_shape_fn((d_model, d_model), |_| dist.sample(&mut rng));

        Self {
            n_heads,
            d_head,
            wq,
            wk,
            wv,
            wo,
        }
    }

    /// Custom stable softmax along the last axis of a 2D matrix
    fn softmax_rowwise(&self, mut matrix: Array2<f32>) -> Array2<f32> {
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();
        for i in 0..nrows {
            let mut row = matrix.row_mut(i);
            // Find max value for numerical stability
            let max_val = row.fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
            // Exponentiate
            row.mapv_inplace(|v| (v - max_val).exp());
            // Sum and divide
            let sum: f32 = row.sum();
            row.mapv_inplace(|v| v / (sum + 1e-9));
        }
        matrix
    }

    /// Computes the forward pass of Multi-Head Self-Attention.
    /// Shape: [S, D] -> [S, D]
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.nrows();
        let d_model = x.ncols();

        // 1. Projections: [S, D] x [D, D] -> [S, D]
        let q_proj = x.dot(&self.wq);
        let k_proj = x.dot(&self.wk);
        let v_proj = x.dot(&self.wv);

        // To make implementation simple and clean for learning, we will loop 
        // through the heads to avoid complex tensor permutation syntax.
        let mut head_outputs = Vec::new();

        let scale = (self.d_head as f32).sqrt();

        for h in 0..self.n_heads {
            let start = h * self.d_head;
            let end = start + self.d_head;

            // Extract the slice for the current head: Shape [S, D_k]
            let q_h = q_proj.slice(ndarray::s![.., start..end]).to_owned();
            let k_h = k_proj.slice(ndarray::s![.., start..end]).to_owned();
            let v_h = v_proj.slice(ndarray::s![.., start..end]).to_owned();

            // 2. Scores: [S, D_k] x [D_k, S] -> [S, S]
            let mut scores = q_h.dot(&k_h.t());

            // 3. Scale scores
            scores.mapv_inplace(|v| v / scale);

            // 4. Apply Causal Mask (因果マスクの適用)
            // Mask out future tokens by setting them to a very large negative value
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    scores[[i, j]] = -1e9; // Represents -infinity
                }
            }

            // 5. Apply Softmax along the columns to get Attention Weights: Shape [S, S]
            let weights = self.softmax_rowwise(scores);

            // 6. Weighted Sum: [S, S] x [S, D_k] -> [S, D_k]
            let output_h = weights.dot(&v_h);
            head_outputs.push(output_h);
        }

        // 7. Concatenate all heads: [S, D_k] x H -> [S, D]
        let mut concatenated = Array2::zeros((seq_len, d_model));
        for h in 0..self.n_heads {
            let start = h * self.d_head;
            let end = start + self.d_head;
            concatenated.slice_mut(ndarray::s![.., start..end]).assign(&head_outputs[h]);
        }

        // 8. Output Linear Projection: [S, D] x [D, D] -> [S, D]
        concatenated.dot(&self.wo)
    }
}
```

---

## 🎯 Understanding Check (理解度チェック)

Let's test your understanding of Multi-Head Self-Attention:

1.  **Question 1**: If the input sequence is `"cat sat"`, representing a sequence length $S = 2$. What is the shape of the attention weight matrix $A_h$ for each head? According to the causal mask rules, what must the value at index `[0, 1]` in $A_h$ be after Softmax?
2.  **Question 2**: Why do we divide query-key dot products by $\sqrt{D_k}$ (the square root of the head dimension) before applying softmax? What would happen to gradients if $D_k$ became very large (e.g., 1024) and we did not scale?

*(Think about how the softmax curve flattens out for large numbers!)*
