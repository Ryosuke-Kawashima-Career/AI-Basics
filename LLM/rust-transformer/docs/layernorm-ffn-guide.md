# Phase 4: Layer Normalization & Feed-Forward Networks Guide

To stabilize training and allow representation mixing within each token, a Transformer uses two key layers in every block:
1. **Layer Normalization (レイヤー正規化)**: Standardizes the features of each token to prevent vanishing or exploding gradients.
2. **Feed-Forward Network (順伝播型ネットワーク)**: Applies non-linear transformations to each token's vector representation independently.

---

## ⚖️ 1. Layer Normalization (LayerNorm)

Unlike Batch Normalization (バッチ正規化), which normalizes across the batch dimension, Layer Normalization normalizes across the feature dimension $D$ (`d_model`) for each token individually.

### Mathematical Formula:
Given an input vector $x \in \mathbb{R}^D$ representing a single token:
\[
\mu = \frac{1}{D}\sum_{i=1}^D x_i
\]
\[
\sigma^2 = \frac{1}{D}\sum_{i=1}^D (x_i - \mu)^2
\]
\[
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
\]
\[
y_i = \gamma_i \hat{x}_i + \beta_i
\]

Where:
- $\mu$: The mean (平均) of the features for the token. Shape: Scalar.
- $\sigma^2$: The variance (分散) of the features for the token. Shape: Scalar.
- $\epsilon$: A small constant (e.g., $10^{-5}$) to avoid division by zero.
- $\gamma$ (gamma): A learnable scaling vector (スケールベクトル). Shape: `[D]`. Initialized to $1.0$.
- $\beta$ (beta): A learnable shifting vector (シフトベクトル). Shape: `[D]`. Initialized to $0.0$.

### Tensor Shapes:
- **Input ($x$)**: Shape `[S, D]` (Sequence Length $\times$ Model Dimension).
- **Normalized Output ($y$)**: Shape `[S, D]`.

### Rust Code Example (LayerNorm):
```rust
use ndarray::{Array1, Array2, Axis};

pub struct LayerNorm {
    gamma: Array1<f32>, // Shape: [D]
    beta: Array1<f32>,  // Shape: [D]
    epsilon: f32,
}

impl LayerNorm {
    pub fn new(d_model: usize, epsilon: f32) -> Self {
        Self {
            gamma: Array1::ones(d_model), // Initialize scale to 1.0
            beta: Array1::zeros(d_model),  // Initialize shift to 0.0
            epsilon,
        }
    }

    /// Normalizes the input representation.
    /// Shape: [S, D] -> [S, D]
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.nrows();
        let d_model = x.ncols();
        let mut output = Array2::zeros((seq_len, d_model));

        for i in 0..seq_len {
            let row = x.row(i);
            // Calculate Mean (平均)
            let mean = row.mean().unwrap_or(0.0);
            // Calculate Variance (分散)
            let variance = row.fold(0.0, |acc, &val| acc + (val - mean).powi(2)) / (d_model as f32);

            // Apply normalization, scale by gamma, and shift by beta
            for j in 0..d_model {
                let x_hat = (row[j] - mean) / (variance + self.epsilon).sqrt();
                output[[i, j]] = self.gamma[j] * x_hat + self.beta[j];
            }
        }
        output
    }
}
```

---

## ⚡ 2. Feed-Forward Network (FFN)

The Feed-Forward Network (順伝播型ネットワーク) is applied to each token vector individually and identically. It consists of two Linear layers (線形レイヤー) with a Non-linear Activation Function (非線形活性化関数) in between.

### Mathematical Formula:
\[
\text{FFN}(x) = \text{Activation}(x W_1 + b_1) W_2 + b_2
\]

Usually, we use the GELU activation function (GELU活性化関数) in modern Transformers:
\[
\text{GELU}(z) \approx 0.5z \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(z + 0.044715 z^3\right)\right)\right)
\]

### Linear Algebraic Calculation & Tensor Shapes:
Let $S$ be sequence length, $D$ be `d_model`, and $D_{ff}$ be `d_ff` (intermediate dimension).
1. **Input Matrix ($X$)**: Shape `[S, D]`.
2. **First Projection ($X W_1 + b_1$)**:
   - $W_1$ Weight shape: `[D, D_ff]`
   - $b_1$ Bias shape: `[D_ff]` (broadcasted across sequence dimension $S$).
   - Output shape: `[S, D] \times [D, D_ff] \to [S, D_ff]`.
3. **Activation Function**: Applied element-wise. Shape remains `[S, D_ff]`.
4. **Second Projection ($H W_2 + b_2$)**:
   - $W_2$ Weight shape: `[D_ff, D]`
   - $b_2$ Bias shape: `[D]` (broadcasted across sequence dimension $S$).
   - Output shape: `[S, D_ff] \times [D_ff, D] \to [S, D]`.

### Rust Code Example (FeedForward):
```rust
use ndarray::{Array1, Array2};
use rand::distributions::Uniform;
use rand::prelude::*;

pub struct FeedForward {
    w1: Array2<f32>, // Shape: [D, D_ff]
    b1: Array1<f32>, // Shape: [D_ff]
    w2: Array2<f32>, // Shape: [D_ff, D]
    b2: Array1<f32>, // Shape: [D]
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = thread_rng();
        // Initialize weights with small random values
        let dist = Uniform::new(-0.1, 0.1);
        
        let w1 = Array2::from_shape_fn((d_model, d_ff), |_| dist.sample(&mut rng));
        let b1 = Array1::zeros(d_ff);
        let w2 = Array2::from_shape_fn((d_ff, d_model), |_| dist.sample(&mut rng));
        let b2 = Array1::zeros(d_model);

        Self { w1, b1, w2, b2 }
    }

    /// GELU Activation Function
    fn gelu(&self, x: f32) -> f32 {
        let const_sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (const_sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// Computes the forward pass of the FFN.
    /// Shape transition: [S, D] -> [S, D_ff] -> [S, D]
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Step 1: Linear 1 (X * W1 + b1) -> Shape: [S, D_ff]
        let mut intermediate = x.dot(&self.w1);
        for mut row in intermediate.rows_mut() {
            row += &self.b1;
        }

        // Step 2: Apply activation function (GELU) element-wise -> Shape: [S, D_ff]
        intermediate.mapv_inplace(|v| self.gelu(v));

        // Step 3: Linear 2 (H * W2 + b2) -> Shape: [S, D]
        let mut output = intermediate.dot(&self.w2);
        for mut row in output.rows_mut() {
            row += &self.b2;
        }

        output
    }
}
```

---

## 🎯 Understanding Check (理解度チェック)

Let's check your understanding of these layers:

1. **Question 1**: If the intermediate dimension $D_{ff}$ is set to `256`, and the model dimension $D$ is `64`, how many floating-point parameters (Weights + Biases) are in the `FeedForward` layer?
2. **Question 2**: In `LayerNorm`, if we pass an input tensor of shape `[S, D]`, what dimensions are the mean ($\mu$) and variance ($\sigma^2$) calculated over? Is it calculated across the $S$ dimension or the $D$ dimension?

*(Take a moment to write down the math before proceeding!)*
