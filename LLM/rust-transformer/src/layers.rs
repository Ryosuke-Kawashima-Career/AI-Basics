use ndarray::{Array1, Array2, ArrayView1};
pub struct LayerNorm {
    /*Standardizes the features of each token
        Formula: y = gamma * (x - mu) / sqrt(var + epsilon) + beta
        - mu: mean of the features
        - var: variance of the features
        - epsilon: small value to avoid division by zero
        - gamma: learnable scaling parameter
        - beta: learnable shifting parameter
    */
    // Scaling(Weight) vector shape: [Dim_model]
    gamma: Array1<f32>,
    // Shift(Bias) vector shape: [Dim_model]
    beta: Array1<f32>,
    // A small value for avoiding division by zero
    epsilon: f32,
}

impl LayerNorm {
    pub fn new(dim_model: usize, epsilon: f32) -> Self {
        Self {
            gamma: Array1::ones(dim_model),
            beta: Array1::zeros(dim_model),
            epsilon: epsilon,
        }
    }
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        /*Normalizes the input tokens
           Input: Input token representations with shape: [S, Dim_model]
           Output: Normalized token representations with shape: [S, Dim_model]
        */
        let seq_len: usize = x.shape()[0];
        let dim_model: usize = x.shape()[1];
        let mut output: Array2<f32> = Array2::zeros((seq_len, dim_model));
        for token in 0..seq_len {
            let token_vec: ArrayView1<f32> = x.row(token);
            let mean: f32 = token_vec.mean().unwrap_or(0.0);
            let variance: f32 = token_vec
                .mapv(|feature| (feature - mean).powi(2))
                .mean()
                .unwrap_or(1.0);
            for feature in 0..dim_model {
                let x_normalized: f32 =
                    (token_vec[feature] - mean) / (variance + self.epsilon).sqrt();
                output[(token, feature)] = self.gamma[feature] * x_normalized + self.beta[feature];
            }
        }
        output
    }
}

use rand::distributions::Uniform;
use rand::prelude::*;

pub struct FeedForward {
    // shape: [Dim_model, Dim_ffn]
    w1: Array2<f32>,
    // shape: [Dim_ffn]
    b1: Array1<f32>,
    // shape: [Dim_ffn, Dim_model]
    w2: Array2<f32>,
    // shape: [Dim_model]
    b2: Array1<f32>,
}

impl FeedForward {
    pub fn new(dim_model: usize, dim_ffn: usize) -> Self {
        let mut rng = thread_rng();
        let uniform_distribution = Uniform::new(-0.1, 0.1);
        let w1: Array2<f32> = Array2::from_shape_fn((dim_model, dim_ffn), |_| {
            uniform_distribution.sample(&mut rng)
        });
        let b1: Array1<f32> = Array1::zeros(dim_ffn);
        let w2: Array2<f32> = Array2::from_shape_fn((dim_ffn, dim_model), |_| {
            uniform_distribution.sample(&mut rng)
        });
        let b2: Array1<f32> = Array1::zeros(dim_model);
        Self { w1, b1, w2, b2 }
    }
    fn gelu(&self, x: f32) -> f32 {
        // gelu activation function
        let const_sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (const_sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
    }
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        /*Expands and contracts token representations
           Input: Input token representations with shape: [S, Dim_model]
           Medium: [S Dim_ffn]
           Output: Transformed token representations with shape: [S, Dim_model]
        */
        // Step 1: Linear 1 (X * W1 + b1) -> Shape: [S, D_ff]
        let mut intermediate: Array2<f32> = x.dot(&self.w1);
        for mut row in intermediate.rows_mut() {
            row += &self.b1;
        }
        // Step 2: GeLU for elementwise -> Shape: [S, D_ff]
        intermediate.mapv(|v| self.gelu(v));
        // Step 3: Linear 2 (Intermediate * W2 + b2) -> Shape: [S, Dim_model]
        let mut output: Array2<f32> = intermediate.dot(&self.w2);
        for mut row in output.rows_mut() {
            row += &self.b2;
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_layer_norm_shapes_and_math() {
        let d_model = 4;
        let epsilon = 1e-5;
        let ln = LayerNorm::new(d_model, epsilon);
        // Input shape: [S=2, D=4] (Sequence Length = 2, Model Dimension = 4)
        let x = array![[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 10.0, 10.0]];
        // Run forward pass
        let y = ln.forward(&x);
        // 1. Shape Verification: Output shape must be [2, 4]
        assert_eq!(y.shape(), &[2, 4]);
        // 2. Mathematical Verification:
        // For the first row [1.0, 2.0, 3.0, 4.0]:
        //   Mean (μ) = 2.5
        //   Variance (σ²) = ((1.5)² + (0.5)² + (-0.5)² + (-1.5)²) / 4 = 1.25
        //   Standardized row = (x - 2.5) / sqrt(1.25)
        let expected_row_0 = array![
            (1.0 - 2.5) / 1.25f32.sqrt(),
            (2.0 - 2.5) / 1.25f32.sqrt(),
            (3.0 - 2.5) / 1.25f32.sqrt(),
            (4.0 - 2.5) / 1.25f32.sqrt()
        ];

        // Verify output matches calculated expectations within float tolerance
        for j in 0..d_model {
            assert!((y[[0, j]] - expected_row_0[j]).abs() < 1e-5);
        }
        // For the second row [10.0, 10.0, 10.0, 10.0]:
        //   Mean (μ) = 10.0, Variance (σ²) = 0.0
        //   Output should normalize all elements to exactly 0.0
        for j in 0..d_model {
            assert!((y[[1, j]] - 0.0).abs() < 1e-5);
        }
    }
    #[test]
    fn test_feed_forward_shapes() {
        let d_model = 4;
        let d_ff = 16;
        let ffn = FeedForward::new(d_model, d_ff);
        // Input shape: [S=3, D=4] (Sequence Length = 3, Model Dimension = 4)
        let x = array![
            [0.5, -0.2, 0.1, 0.9],
            [0.0, 0.1, -0.8, 0.2],
            [1.2, -1.1, 0.5, 0.3]
        ];
        // Run forward pass
        let y = ffn.forward(&x);
        // 1. Shape Verification: Output shape must match input shape [3, 4]
        // Linear transformations: [3, 4] x [4, 16] -> [3, 16] -> [3, 16] x [16, 4] -> [3, 4]
        assert_eq!(y.shape(), &[3, 4]);
        // 2. Check that GELU activation was applied (output is non-linear and not all zeros)
        assert!(y.iter().any(|&val| val != 0.0));
    }
}
