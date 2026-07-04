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
