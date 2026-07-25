use crate::config::Config;
use crate::layers::{Embedding, FeedForward, LayerNorm, MultiHeadAttention, PositionalEncoding};
use ndarray::Array2;
use rand::distributions::Uniform;
use rand::prelude::*;

pub struct TransformerBlock {
    layer_norm_1: LayerNorm,
    layer_norm_2: LayerNorm,
    feed_forward: FeedForward,
    attention: MultiHeadAttention,
}

impl TransformerBlock {
    pub fn new(config: &Config) -> Self {
        Self {
            layer_norm_1: LayerNorm::new(config.dim_model, 1e-5),
            attention: MultiHeadAttention::new(config.dim_model, config.num_heads),
            layer_norm_2: LayerNorm::new(config.dim_model, 1e-5),
            feed_forward: FeedForward::new(config.dim_model, config.dim_ffn),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        /*Forward for [Seq_len, Dim_model] -> [Seq_len, Dim_model]*/
        let x_norm1: Array2<f32> = self.layer_norm_1.forward(x);
        let attention_out: Array2<f32> = self.attention.forward(&x_norm1);
        let x1: Array2<f32> = x + &attention_out;
        let x_norm2: Array2<f32> = self.layer_norm_2.forward(&x1);
        let ffn_out: Array2<f32> = self.feed_forward.forward(&x_norm2);
        let x2: Array2<f32> = &x1 + &ffn_out;
        x2
    }
}

pub struct Transformer {
    embedding: Embedding,
    positional_encoding: PositionalEncoding,
    transformer_blocks: Vec<TransformerBlock>,
    layer_norm_final: LayerNorm,
    // Shape: [Dim_model, Vocabulary]
    lm_head: Array2<f32>,
}

impl Transformer {
    pub fn new(config: &Config) -> Self {
        let embedding = Embedding::new(config.vocab_size, config.dim_model);
        let positional_encoding = PositionalEncoding::new(config.max_seq_len, config.dim_model);
        let transformer_blocks = (0..config.num_layers)
            .map(|_| TransformerBlock::new(config))
            .collect();
        let layer_norm_final = LayerNorm::new(config.dim_model, 1e-5);
        // initialize LM Head weight of shape [dim_model, vocab_size]
        let mut rng = thread_rng();
        let distribution = Uniform::new(-0.1, 0.1);
        let lm_head = Array2::from_shape_fn((config.dim_model, config.vocab_size), |_| {
            distribution.sample(&mut rng)
        });
        Self {
            embedding,
            positional_encoding,
            transformer_blocks,
            layer_norm_final,
            lm_head,
        }
    }

    pub fn forward(&self, token_ids: &[usize]) -> Array2<f32> {
        /*Input: token_ids with shape: [Seq_len]
        Output: Output logits with shape: [S, vocab_size]
        */
        // Embeddings [Sequence] -> [Sequence, Dim_model]
        let embeddings = self.embedding.forward(token_ids);
        // Add Positional Encodings [Sequence, Dim_model] -> [Sequence, Dim_model]
        let mut x: Array2<f32> = self.positional_encoding.forward(&embeddings);
        // Pass Through N stacked Transformer Blocks [Sequence, Dim_model] -> [Sequence, Dim_model]
        for block in &self.transformer_blocks {
            x = block.forward(&x);
        }
        // Final Layer Normalization [Sequence, Dim_model]
        let x_norm = self.layer_norm_final.forward(&x);
        // LM head projection [Sequence, Vocabulary] = [Sequence, Dim_model] X [Dim_model, Vocabulary]
        let logits: Array2<f32> = x_norm.dot(&self.lm_head);
        logits
    }
}

// ==========================================
// Unit Tests for Model Assembly
// ==========================================
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_transformer_block_forward() {
        let config = Config::new(256, 64, 4, 256, 2, 0.0, 32);
        let block = TransformerBlock::new(&config);

        // Input shape: [S=4, D=64]
        let x = Array2::zeros((4, 64));
        let output = block.forward(&x);

        // Verify shape remains [4, 64]
        assert_eq!(output.shape(), &[4, 64]);
    }

    #[test]
    fn test_transformer_end_to_end_forward() {
        let config = Config::new(256, 64, 4, 256, 2, 0.0, 32);
        let transformer = Transformer::new(&config);

        // Input token IDs: "hell" -> [104, 101, 108, 108] (Sequence length S = 4)
        let token_ids = vec![104, 101, 108, 108];
        let logits = transformer.forward(&token_ids);

        // Verify output shape is [S=4, V=256]
        assert_eq!(logits.shape(), &[4, 256]);

        // Verify values are valid floating point numbers (no NaN or Inf)
        assert!(logits.iter().all(|&val| !val.is_nan() && !val.is_infinite()));
    }
}

