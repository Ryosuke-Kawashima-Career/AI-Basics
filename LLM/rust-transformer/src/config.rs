#[derive(Debug, Clone, Copy)]
pub struct Config {
    // number of unique tokens
    pub vocab_size: usize,
    // dimension of the hidden states
    pub dim_model: usize,
    // number of attention heads
    pub num_heads: usize,
    // dimension of the feedforward network
    pub dim_ffn: usize,
    // number of layers
    pub num_layers: usize,
    // dropout rate
    pub dropout_rate: f32,
    // max sequence length
    pub max_seq_len: usize,
}

impl Config {
    pub fn new(
        vocab_size: usize,
        dim_model: usize,
        num_heads: usize,
        dim_ffn: usize,
        num_layers: usize,
        dropout_rate: f32,
        max_seq_len: usize,
    ) -> Self {
        /* Panics when the dimensions of the model is not divisible by the number of heads */
        assert!(
            dim_model % num_heads == 0,
            "The model dimension must be divisible by the number of attention heads"
        );
        Self {
            vocab_size,
            dim_model,
            num_heads,
            dim_ffn,
            num_layers,
            dropout_rate,
            max_seq_len,
        }
    }

    pub fn dim_heads(&self) -> usize {
        self.dim_model / self.num_heads
    }
}

mod tests {
    use super::*;
    #[test]
    fn test_config() {
        let config = Config::new(100, 64, 4, 256, 6, 0.1, 1024);
        assert_eq!(config.dim_heads(), 16);
    }
    #[test]
    #[should_panic(
        expected = "The model dimension must be divisible by the number of attention heads"
    )]
    fn test_config_panic() {
        let _config = Config::new(100, 64, 3, 256, 6, 0.1, 1024);
    }
}
