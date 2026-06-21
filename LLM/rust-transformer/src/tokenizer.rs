pub struct ByteTokenizer;
/*Embedding = Input: text (shape: [S]) X Embedding lookup (shape: [V(vocab_size), D(embedding_dim)])
-> Output: token_embeddings (shape: [S, D])
 */
impl ByteTokenizer {
    // Encodes a string into a sequence of byte-based token IDs
    // Shape: String -> [S: Sequence length] (1D tensor of length S)
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.as_bytes().iter().map(|&b| b as usize).collect()
    }

    // Decodes a sequence of byte-based token IDs back into a string
    // Shape: [S: Sequence length] -> String
    pub fn decode(&self, ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids.iter().map(|&id| id as u8).collect();
        String::from_utf8(bytes).unwrap_or_else(|_| String::from("[INVALID UTF-8]"))
    }
}

use std::collections::HashMap;

pub struct WordTokenizer {
    vocab: HashMap<String, usize>,
    id_to_word: Vec<String>,
}

impl WordTokenizer {
    // Builds a vocabulary map from a training text corpus
    // Shape: String -> Vocab size V
    pub fn new(corpus: &str) -> Self {
        // Word: ID
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut id_to_word: Vec<String> = Vec::new();

        // <PAD>: Head <UNK>: Unknown <EOS>: End of sentence
        let special_tokens = vec!["<PAD>", "<UNK>", "<EOS>"];
        let mut words: Vec<String> = Vec::new();
        for word in corpus.split_whitespace() {
            let cleaned = word.to_lowercase().replace(&['.', ',', '!', '?'][..], "");
            if cleaned.is_empty() {
                continue;
            }
            words.push(cleaned);
        }
        let tokens: Vec<String> = special_tokens
            .iter()
            .map(|s| s.to_string())
            .chain(words.into_iter())
            .collect();

        for token in tokens.iter() {
            if !vocab.contains_key(token) {
                let id: usize = id_to_word.len();
                vocab.insert(token.to_string(), id);
                id_to_word.push(token.to_string());
            }
        }
        Self { vocab, id_to_word }
    }

    // Encodes text into a sequence of Word IDs (includes <PAD> at index 0 and <EOS> at index 2)
    // Shape: String -> [S: Sequence length] where S = num_words + 2
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut encoded: Vec<usize> = Vec::new();
        // <PAD>
        encoded.push(0);
        for word in words.iter() {
            let cleaned: String = word.to_lowercase().replace(&['.', ',', '!', '?'][..], "");
            if cleaned.is_empty() {
                continue;
            }
            if let Some(id) = self.vocab.get(&cleaned) {
                encoded.push(*id);
            } else {
                // <UNK>
                encoded.push(1);
            }
        }
        // <EOS>
        encoded.push(2);
        encoded
    }

    // Decodes Word IDs back to a space-separated string, filtering out <PAD> and <EOS>
    // Shape: [S: Sequence length] -> String
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut words: Vec<&str> = Vec::new();
        for id in ids {
            let word: &str = match id {
                0 => "<PAD>",
                1 => "<UNK>",
                2 => "<EOS>",
                _ => {
                    if let Some(word) = self.id_to_word.get(*id) {
                        word.as_str()
                    } else {
                        "<UNK>"
                    }
                }
            };
            if word != "<PAD>" && word != "<EOS>" {
                words.push(word);
            }
        }
        words.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_tokenizer_encode_decode_roundtrip() {
        let tokenizer = ByteTokenizer;
        let text = "Hello, world!";
        let ids = tokenizer.encode(text);
        // Input text shape: scalar string
        // Encoded IDs shape: [S] (length 13)
        assert_eq!(ids.len(), 13);
        let decoded = tokenizer.decode(&ids);
        // Decoded string shape: scalar string
        assert_eq!(text, decoded);
    }

    #[test]
    fn test_byte_tokenizer_utf8() {
        let tokenizer = ByteTokenizer;
        let text = "Rust 🦀";
        let ids = tokenizer.encode(text);
        // Input text shape: scalar string
        // Encoded IDs shape: [S] (length 9, 'R','u','s','t',' ' plus 4 bytes for 🦀)
        assert_eq!(ids.len(), 9);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(text, decoded);
    }

    #[test]
    fn test_byte_tokenizer_invalid_utf8() {
        let tokenizer = ByteTokenizer;
        // 0xC0 and 0xAF are invalid UTF-8 byte sequences
        let ids = vec![0xC0, 0xAF];
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "[INVALID UTF-8]");
    }

    #[test]
    fn test_word_tokenizer_vocabulary_creation() {
        let corpus = "The quick brown fox jumps over the lazy dog.";
        let tokenizer = WordTokenizer::new(corpus);

        // Let's verify special tokens: <PAD> = 0, <UNK> = 1, <EOS> = 2
        assert_eq!(tokenizer.id_to_word[0], "<PAD>");
        assert_eq!(tokenizer.id_to_word[1], "<UNK>");
        assert_eq!(tokenizer.id_to_word[2], "<EOS>");

        // Casing should be normalized (lowercase)
        assert!(tokenizer.vocab.contains_key("the"));
        assert!(tokenizer.vocab.contains_key("dog"));
        // Punctuation should be stripped
        assert!(!tokenizer.vocab.contains_key("dog."));
    }

    #[test]
    fn test_word_tokenizer_encode_decode() {
        let corpus = "I love programming in Rust.";
        let tokenizer = WordTokenizer::new(corpus);

        let text = "I love Rust!";
        let ids = tokenizer.encode(text);
        // Expected shape of ids: [S] where S = 5 (PAD, "i", "love", "rust", EOS)
        // Note that 'Rust!' is cleaned to 'rust' and 'I' to 'i'.
        assert_eq!(ids, vec![0, 3, 4, 7, 2]);

        let decoded = tokenizer.decode(&ids);
        // Expected output: "i love rust" (special tokens stripped, lowercase, punctuation removed)
        assert_eq!(decoded, "i love rust");
    }

    #[test]
    fn test_word_tokenizer_unknown_word() {
        let corpus = "I love programming in Rust.";
        let tokenizer = WordTokenizer::new(corpus);

        let text = "I love Python.";
        let ids = tokenizer.encode(text);
        // "Python." gets cleaned to "python", which is not in vocab. It should map to <UNK> (1).
        // IDs: [0, 3, 4, 1, 2] (PAD, "i", "love", UNK, EOS)
        assert_eq!(ids, vec![0, 3, 4, 1, 2]);

        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "i love <UNK>");
    }
}
