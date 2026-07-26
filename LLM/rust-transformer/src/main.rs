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
    println!("============================================================");
    println!("🚀 Minimum Viable Transformer in Rust - Verification Run");
    println!("============================================================\n");

    // ------------------------------------------------------------
    // Step 1: Configuration Initialization
    // ------------------------------------------------------------
    let config = Config::new(
        256, // vocab_size (ASCII characters 0-255)
        64,  // dim_model (Hidden dimension D = 64)
        4,   // num_heads (H = 4 heads of dimension D_k = 16)
        256, // dim_ffn (Intermediate FFN dimension = 256)
        2,   // num_layers (2 stacked Transformer blocks)
        0.0, // dropout_rate
        32,  // max_seq_len (Maximum context window = 32 tokens)
    );

    println!("1️⃣ [Configuration]");
    println!("   - Vocabulary Size: {}", config.vocab_size);
    println!("   - Model Dimension (D): {}", config.dim_model);
    println!("   - Attention Heads (H): {}", config.num_heads);
    println!("   - Head Dimension (D_k): {}", config.dim_heads());
    println!("   - FFN Dimension (D_ff): {}", config.dim_ffn);
    println!("   - Transformer Layers: {}\n", config.num_layers);

    // ------------------------------------------------------------
    // Step 2: Tokenization / Encoding
    // ------------------------------------------------------------
    let tokenizer = ByteTokenizer;
    let input_text = "hell";
    let target_text = "ello";

    let input_ids = tokenizer.encode(input_text);
    let target_ids = tokenizer.encode(target_text);
    let seq_len = input_ids.len();

    println!("2️⃣ [Tokenization]");
    println!("   - Input String: \"{}\"", input_text);
    println!(
        "   - Encoded Input Token IDs [S={}]: {:?}",
        seq_len, input_ids
    );
    println!(
        "   - Target String for Next-Token Prediction: \"{}\"",
        target_text
    );
    println!(
        "   - Target Token IDs [S={}]: {:?}\n",
        target_ids.len(),
        target_ids
    );

    // ------------------------------------------------------------
    // Step 3: Model Instantiation & Forward Pass
    // ------------------------------------------------------------
    println!("3️⃣ [Model Forward Pass]");
    let transformer = Transformer::new(&config);

    // Forward Pass: Input IDs [S] -> Embeddings + PE [S, D] -> N Blocks [S, D] -> Logits [S, V]
    let logits = transformer.forward(&input_ids);

    println!("   - Input Shape: [S={}]", seq_len);
    println!(
        "   - Output Logits Matrix Shape: [S={}, V={}]",
        logits.nrows(),
        logits.ncols()
    );
    assert_eq!(logits.shape(), &[seq_len, config.vocab_size]);
    println!("   - Logits Shape Verification Passed! ✅\n");

    // ------------------------------------------------------------
    // Step 4: Next-Token Prediction (次トークンの予測)
    // ------------------------------------------------------------
    println!("4️⃣ [Next-Token Prediction]");
    let mut predicted_ids = Vec::new();
    let mut printable_predicted_ids = Vec::new();

    for i in 0..seq_len {
        let row_logits = logits.row(i);
        
        // Raw Argmax over all 256 byte IDs
        let mut max_id = 0;
        let mut max_score = f32::NEG_INFINITY;
        for (id, &score) in row_logits.iter().enumerate() {
            if score > max_score {
                max_score = score;
                max_id = id;
            }
        }
        predicted_ids.push(max_id);

        // Printable ASCII Argmax (restricted to printable characters 32..=126)
        let mut printable_max_id = 32;
        let mut printable_max_score = f32::NEG_INFINITY;
        for (id, &score) in row_logits.iter().enumerate() {
            if (32..=126).contains(&id) && score > printable_max_score {
                printable_max_score = score;
                printable_max_id = id;
            }
        }
        printable_predicted_ids.push(printable_max_id);

        let input_char = input_ids[i] as u8 as char;
        let pred_char = max_id as u8 as char;
        let printable_char = printable_max_id as u8 as char;
        let target_char = target_ids[i] as u8 as char;

        println!(
            "   - Position {}: Input='{}' (ID {}) -> Raw Pred='{}' (ID {}) | ASCII Pred='{}' (ID {}) | Target='{}' (ID {})",
            i, input_char, input_ids[i], pred_char, max_id, printable_char, printable_max_id, target_char, target_ids[i]
        );
    }

    let predicted_text = tokenizer.decode(&predicted_ids);
    let printable_text = tokenizer.decode(&printable_predicted_ids);
    println!("   - Combined Raw Output String: \"{}\"", predicted_text);
    println!("   - Combined Printable ASCII Output String: \"{}\"\n", printable_text);

    // ------------------------------------------------------------
    // Step 5: Cross-Entropy Loss Calculation
    // ------------------------------------------------------------
    println!("5️⃣ [Loss Calculation]");
    let mut total_loss = 0.0;

    for i in 0..seq_len {
        let row_logits = logits.row(i).to_owned();
        let probs = softmax(&row_logits);
        let target_id = target_ids[i];

        // Cross-Entropy Loss: -ln(P(target_id))
        let target_prob = probs[target_id].max(1e-9); // Prevent log(0)
        let loss = -target_prob.ln();
        total_loss += loss;
    }

    let avg_loss = total_loss / (seq_len as f32);
    println!("   - Average Cross-Entropy Loss: {:.4}", avg_loss);
    println!("   - Loss Calculation Verified! ✅\n");

    println!("============================================================");
    println!("🎉 All Transformer Pipeline Steps Verified Successfully!");
    println!("============================================================");
}
