# Q&A Minutes

This document records the key insights, explanations, and decisions made during Q&A sessions.

## Entry #1: AI Mastery Roadmap (MLP to RAG)
**Timestamp:** 2026-05-27

### Summary
A structured progression roadmap for mastering AI technologies, starting with basic Multi-Layer Perceptrons (MLPs) and building up to advanced Retrieval-Augmented Generation (RAG) systems.

### Issue
Connecting foundational feedforward neural networks (which learn static weights from inputs) to modern LLM-based applications (which require dynamic external knowledge retrieval and prompt engineering) can be conceptually overwhelming.

### Approach
1. **Foundation (MLP):** Master backpropagation, gradient descent, and basic activation functions by building a neural network from scratch.
2. **Sequential & Attention Models (Transformers):** Move beyond feedforward models to self-attention mechanisms, understanding how sequence length and positional encoding work.
3. **Application & Augmentation (RAG):** Integrate pre-trained LLMs with custom external data using vector embeddings, document chunking, and similarity searches.

### Example or Analogy
- **MLP is like learning basic vocabulary:** You learn what individual words mean (simple input-to-output mappings).
- **Transformer is like writing a coherent story:** You learn to pay attention to context and relationships between words across a page (attention mechanisms).
- **RAG is like writing a research paper with an open textbook:** Instead of just relying on what you memorized, you look up specific information in a reference library (vector database) to answer a specific prompt accurately.

## Entry #2: LayerNorm & FeedForward Parameters and Operations
**Timestamp:** 2026-07-05

### Summary
Calculation of model parameters for the Feed-Forward Network (FFN) layer and identifying the axis of calculation for Layer Normalization (LayerNorm) within a Transformer block.

### Issue
Determining how hyperparameter dimensions dictate parameters in neural layers and understanding whether normalization occurs along the sequence or feature dimension.

### Approach
1. **FeedForward Layer Parameter Count**:
   - Linear 1 Weight shape `[D, d_ff]` and Bias shape `[d_ff]`.
   - Linear 2 Weight shape `[d_ff, D]` and Bias shape `[D]`.
   - For $D = 64$ and $d_{ff} = 256$, the total parameter count is:
     \[ (64 \times 256 + 256) + (256 \times 64 + 64) = 16,640 + 16,448 = 33,088 \]
2. **LayerNorm Normalization Axis**:
   - LayerNorm normalizes elements across the model/feature dimension ($D$) for each token independently. It does not normalize across the sequence length dimension ($S$).

### Example or Analogy
- **FeedForward count is like counting rows in a two-stage warehouse**: You calculate the boxes stored on the ground floor ($D \times d_{ff}$ weights + $d_{ff}$ labels), and then the boxes moved to the shipping dock ($d_{ff} \times D$ weights + $D$ labels).
- **LayerNorm is like grading each student individually on their own report card**: Instead of finding the class average for a single subject (Batch Normalization), LayerNorm averages all subject grades for a single student (features) so that every student's individual average is normalized to the same scale.

## Entry #3: Linear Layers and FFN Dimension Expansion
**Timestamp:** 2026-07-05

### Summary
Explaining why we implement Linear Layers even when the tensor shape is unchanged, and clarifying the role of intermediate dimension expansion ($d_{ff}$) in Feed-Forward Networks.

### Issue
Addressing the misconception that Linear Layers are only used to change tensor shapes, and explaining why Transformers project embeddings into a higher-dimensional space ($d_{ff}$) and back.

### Approach
1. **Linear Layer Transformation ($Y = XW + B$)**: Even if the input shape `[S, D]` and output shape `[S, D]` are identical, the linear layer mixes features. Each output feature is a learned weighted sum of all input features.
2. **Dimension Expansion ($d_{ff}$)**: Expanding the model dimension $D \to d_{ff}$ (typically $4 \times D$), applying a non-linear activation (e.g. GELU), and projecting back to $D$ allows the network to partition features into a higher-dimensional space where non-linear patterns are easier to learn and separate.

### Example or Analogy
- **Linear Layer is like cooking soup**: The ingredients (features) are mixed together in a pot with spices (weights) to create a new flavor (transformed features). The pot (tensor shape) remains exactly the same, but the taste (data representation) is completely changed.
- **FFN Expansion is like unpacking a folded tent**: You unpack the compact tent from its carrying bag ($D$) into a large open field ($d_{ff}$), make your structural changes or attach canopies (non-linear activations), and then fold it back up into the compact bag ($D$) for transport to the next layer.

## Entry #4: Embedding and Positional Encoding Shapes and Frequencies
**Timestamp:** 2026-07-05

### Summary
Evaluation of the output shape in token embedding + positional encoding addition, and explanation of why sinusoidal frequencies are scaled by $10000^{2i/D}$.

### Issue
Understanding how element-wise addition preserves tensor shapes, and explaining how frequency scaling across model dimensions helps represent absolute and relative position relationships.

### Approach
1. **Shape Preservation**: Adding token embedding matrix ($[S, D]$) and positional encoding matrix ($[S, D]$) element-wise preserves the dimension shape as $[S, D]$. For $S=5$ and $D=64$, the shape is exactly $[5, 64]$.
2. **Frequency Scaling**: The division by $10000^{2i/D}$ scales the wavelength of the waves across the dimension channels $i$. Low-index dimensions have high-frequency waves, while high-index dimensions have low-frequency waves. This allows the model to compute relative offsets (distances) as linear transformations.

### Example or Analogy
- **Shape addition is like combining two template slides on a projector**: Overlaying two $5 \times 64$ slide grids results in a combined grid of exactly $5 \times 64$.
- **Frequency scaling is like the hands of a clock**: The second hand ($i=0$) moves fast, the minute hand ($i=1$) moves slower, and the hour hand ($i=2$) moves very slowly. A clock can tell absolute time and relative duration because the hands move at different scales. If all hands moved at the same speed, the clock would be useless.

## Entry #5: Attention Matrix Shapes and Softmax Scaling
**Timestamp:** 2026-07-12

### Summary
Evaluating the dimensions of the attention weight matrix ($A_h$) and explaining how the $\sqrt{d_{head}}$ scaling factor prevents vanishing gradients in the Softmax function.

### Issue
Clarifying the difference between token feature shapes ($[S, d_{head}]$) and attention relationship shapes ($[S, S]$), and understanding the impact of large dot product variances on backpropagation.

### Approach
1. **Attention Shape and Mask**: The attention weights matrix $A_h$ represents the token-to-token relationship, so its shape is $[S, S]$ (which is $[2, 2]$ when $S=2$). The causal mask ensures that the index $[0, 1]$ (token 0 attending to token 1 in the future) has a post-softmax probability of exactly $0.0$.
2. **Softmax Scaling**: For independent random query and key vectors of dimension $d_{head}$, the variance of their dot product is $d_{head}$. For large dimensions, the dot products grow very large in magnitude, causing Softmax to saturate (converging to near-0 or near-1). In these flat regions, the gradients become extremely close to zero, leading to vanishing gradients (勾配消失). Dividing by $\sqrt{d_{head}}$ normalizes the variance to $1.0$, keeping gradients active.

### Example or Analogy
- **Attention shape is like a classmate relationship grid**: If there are $S$ students in a class, a grid showing who pays attention to whom is of size $S \times S$. The feature size $d_{head}$ (e.g., how many hobbies each student has) does not affect the size of this relationship grid.
- **Softmax scaling is like using sunglasses in bright light**: If the light intensity ($d_{head}$) increases, everything becomes overexposed (saturated). Scaling behaves like sunglasses, bringing the intensity back to a range where you can see colors and details clearly (gradients remain active).

## Entry #6: Residual Connections and LM Head Logit Dimensions
**Timestamp:** 2026-07-25

### Summary
Evaluation of Residual Connections (残差接続) in Transformer Blocks and tensor shape transformations leading to the Language Modeling Head (LM Head) projection.

### Issue
Clarifying why Residual Connections ($x + f(x)$) are essential within Transformer blocks, and understanding how hidden state dimensions ($[S, D]$) map to Vocabulary Logits ($[S, V]$).

### Approach
1. **Residual Connections**: While Causal Masking prevents future data leakage inside Attention, Residual Connections ($x1 = x + \text{attn\_out}$) prevent vanishing gradients across stacked blocks by creating an identity shortcut.
2. **LM Head Tensor Dimensions**:
   - Representation shape before LM Head: $[S, D]$ (e.g. $[4, 64]$). Stacking $N$ blocks preserves $[S, D]$.
   - LM Head Weight shape: $[D, V]$ (e.g. $[64, 256]$).
   - Logits Output shape: $[S, D] \times [D, V] \to [S, V]$ (e.g. $[4, 256]$).

### Example or Analogy
- **Residual Connection is like taking an highway alongside scenic detours**: The main highway ($x$) passes directly through to the end, while scenic loops ($\text{attn\_out}$) add local context. If a detour is blocked, the main highway still carries gradient signals directly to early layers.
- **LM Head Projection is like translating summary cards to dictionary terms**: A summary matrix of $S$ tokens represented by $D$ features ($[S, D]$) is multiplied by a dictionary matrix mapping $D$ features to $V$ vocabulary words ($[D, V]$), producing word probability scores of shape $[S, V]$.

## Entry #7: Untrained Model Decoding & Invalid UTF-8 Handling
**Timestamp:** 2026-07-26

### Summary
Explanation of why an untrained Transformer produces invalid UTF-8 byte outputs, and implementation of lossy decoding and logit masking solutions.

### Issue
Randomly initialized model weights produce random logit distributions across the 256 vocabulary IDs. When `argmax` selects token IDs in the byte range $128..255$, strict UTF-8 decoders fail because standalone continuation bytes do not form valid UTF-8 strings, resulting in `[INVALID UTF-8]`.

### Approach
1. **Lossy UTF-8 Decoding**: Update `ByteTokenizer::decode` to use `String::from_utf8_lossy(&bytes)` which replaces invalid byte sequences with unicode replacement characters () rather than breaking string creation.
2. **Printable ASCII Logit Masking**: Mask out non-printable ASCII IDs ($id < 32$ or $id > 126$) by setting their logits to $-\infty$ during evaluation so that argmax only picks printable characters.
3. **Model Training**: Acknowledge that random outputs are completely expected prior to weight updates via backpropagation.

### Example or Analogy
- **Untrained model decoding is like a cat walking across a typewriter**: The cat hits random keys, producing arbitrary byte sequences. A strict dictionary checker (strict UTF-8) marks the page as invalid gibberish, whereas a lossy reader (`from_utf8_lossy`) prints the visible letters and places a symbol over unreadable scuffs.
