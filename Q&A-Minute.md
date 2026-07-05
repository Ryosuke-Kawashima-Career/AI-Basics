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
