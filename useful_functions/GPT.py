import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MultiHeadAttention(nn.Module):
    """Causal Multi-Head Self-Attention for GPT"""
    def __init__(self, dim_model: int, num_heads: int, block_size: int):
        super().__init__()
        assert dim_model % num_heads == 0, "dim_model must be divisible by num_heads"
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.block_size = block_size

        self.q_proj = nn.Linear(dim_model, dim_model, bias=False)
        self.k_proj = nn.Linear(dim_model, dim_model, bias=False)
        self.v_proj = nn.Linear(dim_model, dim_model, bias=False)
        self.out_proj = nn.Linear(dim_model, dim_model, bias=False)

        # Causal mask: lower triangular matrix of shape (block_size, block_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Shape: (B: Batch, T: Sequence Length, C: dim_model)
        nh, hs = self.num_heads, self.head_dim

        # 1. Linear Projections & Reshape into Multi-Heads:
        # (B, T, C) -> Linear -> (B, T, C) -> View -> (B, T, nh, hs) -> Transpose -> (B, nh, T, hs)
        q = self.q_proj(x).view(B, T, nh, hs).transpose(1, 2)  # Shape: (B, nh, T, hs)
        k = self.k_proj(x).view(B, T, nh, hs).transpose(1, 2)  # Shape: (B, nh, T, hs)
        v = self.v_proj(x).view(B, T, nh, hs).transpose(1, 2)  # Shape: (B, nh, T, hs)

        # 2. Scaled Dot-Product Attention Scores:
        # (B, nh, T, hs) @ (B, nh, hs, T) -> Shape: (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        
        # 3. Apply Causal Masking (Mask future tokens with -inf before Softmax):
        # att shape remains: (B, nh, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)                           # Shape: (B, nh, T, T)

        # 4. Weighted Aggregation of Value Vectors:
        # (B, nh, T, T) @ (B, nh, T, hs) -> Shape: (B, nh, T, hs)
        y = att @ v
        
        # 5. Concatenate Heads Side-by-Side:
        # (B, nh, T, hs) -> Transpose -> (B, T, nh, hs) -> Contiguous View -> Shape: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 6. Final Linear Output Projection:
        # Shape: (B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    """GPT Transformer Block using Pre-LayerNorm Architecture:
    x (B, T, C) -> LayerNorm -> Attention -> Add -> LayerNorm -> MLP -> Add -> Output (B, T, C)
    """
    def __init__(self, dim_model: int, num_heads: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim_model)
        self.attn = MultiHeadAttention(dim_model, num_heads=num_heads, block_size=block_size)
        self.ln2 = nn.LayerNorm(dim_model)
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, 4 * dim_model),  # Expand: (B, T, C) -> (B, T, 4C)
            nn.GELU(),
            nn.Linear(4 * dim_model, dim_model),  # Project back: (B, T, 4C) -> (B, T, C)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C)
        x = x + self.attn(self.ln1(x))  # Residual Connection 1: (B, T, C)
        x = x + self.mlp(self.ln2(x))   # Residual Connection 2: (B, T, C)
        return x


class GPT(nn.Module):
    """Decoder-Only GPT Model"""
    def __init__(self, vocab_size: int, dim_model: int, num_heads: int, num_layers: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        
        # Token and Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim_model)
        self.positional_embedding = nn.Embedding(block_size, dim_model)
        
        # Stacked Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim_model, num_heads, block_size) for _ in range(num_layers)
        ])
        
        # Final LayerNorm and Language Model Head
        self.ln_f = nn.LayerNorm(dim_model)
        self.head = nn.Linear(dim_model, vocab_size, bias=False)

        # Weight tying (standard in GPT)
        self.token_embedding.weight = self.head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape  # Input indices shape: (B: Batch, T: Sequence Length)
        assert T <= self.block_size, f"Cannot forward sequence length {T}, exceeds block_size {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # Shape: (T,)
        
        tok_emb = self.token_embedding(idx)       # Token Embeddings:     Shape (B, T, C)
        pos_emb = self.positional_embedding(pos)   # Positional Embeddings: Shape (T, C)
        
        x = tok_emb + pos_emb                     # Combined Input Embeddings: Shape (B, T, C)

        for block in self.blocks:
            x = block(x)                          # Passes through each Block: Shape (B, T, C)

        x = self.ln_f(x)                          # Final LayerNorm: Shape (B, T, C)
        logits = self.head(x)                      # Language Model Head: Shape (B, T, V)

        loss = None
        if targets is not None:
            B_log, T_log, V_log = logits.shape
            # Use .reshape() or .contiguous().view() to safely handle non-contiguous sliced tensors
            logits_flat = logits.reshape(B_log * T_log, V_log)  # Flattened Logits:  Shape (B*T, V)
            targets_flat = targets.reshape(B_log * T_log)       # Flattened Targets: Shape (B*T,)
            loss = F.cross_entropy(logits_flat, targets_flat)   # Scalar Cross-Entropy Loss: ()

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively generate new tokens given prompt indices"""
        for _ in range(max_new_tokens):
            # Crop prompt to max context size: (B, T_current) -> (B, min(T_current, block_size))
            idx_cond = idx[:, -self.block_size:]
            
            # Forward pass: logits shape -> (B, T_cond, V)
            logits, _ = self(idx_cond)
            
            # Pluck logits at final time step: Shape (B, V)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)                  # Probabilities: Shape (B, V)
            
            # Sample next token from multinomial distribution: Shape (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append next token to sequence: (B, T_current) + (B, 1) -> Shape (B, T_current + 1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


class Dataloader:
    """Synthetic DataLoader producing autoregressive shift target sequences"""
    def __init__(self, vocab_size=50, block_size=16, batch_size=14):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.batch_size = batch_size

    def get_batch(self):
        # Generate random integer tokens: Shape (Batch, block_size + 1)
        data = torch.randint(0, self.vocab_size, (self.batch_size, self.block_size + 1))
        x = data[:, :-1]  # Input sequence X:  Shape (Batch, block_size)
        y = data[:, 1:]   # Target sequence Y: Shape (Batch, block_size) (Shifted by 1)
        return x, y


class Trainer:
    def __init__(self, model, dataloader, optimizer, steps):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.steps = steps
        self.metrics = {'loss': []}

    def train(self):
        self.model.train()
        for step in range(self.steps):
            x, y = self.dataloader.get_batch()
            self.optimizer.zero_grad(set_to_none=True)
            logits, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()

            if step % 10 == 0 or step == self.steps - 1:
                print(f"Step {step:03d} | Loss: {loss.item():.4f}")
                self.metrics['loss'].append(loss.item())

        return self.metrics

    def plot(self):
        plt.figure(figsize=(7, 4))
        plt.plot(self.metrics['loss'], marker='o')
        plt.xlabel("Evaluation Step (x10)")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("GPT Training Loss Curve")
        plt.grid(True)
        plt.show()


def main():
    torch.manual_seed(42)
    VOCAB_SIZE = 50
    D_MODEL = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    BLOCK_SIZE = 16
    BATCH_SIZE = 8
    STEPS = 100

    print("Initializing GPT Model...")
    model = GPT(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE)
    dataloader = Dataloader(VOCAB_SIZE, BLOCK_SIZE, BATCH_SIZE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    trainer = Trainer(model, dataloader, optimizer, steps=STEPS)
    print("Starting Training...")
    trainer.train()
    
    # Test text generation
    print("\nTesting Autoregressively Generating 10 Tokens...")
    context = torch.zeros((1, 1), dtype=torch.long)  # Start token [0] shape: (1, 1)
    generated = model.generate(context, max_new_tokens=10)
    print(f"Generated token sequence: {generated.tolist()[0]}")

if __name__ == '__main__':
    main()
