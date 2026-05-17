class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class SparseMoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        orig_shape = x.shape
        flat_x = x.view(-1, orig_shape[-1]) # (total_tokens, d_model)
        
        # TODO 1: Compute gating logits and apply softmax to get routing weights
        ## shape = (total_tokens, num_experts)
        logits = self.router(flat_x)
        weights = F.softmax(logits, dim=-1)
        
        # TODO 2: Get top-k weights and their expert indices using torch.topk
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
        
        # TODO 3: Normalize the top-k weights so they sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        out = torch.zeros_like(flat_x)
        
        # TODO 4: Route tokens to their selected experts and combine their outputs
        # Loop through each expert and gather tokens assigned to it
        for i, expert in enumerate(self.experts):
            # Find which tokens (rows in flat_x) had expert 'i' selected in their top-k
            mask = (topk_indices == i)
            token_indices, k_indices = torch.where(mask)
            
            if len(token_indices) > 0:
                # Get the matching input tokens for this expert
                expert_inputs = flat_x[token_indices]
                expert_outputs = expert(expert_inputs)
                
                # Multiply by the normalized gating weights and accumulate
                gating_w = topk_weights[token_indices, k_indices].unsqueeze(-1)
                out[token_indices] += gating_w * expert_outputs
                
        return out.view(orig_shape)
