import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# 1. Reproducibility setup
torch.manual_seed(42)

# --- STEP 1: Define a Single Expert ---
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

# --- STEP 2: Define the Top-K Router & MoE Layer ---
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

# --- STEP 3: Inject MoE Layer into GPT-2 ---
def inject_moe_to_gpt2(model, num_experts=4, top_k=2):
    for i, block in enumerate(model.transformer.h):
        # Extract original FFN dimensions from GPT-2's Conv1D FFN
        d_model = block.mlp.c_fc.weight.shape[0]
        d_ff = block.mlp.c_fc.weight.shape[1]
        
        # Replace the MLP with our custom SparseMoELayer
        block.mlp = SparseMoELayer(d_model, d_ff, num_experts=num_experts, top_k=top_k)
    print(f"Successfully injected Sparse MoE Layers into all transformer blocks!")
    return model

# --- STEP 4: Main Training & Evaluation Script ---
if __name__ == "__main__":
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and inject model
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    moe_model = inject_moe_to_gpt2(base_model, num_experts=4, top_k=2)

    # Data Pipeline (1% of Wikitext for fast local run)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0).train_test_split(test_size=0.1)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=moe_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    # Train!
    print("Starting MoE training...")
    trainer.train()
    print("Training Complete!")
