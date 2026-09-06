import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import math, time, random
from dataclasses import dataclass
from typing import Tuple, List, Dict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
else (torch.float16 if torch.cuda.is_available() else torch.float32))

BLOCK_SIZE = 256
SEQ_LEN = BLOCK_SIZE
NUM_WORKERS = 0 if os.name == 'nt' else 2
SEED = 42

BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

MICRO_BS  = 8
GRAD_ACCUM = 4
GLOBAL_BATCH_TOKENS = MICRO_BS * SEQ_LEN * GRAD_ACCUM
print("[info] global_batch_tokens =", GLOBAL_BATCH_TOKENS)

BASE_LR   = 1e-3
REF_TOKENS= 8192
LR_BASE_SCALED = BASE_LR * (GLOBAL_BATCH_TOKENS / REF_TOKENS) ** 0.5

def build_llama(dim_hidden=128, layers=1, num_heads=4, dim_ffn=None, vocab_size=32000, seq_len=512) -> Tuple[nn.Module, int]:
    if dim_ffn is None:
        dim_ffn = int(4 * dim_hidden)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=dim_hidden,
        intermediate_size=dim_ffn,
        num_hidden_layers=layers,
        num_attention_heads=num_heads,
        max_position_embeddings=seq_len,
        rms_norm_eps=1e-9,
        rope_theta=1e6,
    )
    model = LlamaForCausalLM(cfg)
    return model, model.num_parameters()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor

def collate_fn(examples: List[Dict[str, List[int]]]) -> Batch:
    input_ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
    labels = torch.tensor([e.get("labels", e["input_ids"]) for e in examples], dtype=torch.long)
    return Batch(input_ids, labels)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, max_batches: int = 100) -> Tuple[float, float]:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    amp = (DTYPE != torch.float32)
    for i, batch in enumerate(loader):
        if i >= max_batches: break
        input_ids = batch.input_ids.to(DEVICE)
        labels = batch.labels.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=DTYPE) if amp and "cuda" in DEVICE else nullcontext():
            outputs = model(input_ids=input_ids, labels=labels)
            total_nll += outputs.loss.item() * labels.numel()
            total_tokens += labels.numel()
    model.train()
    avg_loss = total_nll / max(1, total_tokens)
    return avg_loss, math.exp(min(100, avg_loss))

def get_lr(step, max_steps, warmup_steps, base_lr):
    if step < warmup_steps: return base_lr * (step + 1) / max(1, warmup_steps)
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_ratio)))
    return base_lr * 0.1 + coeff * (base_lr * 0.9)

def main():
    # RESTART SESSION IF CUDA ERROR PERSISTS
    try:
        set_seed(SEED)
    except Exception as e:
        print(f"[critical] CUDA Error during seed: {e}. Please Restart Session!")
        return

    print(f"[info] Device: {DEVICE}, Dtype: {DTYPE}")
    
    raw = load_dataset("roneneldan/TinyStories", split="train").shuffle(seed=SEED)
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except:
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    actual_vocab_size = len(tokenizer)
    print(f"[info] Tokenizer vocab size: {actual_vocab_size}")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=SEQ_LEN, padding="max_length")

    sub_raw = raw.select(range(min(5000, len(raw))))
    tokenized = sub_raw.map(tokenize_fn, batched=True, remove_columns=sub_raw.column_names)
    split_idx = int(len(tokenized) * 0.9)
    train_data, val_data = tokenized.select(range(split_idx)), tokenized.select(range(split_idx, len(tokenized)))

    train_loader = DataLoader(train_data, batch_size=MICRO_BS, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=MICRO_BS, shuffle=False, collate_fn=collate_fn)

    model, params = build_llama(dim_hidden=128, layers=1, num_heads=4, vocab_size=actual_vocab_size, seq_len=SEQ_LEN)
    model.to(device=DEVICE, dtype=DTYPE)
    print(f"[info] Model parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_BASE_SCALED, betas=BETAS, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler() if DTYPE == torch.float16 and "cuda" in DEVICE else None

    total_steps = (len(train_loader) // GRAD_ACCUM)
    warmup_steps = int(total_steps * 0.05)
    
    step, accum_loss = 0, 0.0
    model.train()
    
    for micro_step, batch in enumerate(train_loader):
        input_ids, labels = batch.input_ids.to(DEVICE), batch.labels.to(DEVICE)
        
        # Safety check for vocab bounds
        if input_ids.max() >= actual_vocab_size:
            print(f"[error] Index out of bounds: {input_ids.max()} >= {actual_vocab_size}")
            break

        with torch.autocast(device_type="cuda", dtype=DTYPE) if "cuda" in DEVICE else nullcontext():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
        
        if scaler: scaler.scale(loss).backward()
        else: loss.backward()
        accum_loss += loss.item() * GRAD_ACCUM

        if (micro_step + 1) % GRAD_ACCUM == 0:
            if GRAD_CLIP > 0:
                if scaler: scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            lr = get_lr(step, total_steps, warmup_steps, LR_BASE_SCALED)
            for pg in optimizer.param_groups: pg['lr'] = lr
            
            if step % 10 == 0:
                print(f"Step {step}/{total_steps} | Loss: {accum_loss:.4f} | LR: {lr:.2e}")
            
            step += 1
            accum_loss = 0.0
            if step >= total_steps: break

if __name__ == '__main__':
    main()