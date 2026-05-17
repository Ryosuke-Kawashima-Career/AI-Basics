# This is the practice of week 5 of LLM advanced course
# Link to the environment: <https://drive.google.com/drive/folders/1aEV4r0ujiZ8J5CLjGxSXDSSikHvV2Fny?dmr=1&ec=wgc-drive-globalnav-goto>

# Library Import
import re
from typing import Dict

import MeCab
import torch
from datasets import load_dataset
from pynvml import *
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    AutoConfig,
    GenerationConfig,
)

# Model Definition

def model_create(model_dir: str, save_dir: str):
    # --- Model Initialization ---
    # config.json, tokenizer.model, tokenizer_config.json, special_tokens_map.json, generation_config.json should be included.

    # config.json for initialization 
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_config(config)
    print("model has been initialized")

    # tokenizer.model exists, use_fast=False, otherwise True
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    print("tokenizer has been loaded")

    # generation_config is loaded
    gen_config = GenerationConfig.from_pretrained(model_dir)
    print("GenerationConfig has been loaded")

    # Save model as Hugging Face format
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    gen_config.save_pretrained(save_dir)
    print(f"model and tokenizer has been saved in {save_dir}")

# Loading Model
model_name = "gpt-2"
"""
model_dir = "./Qwen3-30B-A3B"
save_dir = "./Qwen3_moe_small"
model_create(model_dir, save_dir)
"""
model = AutoModelForCausalLM.from_pretrained(model_name)
print("model has been loaded")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Data Preparation
## Data Preprocessing
def tokenizer_function(examples: Dict[str, str]) -> Dict[str, str]:
    # padding means the lengths of the training sentences the same to put them in batch.
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
""" For Japanese
tagger = MeCab.Tagger("o-Chasen")
def preprocessing(example: Dict[str, str]) -> Dict[str, str]:
    example["text"] = re.sub(r"[\s\t\n]", "", example["text"])  
    return example
"""
## Data Loading
original_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
print("original_dataset has been loaded")
tokenized_dataset = original_dataset.map(tokenizer_function, batched=True)
print("tokenized_dataset has been created")
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0).train_test_split(test_size=0.1)
test_dataset = tokenized_dataset["test"]
train_dataset = tokenized_dataset["train"]
print("tokenized_dataset has been filtered")
## Data Collator (Formatting) mlm = masking language model
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print("data_collator has been created")
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=5,
    save_steps=50,
    report_to='none',
)
print("training_args has been created")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)
print("trainer has been created")

# Training & Evaluating(Inference)
trainer.train()
print("training has been finished")
trainer.evaluate()
print("evaluation has been finished")

# Save model
trainer.save_model("./results/gpt-2_moe_small")
tokenizer.save_pretrained("./results/gpt-2_moe_small")
print("model has been saved in ./results/gpt-2_moe_small")

# Benchmark
benchmark_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[1%:2%]")
benchmark_data = benchmark_data.map(tokenizer_function, batched=True)
print("benchmark_data has been loaded")
model = AutoModelForCausalLM.from_pretrained("./results/gpt-2_moe_small")
print("model has been loaded")
with torch.no_grad():
    result = model(**benchmark_data)
    print(result.loss)
print("Bench marking has been finished")
