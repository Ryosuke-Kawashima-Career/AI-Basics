from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
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
