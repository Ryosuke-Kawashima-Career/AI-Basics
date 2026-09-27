"""
This script fine-tunes an LLM for Ogiri (humorous replies) using QLoRA and TRL.
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # or your target model
MAX_SEQ_LENGTH = 1024
OUTPUT_DIR = "./LORA-SFT"

# 1. Quantization Configuration (4-bit QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 2. LoRA Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


def formatting_prompts_func(examples, tokenizer):
    texts = []
    for question, answer in zip(examples["question"], examples["answer"]):
        user_text = (
            "Below is the theme of funny talks. The task is not to answer correct ones but hilarious replies.\n"
            f"Theme: {question}"
        )
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(formatted_text)
    return {"text": texts}


class LoRAEvaluatorAndTrainer:
    def __init__(self, model, tokenizer, peft_config, train_dataset, valid_dataset, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset

        # Modern TRL uses SFTConfig instead of passing kwargs to SFTTrainer
        sft_config = SFTConfig(
            output_dir=OUTPUT_DIR,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            packing=False,  # Set False unless sequences are short & padded
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            num_train_epochs=3,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            max_grad_norm=0.3,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            report_to="none",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            args=sft_config,
        )

    def train(self):
        self.trainer.train()

    def evaluate(self, num_samples: int = 5) -> list[float]:
        """Interactive evaluation by generating model answers."""
        evaluations = []
        self.model.eval()

        print(f"\n--- Starting Interactive Evaluation ({num_samples} samples) ---")
        for i, row in enumerate(self.test_dataset):
            if i >= num_samples:
                break

            user_text = (
                "Below is the theme of funny talks. The task is not to answer correct ones but hilarious replies.\n"
                f"Theme: {row['question']}"
            )
            messages = [{"role": "user", "content": user_text}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Slice out only the generated tokens
            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            model_reply = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            print(f"\n[Question #{i+1}]: {row['question']}")
            print(f"[Reference Answer]: {row.get('answer', 'N/A')}")
            print(f"[Model Reply]: {model_reply}")

            score_str = input("Rate this joke (0-10, Enter defaults to 5): ").strip()
            try:
                score = float(score_str)
            except ValueError:
                score = 5.0
            evaluations.append(score)

        return evaluations

    def save(self, path=OUTPUT_DIR):
        self.trainer.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model and tokenizer saved to {path}")


def main():
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Model (loaded in 4-bit with BitsAndBytesConfig)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. Dataset
    raw_dataset = load_dataset(
        "csv",
        data_files="https://huggingface.co/datasets/watashihakobashi/ogiri/raw/main/ogiri.tsv",
        delimiter="\t",
        split="train",
    )

    # Train / Val / Test Split
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    test_val_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

    train_data = split_dataset["train"].map(lambda x: formatting_prompts_func(x, tokenizer), batched=True)
    valid_data = test_val_split["train"].map(lambda x: formatting_prompts_func(x, tokenizer), batched=True)
    test_data = test_val_split["test"]

    # 4. Training & Evaluation
    lora_pipeline = LoRAEvaluatorAndTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=lora_config,
        train_dataset=train_data,
        valid_dataset=valid_data,
        test_dataset=test_data,
    )

    lora_pipeline.train()
    lora_pipeline.evaluate(num_samples=5)
    lora_pipeline.save(OUTPUT_DIR)


if __name__ == "__main__":
    main()
