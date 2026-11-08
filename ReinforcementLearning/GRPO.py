"""Train Qwen on GSM8K with GRPO and verifiable integer-answer rewards.

See GRPO_practice.md for setup, the objective, and worked exercises.
Imports are lazy: --help, --demo, and reward tests need only Python.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, stdev

INSTRUCTION = r"Solve the problem step by step. End with one final integer answer in \boxed{...}."
INTEGER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"


def extract_hash_answer(text):
    """Parse GSM8K's final #### answer; reject malformed labels early."""
    match = re.search(r"####\s*(" + INTEGER + r")\s*$", text)
    if match is None:
        raise ValueError(f"Invalid GSM8K answer: {text!r}")
    return int(match.group(1).replace(",", ""))


def extract_boxed_int(text):
    """Use the last box; decimals, expressions and malformed commas earn zero."""
    # Include nested/invalid box starts so an earlier valid box cannot hide one.
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if not starts:
        return None
    tail = text[starts[-1].end():]
    raw, separator, _ = tail.partition("}")
    if not separator or re.fullmatch(INTEGER, raw.strip()) is None:
        return None
    return int(raw.strip().replace(",", ""))


def completion_text(completion):
    """TRL supports plain text and conversational assistant messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(message["content"] for message in completion
                         if message.get("role") == "assistant")
    raise TypeError("Expected text or a list of chat messages")


def reward_func(completions, answer, **kwargs):
    """One outcome reward per sampled completion; labels never enter prompts."""
    if len(completions) != len(answer):
        raise ValueError("Each completion must have one ground-truth answer")
    rewards = []
    for completion, target in zip(completions, answer):
        predicted = extract_boxed_int(completion_text(completion))
        rewards.append(float(predicted is not None and predicted == int(target)))
    return rewards


def group_advantages(rewards, epsilon=1e-4):
    """Teaching helper matching TRL's sample-std group normalization.

    TRL performs the actual tensor calculation and gradient updates.
    Equal-reward groups have zero task-reward learning signal.
    """
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two completions per prompt")
    baseline, spread = mean(rewards), stdev(rewards)
    return [(reward - baseline) / (spread + epsilon) for reward in rewards]


def prepare_example(example, tokenizer):
    messages = [{"role": "user", "content": example["question"] + "\n" + INSTRUCTION}]
    # Render once so training and evaluation use the same non-thinking template.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return {"prompt": prompt, "answer": extract_hash_answer(example["answer"])}


def load_dataset(tokenizer, train_size=256, eval_size=32, seed=42):
    from datasets import load_dataset as hf_load_dataset

    splits = []
    for split, limit in (("train", train_size), ("test", eval_size)):
        dataset = hf_load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=seed)
        dataset = dataset.select(range(min(limit, len(dataset))))
        dataset = dataset.map(lambda row: prepare_example(row, tokenizer),
                              remove_columns=dataset.column_names)
        splits.append(dataset)
    return tuple(splits)


def load_library(backend="transformers", device="cuda"):
    """Load Unsloth first when requested so its patches precede TRL imports."""
    if device == "cpu" and backend == "unsloth":
        raise ValueError("CPU training uses --backend transformers; Unsloth requires a GPU.")
    try:
        if backend == "unsloth":
            import unsloth  # noqa: F401
        import torch
        from trl import GRPOConfig, GRPOTrainer
        import peft  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Missing training dependencies; see GRPO_practice.md setup.") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for CPU training.")
    return torch, GRPOConfig, GRPOTrainer


def load_model(model_name, backend="transformers", max_seq_length=1536, seed=42,
               device="cuda", lora_rank=32):
    import torch

    if backend == "unsloth":
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=max_seq_length,
            load_in_4bit=False, full_finetuning=False,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=lora_rank, lora_alpha=2 * lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=seed,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=(torch.float32 if device == "cpu" else
                                     torch.bfloat16 if torch.cuda.is_bf16_supported()
                                     else torch.float16),
        )
        model = get_peft_model(model, LoraConfig(
            r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


class GRPO:
    """Coordinate rollouts, rewards, optimization and held-out evaluation."""

    reward_func = staticmethod(reward_func)

    def __init__(self, options):
        self.options = options
        torch, config_class, trainer_class = load_library(options.backend, options.device)
        from transformers import set_seed
        set_seed(options.seed)
        if options.device == "cpu":
            torch.set_num_threads(options.cpu_threads)
        self.model, self.tokenizer = load_model(
            options.model, options.backend,
            options.max_prompt_length + options.max_completion_length, options.seed,
            device=options.device, lora_rank=options.lora_rank,
        )
        self.ds_train, self.ds_test = load_dataset(
            self.tokenizer, options.train_size, options.eval_size, options.seed,
        )
        self.output_dir = Path(options.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cpu = options.device == "cpu"
        bf16 = not cpu and torch.cuda.is_bf16_supported()
        training_args = config_class(
            output_dir=str(self.output_dir), learning_rate=5e-6,
            adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1,
            # A one-step smoke run must not spend its only update at zero LR.
            warmup_ratio=0.0 if options.max_steps == 1 else 0.1,
            lr_scheduler_type="cosine", logging_steps=1,
            use_cpu=cpu, bf16=bf16, fp16=not cpu and not bf16,
            optim="adamw_torch", dataloader_pin_memory=not cpu,
            per_device_train_batch_size=options.per_device_train_batch_size,
            gradient_accumulation_steps=options.gradient_accumulation_steps,
            num_generations=options.num_generations,
            max_prompt_length=options.max_prompt_length,
            max_completion_length=options.max_completion_length,
            max_steps=options.max_steps, save_steps=options.max_steps,
            report_to="none", use_vllm=False, remove_unused_columns=False,
            temperature=1.0, top_p=0.95, seed=options.seed,
            # Explicit choices teach original GRPO instead of changing TRL defaults.
            beta=options.beta, epsilon=0.2, loss_type="grpo", scale_rewards="group",
            num_iterations=options.num_iterations, gradient_checkpointing=True,
            # TRL 0.22 re-enables reentrant checkpointing after generation;
            # use that mode consistently so it also enables input gradients.
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
        self.trainer = trainer_class(
            model=self.model, processing_class=self.tokenizer,
            reward_funcs=[reward_func], args=training_args, train_dataset=self.ds_train,
        )

    def training(self):
        """TRL handles old-policy log-probs, clipping, reference KL and backprop."""
        result = self.trainer.train()
        self.trainer.save_model(str(self.output_dir / "adapter"))
        self.tokenizer.save_pretrained(str(self.output_dir / "adapter"))
        self.trainer.save_state()
        history = self.trainer.state.log_history
        (self.output_dir / "log_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8",
        )
        rows = [entry for entry in history if "reward" in entry]
        with (self.output_dir / "rewards.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "reward", "moving_average_5"])
            for index, row in enumerate(rows):
                window = rows[max(0, index - 4):index + 1]
                writer.writerow([row["step"], row["reward"], mean(x["reward"] for x in window)])
        return result

    def evaluation(self, name="after"):
        """Greedy exact-match accuracy on the same held-out subset before/after."""
        import torch
        if self.options.backend == "unsloth":
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(self.model)
        was_training = self.model.training
        self.model.eval()
        rows = []
        try:
            with torch.inference_mode():
                for example in self.ds_test:
                    inputs = self.tokenizer(
                        example["prompt"], return_tensors="pt", truncation=True,
                        max_length=self.options.max_prompt_length, add_special_tokens=False,
                        return_token_type_ids=False,
                    ).to(next(self.model.parameters()).device)
                    output = self.model.generate(
                        **inputs, max_new_tokens=self.options.max_completion_length,
                        do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id, use_cache=True,
                    )
                    # Score only new tokens; prompt text must never supply the answer.
                    text = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)
                    predicted = extract_boxed_int(text)
                    rows.append({"answer": example["answer"], "prediction": predicted,
                                 "correct": predicted == example["answer"], "completion": text})
        finally:
            if self.options.backend == "unsloth":
                FastLanguageModel.for_training(self.model)
            self.model.train(was_training)
        metrics = {"accuracy": mean(row["correct"] for row in rows),
                   "format_rate": mean(row["prediction"] is not None for row in rows),
                   "examples": len(rows)}
        (self.output_dir / f"evaluation_{name}.json").write_text(
            json.dumps({"metrics": metrics, "predictions": rows}, indent=2), encoding="utf-8",
        )
        print(f"{name}: {metrics}")
        return metrics


def Huggingface_upload(experiment, repo_id):
    """Publish adapters only when --push-to-hub is explicitly supplied."""
    experiment.model.push_to_hub(repo_id)
    experiment.tokenizer.push_to_hub(repo_id)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Show group rewards without downloads")
    parser.add_argument("--backend", choices=["transformers", "unsloth"], default="transformers")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="CPU selects smaller defaults and float32 training")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-iterations", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-to-hub", metavar="USER/REPO")
    args = parser.parse_args(argv)
    defaults = {
        "output_dir": ("GRPO_output", "GRPO_output_cpu"),
        "train_size": (256, 16), "eval_size": (32, 2), "max_steps": (50, 5),
        "num_generations": (8, 2), "max_prompt_length": (1024, 512),
        "max_completion_length": (512, 128), "per_device_train_batch_size": (2, 1),
        "gradient_accumulation_steps": (4, 2), "num_iterations": (2, 1),
        "lora_rank": (32, 8),
    }
    for field, values in defaults.items():
        if getattr(args, field) is None:
            setattr(args, field, values[1 if args.device == "cpu" else 0])
    if args.device == "cpu" and args.backend == "unsloth":
        parser.error("--device cpu requires --backend transformers")
    for field in ("train_size", "eval_size", "max_steps", "max_prompt_length",
                  "max_completion_length", "per_device_train_batch_size",
                  "gradient_accumulation_steps", "num_iterations", "lora_rank", "cpu_threads"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    effective_batch = (args.per_device_train_batch_size * args.gradient_accumulation_steps
                       * int(os.environ.get("WORLD_SIZE", "1")))
    if args.num_generations < 2 or effective_batch % args.num_generations:
        parser.error(f"--num-generations must be >=2 and divide effective batch {effective_batch}")
    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error("--beta must be finite and nonnegative")
    return args


def main(argv=None):
    options = parse_args(argv)
    if options.demo:
        completions = [r"2 + 2 = 4. \boxed{4}", r"\boxed{5}", "4", r"\boxed{4}"]
        rewards = reward_func(completions, [4] * 4)
        print("Rewards:", rewards)
        print("Group-relative advantages:", [round(x, 4) for x in group_advantages(rewards)])
        print("Equal-reward group:", group_advantages([0.0] * 4))
        return
    print(f"Training on {options.device}: {options.max_steps} steps, "
          f"{options.num_generations} completions per question, "
          f"adapter output: {Path(options.output_dir) / 'adapter'}", flush=True)
    experiment = GRPO(options)
    experiment.evaluation("before")
    experiment.training()
    experiment.evaluation("after")
    if options.push_to_hub:
        Huggingface_upload(experiment, options.push_to_hub)


if __name__ == "__main__":
    main()
