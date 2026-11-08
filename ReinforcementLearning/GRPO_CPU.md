# Train GRPO on a CPU and save an uploadable adapter

CPU mode uses the same Qwen3-0.6B model and integer-answer reward as GPU mode. It uses float32 and Transformers/PEFT, with smaller batches, shorter completions and rank-8 LoRA adapters. Unsloth is unavailable in this mode.

## 1. Create a separate CPU environment

Run from `D:\Training\AI-Basics` in PowerShell:

```powershell
uv venv --python 3.11 .venv-grpo-cpu
uv pip install --python .venv-grpo-cpu/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-grpo-cpu/Scripts/python.exe -r ReinforcementLearning/requirements-grpo.txt
```

This leaves the existing project environment and `uv.lock` unchanged. The CPU PyTorch wheel does not need CUDA. The training package versions are pinned in the requirements file.

## 2. Start with a smoke run

```powershell
.venv-grpo-cpu\Scripts\python.exe ReinforcementLearning\GRPO.py --device cpu --max-steps 1 --train-size 4 --eval-size 1 --max-completion-length 32
```

The first run downloads Qwen and GSM8K. Generation and backpropagation both run on the CPU. Even this small run can take time; the output folder is printed at startup. A short completion limit checks execution but often cuts off solutions before their final boxed answers.

## 3. Run the CPU learning example

```powershell
.venv-grpo-cpu\Scripts\python.exe ReinforcementLearning\GRPO.py --device cpu
```

The defaults selected by `--device cpu` are:

| Setting | CPU default |
| --- | --- |
| Model | Qwen/Qwen3-0.6B |
| Precision | float32; fp16 and bf16 disabled |
| Training / evaluation questions | 16 / 2 |
| Optimizer steps | 5 |
| Completions per question | 2 |
| Training microbatch / gradient accumulation | 1 / 2 |
| Prompt / completion token limits | 512 / 128 |
| Updates per rollout batch | 1 |
| LoRA rank / alpha | 8 / 16 |
| CPU threads | Up to 4 |
| Output directory | GRPO_output_cpu |

These defaults demonstrate the workflow; five steps and two evaluation questions cannot establish improved reasoning. Generating multiple answers is the expensive part of GRPO, and LoRA does not eliminate the cost of running the frozen base model. Qwen's roughly 0.6 billion float32 parameters alone occupy about 2.4 GB; activations, temporary tensors and other state require additional RAM.

For a larger experiment after the smoke run works:

```powershell
.venv-grpo-cpu\Scripts\python.exe ReinforcementLearning\GRPO.py --device cpu --max-steps 20 --train-size 64 --eval-size 8 --max-completion-length 256 --cpu-threads 4 --output-dir GRPO_output_cpu_longer
```

Longer answers can reach the final box more often but increase compute and memory. If increasing `--num-generations`, increase `--gradient-accumulation-steps` so `microbatch * accumulation * process_count` remains divisible by the generation count. For example, four generations with microbatch one require accumulation four on one CPU process. GPU defaults remain available with `--device cuda`.

## 4. Find and upload the adapter

For the default run, the uploadable folder is:

```text
D:\Training\AI-Basics\GRPO_output_cpu\adapter
```

It contains the adapter weights, adapter configuration and tokenizer. Before/after evaluation JSON files and training logs live in the parent output directory. Add a model card describing the base model, training settings and actual results before sharing it.

Using the Hugging Face CLI environment from the upload plan:

```powershell
.venv-hf\Scripts\hf.exe auth login
.venv-hf\Scripts\hf.exe repos create YOUR_USERNAME/qwen3-0.6b-grpo-gsm8k-cpu --private
.venv-hf\Scripts\hf.exe upload YOUR_USERNAME/qwen3-0.6b-grpo-gsm8k-cpu .\GRPO_output_cpu\adapter .
```

Replace `YOUR_USERNAME` with your account name. An adapter trained on CPU can be loaded on a GPU later: the hardware used for training does not change the PEFT artifact format. Use the same base model named in `adapter_config.json`.

## Local verification

```powershell
.venv-grpo-cpu\Scripts\python.exe -m unittest discover -s ReinforcementLearning -p 'test_*.py' -v
```

The optional integration test creates a tiny random Qwen3 model locally, uses synthetic rewards to exercise a real optimizer update, checks that LoRA weights change, saves the adapter and reloads it. This tests CPU execution and serialization without downloading pretrained weights. It does not measure Qwen/GSM8K accuracy.
