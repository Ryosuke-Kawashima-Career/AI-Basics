"""Build a self-contained Colab notebook from the current GRPO implementation.

Run: python ReinforcementLearning/build_colab_notebook.py
No notebook, GPU, or training dependencies are required to build it.
"""
import ast
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent


def build_notebook():
    cells = []

    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": textwrap.dedent(source).strip() + "\n"})

    def code(source):
        source = textwrap.dedent(source).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": source})

    markdown('''
        # Train Qwen3 with GRPO on a Google Colab GPU

        This self-contained notebook trains LoRA adapters for **Qwen/Qwen3-0.6B**
        using GSM8K and a verified final-integer reward. No repository clone is needed.

        1. Select **Runtime → Change runtime type → GPU** (for example, a T4 if offered).
        2. Run cells in order. The default is a small smoke run; switch `SMOKE_TEST`
           to `False` for the longer learning experiment.
        3. Inspect results, reload the adapter, and download its ZIP. Hugging Face
           publication and Google Drive backup are optional and disabled initially.

        Colab GPU availability and session duration vary. `/content` storage is
        temporary: download or back up results before the runtime is deleted.
        A few training steps demonstrate the machinery, not improved reasoning.
        This notebook has local structural checks and shares the CPU-tested training
        implementation; end-to-end execution on a Colab GPU has not been verified.
    ''')
    markdown('''
        ## 1. Install the training stack

        Keep Colab's existing CUDA-enabled PyTorch. The pinned TRL, Transformers and
        PEFT versions match this implementation. Training and model loading run in
        fresh Python subprocesses so an older library imported in the notebook
        kernel does not need to be reused after installation.
    ''')
    requirements = (HERE / 'requirements-grpo.txt').read_text(encoding='utf-8-sig')
    code(textwrap.dedent('''
        import os
        from pathlib import Path
        import subprocess
        import sys

        WORK_DIR = Path("/content/grpo_qwen")
        WORK_DIR.mkdir(parents=True, exist_ok=True)
    ''') + '\nREQUIREMENTS = ' + repr(requirements) + '''
requirements_path = WORK_DIR / "requirements-grpo.txt"
requirements_path.write_text(REQUIREMENTS, encoding="utf-8")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)], check=True)
print("Training dependencies installed. Continue to the GPU check.")
''')
    markdown('''
        ## 2. Verify that the runtime has a CUDA GPU

        This intentionally stops if Colab is still using a CPU runtime. A T4 uses
        fp16 here; GPUs with bf16 support use bf16. Both modes train LoRA weights.
    ''')
    code('''
        gpu_check = """
        import torch
        import transformers, trl, peft
        assert torch.cuda.is_available(), "Select Runtime > Change runtime type > GPU, then rerun."
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
        print("Precision:", "bf16" if torch.cuda.is_bf16_supported() else "fp16")
        print("Versions:", torch.__version__, transformers.__version__, trl.__version__, peft.__version__)
        """
        import textwrap
        subprocess.run([sys.executable, "-c", textwrap.dedent(gpu_check)], check=True)
    ''')
    markdown('''
        ## 3. Write the training implementation

        The next cell embeds the complete project script, including answer parsing,
        model loading, GRPO optimization, before/after evaluation and adapter saving.
        Expand it to study the implementation. The demo checks reward arithmetic
        without downloading the model or dataset.
    ''')
    source = (HERE / 'GRPO.py').read_text(encoding='utf-8-sig')
    assert "'''" not in source, 'Choose another string delimiter for the embedded script'
    code("GRPO_SOURCE = r'''" + source.rstrip() + "\n'''\n"
         'SCRIPT = WORK_DIR / "GRPO.py"\n'
         'SCRIPT.write_text(GRPO_SOURCE, encoding="utf-8")\n'
         'subprocess.run([sys.executable, str(SCRIPT), "--demo"], check=True)\n')
    markdown('''
        ## 4. Choose training settings

        Four completions are sampled for each question. Microbatch 1 × accumulation 4
        gives a generation batch of four completions on this single-GPU runtime.
        Original GRPO uses group-relative rewards and a KL penalty to the initial model.

        Start with `SMOKE_TEST=True`. After it succeeds, set it to `False` and rerun
        this cell and the cells below for a 25-step experiment in a different folder.
        Short smoke completions often stop before the final answer. Increase the
        completion limit for learning experiments if many responses are truncated.
    ''')
    code('''
        SMOKE_TEST = True
        MODEL_NAME = "Qwen/Qwen3-0.6B"
        MAX_STEPS = 2 if SMOKE_TEST else 25
        TRAIN_SIZE = 8 if SMOKE_TEST else 256
        EVAL_SIZE = 2 if SMOKE_TEST else 16
        MAX_PROMPT_LENGTH = 512
        MAX_COMPLETION_LENGTH = 64 if SMOKE_TEST else 256
        NUM_GENERATIONS = 4
        MICRO_BATCH_SIZE = 1
        GRADIENT_ACCUMULATION_STEPS = 4
        LORA_RANK = 16
        BETA = 0.04
        SEED = 42
        OUTPUT_DIR = WORK_DIR / ("qwen3_grpo_smoke" if SMOKE_TEST else "qwen3_grpo")
        ADAPTER_DIR = OUTPUT_DIR / "adapter"

        assert NUM_GENERATIONS >= 2
        assert (MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) % NUM_GENERATIONS == 0
        run_settings = {
            "model": MODEL_NAME, "device": "cuda", "backend": "transformers",
            "max_steps": MAX_STEPS, "train_size": TRAIN_SIZE, "eval_size": EVAL_SIZE,
            "max_prompt_length": MAX_PROMPT_LENGTH, "max_completion_length": MAX_COMPLETION_LENGTH,
            "num_generations": NUM_GENERATIONS, "per_device_train_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "lora_rank": LORA_RANK, "beta": BETA, "seed": SEED, "num_iterations": 1,
        }
        command = [sys.executable, "-u", str(SCRIPT)]
        for key, value in run_settings.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        command.extend(["--output-dir", str(OUTPUT_DIR)])
        print("Mode:", "smoke test" if SMOKE_TEST else "learning experiment")
        print("Adapter will be saved to:", ADAPTER_DIR)
    ''')
    markdown('''
        ## 5. Train and evaluate

        The first run downloads Qwen and GSM8K. Logs are streamed below. Evaluation
        uses the same held-out subset before and after training and scores only new
        response tokens. The adapter is saved before the final evaluation starts.

        If you get CUDA out-of-memory, reduce the completion limit first, keep
        microbatch 1, and try two generations with accumulation 2. If the process
        was interrupted or lost GPU memory, restart the runtime before retrying.
        This notebook starts a new training run; it does not automatically resume
        an interrupted checkpoint. Reusing an output folder can overwrite results.
    ''')
    code('''
        import json
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "run_settings.json").write_text(json.dumps(run_settings, indent=2), encoding="utf-8")
        child_environment = os.environ.copy()
        child_environment["TOKENIZERS_PARALLELISM"] = "false"
        subprocess.run(command, check=True, env=child_environment, cwd=str(WORK_DIR))
        assert (ADAPTER_DIR / "adapter_config.json").is_file()
        assert (ADAPTER_DIR / "adapter_model.safetensors").is_file()
        print("Training completed. Adapter:", ADAPTER_DIR)
    ''')
    markdown('''
        ## 6. Inspect results and prepare a model card

        Accuracy is exact final-integer correctness under the boxed-answer contract.
        `format_rate` measures whether a valid boxed integer was generated.
        A correct final number does not verify intermediate reasoning. Small test
        subsets are noisy; avoid repeatedly tuning on the test split.
    ''')
    code('''
        import csv
        summary = {"settings": run_settings}
        for stage in ("before", "after"):
            record = json.loads((OUTPUT_DIR / f"evaluation_{stage}.json").read_text())
            summary[stage] = record["metrics"]
            print(stage, record["metrics"])
        print("\\nExample response after training:")
        print(record["predictions"][0]["completion"])
        with (OUTPUT_DIR / "rewards.csv").open() as handle:
            reward_rows = list(csv.DictReader(handle))
        print("\\nLast reward records:", reward_rows[-5:])
        (ADAPTER_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        card = f"""---
        base_model: {MODEL_NAME}
        library_name: peft
        license: apache-2.0
        datasets:
        - openai/gsm8k
        tags:
        - grpo
        - lora
        - reinforcement-learning
        ---
        # Qwen3 GSM8K GRPO adapter

        A LoRA adapter for `{MODEL_NAME}`, trained on a shuffled GSM8K training subset
        with a verifiable boxed-integer outcome reward. The original base weights
        are required to load this adapter. See `training_summary.json` for settings
        and before/after results, including sample counts.

        This is an educational {"smoke run" if SMOKE_TEST else "training experiment"}.
        Results do not establish general reasoning improvements. Exact-answer scoring
        does not verify intermediate reasoning, and the held-out subset is small.
        The base model is provided by Qwen under Apache-2.0.
        """
        (ADAPTER_DIR / "README.md").write_text(textwrap.dedent(card), encoding="utf-8")
        print("\\nFiles ready to share:", sorted(path.name for path in ADAPTER_DIR.iterdir()))
    ''')
    markdown('''
        ## 7. Reload the saved adapter and generate an answer

        This fresh process checks that the adapter works with its original base
        model. The training process has exited, so its GPU model memory is released.
    ''')
    code('''
        inference_code = r"""
        import sys
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        adapter = sys.argv[1]
        tokenizer = AutoTokenizer.from_pretrained(adapter)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoPeftModelForCausalLM.from_pretrained(adapter, dtype=dtype).to("cuda")
        model.eval()
        messages = [{"role": "user", "content":
            r"A box holds 6 apples. How many apples are in 3 boxes? Reason step by step and end in \\boxed{...}."}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                           return_token_type_ids=False).to("cuda")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        """
        subprocess.run([sys.executable, "-c", textwrap.dedent(inference_code), str(ADAPTER_DIR)], check=True)
    ''')
    markdown('''
        ## 8. Download the adapter ZIP

        The ZIP contains adapter weights, configuration, tokenizer, model card and
        training summary. It contains no full base-model weights. Download it before
        ending the Colab session. Extract it before uploading the folder to the Hub.
    ''')
    code('''
        import shutil
        from google.colab import files
        archive = shutil.make_archive(str(WORK_DIR / (OUTPUT_DIR.name + "_adapter")),
                                      "zip", root_dir=str(ADAPTER_DIR))
        print("Download:", archive)
        files.download(archive)
    ''')
    markdown('''
        ## 9. Optional: upload the adapter to Hugging Face

        Set `UPLOAD_TO_HUB=True` and replace the repository ID. This explicitly creates
        a **private** repository if it does not exist and uploads only the adapter
        folder. Existing repository visibility is not changed. Enter a write-capable
        token in the hidden prompt; never paste it into a code cell. Review the card
        and results before changing the repository to public.
    ''')
    code('''
        UPLOAD_TO_HUB = False
        REPO_ID = "YOUR_USERNAME/qwen3-0.6b-grpo-gsm8k"
        if UPLOAD_TO_HUB:
            assert "YOUR_USERNAME" not in REPO_ID and REPO_ID.count("/") == 1
            import getpass
            from huggingface_hub import HfApi
            token = getpass.getpass("Hugging Face write token: ")
            api = HfApi(token=token)
            try:
                api.create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=True)
                api.upload_folder(repo_id=REPO_ID, repo_type="model", folder_path=str(ADAPTER_DIR),
                                  commit_message="Upload Qwen3 GSM8K GRPO LoRA adapter")
                print("Uploaded:", "https://huggingface.co/" + REPO_ID)
            finally:
                del api, token
        else:
            print("Upload disabled. The local adapter and ZIP are ready.")
    ''')
    markdown('''
        ## 10. Optional: back up the full run to Google Drive

        Enable this to preserve checkpoints, evaluation predictions and logs as well
        as the adapter. It mounts your Drive interactively and writes a ZIP to
        `MyDrive/GRPO_backups/`. Downloads and Drive backup are independent of Hub upload.
    ''')
    code('''
        BACKUP_TO_DRIVE = False
        if BACKUP_TO_DRIVE:
            from datetime import datetime, timezone
            from google.colab import drive
            drive.mount("/content/drive")
            backup_dir = Path("/content/drive/MyDrive/GRPO_backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
            backup = shutil.make_archive(str(WORK_DIR / (OUTPUT_DIR.name + "_" + timestamp)),
                                         "zip", root_dir=str(OUTPUT_DIR))
            destination = shutil.copy2(backup, backup_dir)
            print("Full-run backup:", destination)
        else:
            print("Drive backup disabled.")
    ''')
    markdown('''
        ## Learn from this run

        - For rewards `[1, 0, 0, 1]`, the sample-standard-deviation normalized
          advantages are about `[+0.866, -0.866, -0.866, +0.866]`.
        - If all completions for a question get the same reward, their task-reward
          advantages are zero. Check outputs and completion limits before adding steps.
        - Compare final-answer accuracy and valid-answer rate separately. A format
          improvement is useful, but does not by itself demonstrate better arithmetic.

        References: [Colab resource limits](https://research.google.com/colaboratory/faq.html),
        [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B),
        [versioned TRL implementation](https://github.com/huggingface/trl/tree/v0.22.2),
        [Hugging Face uploads](https://huggingface.co/docs/huggingface_hub/guides/upload).
    ''')
    for index, cell in enumerate(cells):
        cell['id'] = f'grpo-cell-{index:02d}'
        cell['source'] = cell['source'].splitlines(keepends=True)
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    notebook = build_notebook()
    destination = HERE / 'GRPO_Colab_GPU.ipynb'
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Created {destination.name}: {len(notebook["cells"])} cells')


if __name__ == '__main__':
    main()
