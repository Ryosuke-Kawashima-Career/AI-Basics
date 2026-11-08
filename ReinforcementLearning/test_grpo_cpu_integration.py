"""Optional offline integration test using the installed GRPO training stack.

Builds a tiny random Qwen3 locally, takes a real CPU optimizer step, saves an
adapter, and reloads it. Synthetic rewards test the training machinery only.
"""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from GRPO import GRPO, parse_args

HAS_STACK = all(importlib.util.find_spec(name) is not None
                for name in ("torch", "transformers", "trl", "peft", "datasets"))


@unittest.skipUnless(HAS_STACK, "Optional GRPO training dependencies are not installed")
class CPUTrainingIntegrationTests(unittest.TestCase):
    def test_cpu_optimizer_step_adapter_save_and_reload(self):
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (AutoModelForCausalLM, PreTrainedTokenizerFast,
                                  Qwen3Config, Qwen3ForCausalLM)

        with tempfile.TemporaryDirectory(prefix="grpo-cpu-test-") as directory:
            root = Path(directory)
            base = root / "base"
            vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
            vocab.update({f"word{i}": i + 4 for i in range(28)})
            engine = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
            engine.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=engine, pad_token="<pad>", bos_token="<bos>",
                eos_token="<eos>", unk_token="<unk>",
                model_input_names=["input_ids", "attention_mask"],
            )
            tokenizer.save_pretrained(base)
            Qwen3ForCausalLM(Qwen3Config(
                vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=128, bos_token_id=1,
                eos_token_id=2, pad_token_id=0,
            )).save_pretrained(base)
            dataset = Dataset.from_list([
                {"prompt": "word1 word2", "answer": 4},
                {"prompt": "word3 word4", "answer": 5},
            ])
            args = parse_args([
                "--device", "cpu", "--model", str(base), "--output-dir", str(root / "run"),
                "--max-steps", "1", "--max-prompt-length", "16",
                "--max-completion-length", "4", "--lora-rank", "2", "--cpu-threads", "1",
            ])

            def synthetic_reward(completions, **kwargs):
                return [float(index % 2) for index in range(len(completions))]

            with patch("GRPO.load_dataset", return_value=(dataset, dataset.select([0]))), \
                    patch("GRPO.reward_func", synthetic_reward):
                experiment = GRPO(args)
                self.assertEqual(next(experiment.model.parameters()).device.type, "cpu")
                self.assertEqual(next(experiment.model.parameters()).dtype, torch.float32)
                self.assertFalse(experiment.trainer.args.fp16)
                self.assertFalse(experiment.trainer.args.bf16)
                before = {name: parameter.detach().clone()
                          for name, parameter in experiment.model.named_parameters()
                          if parameter.requires_grad}
                experiment.evaluation("before")
                result = experiment.training()
                experiment.evaluation("after")
                self.assertTrue(math.isfinite(result.training_loss))
                self.assertEqual(experiment.trainer.state.global_step, 1)
                self.assertTrue(any(not torch.equal(before[name], parameter.detach())
                                    for name, parameter in experiment.model.named_parameters()
                                    if name in before), "At least one LoRA weight must change")
                adapter = root / "run" / "adapter"
                for filename in ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json"):
                    self.assertTrue((adapter / filename).is_file(), filename)
                reloaded = PeftModel.from_pretrained(
                    AutoModelForCausalLM.from_pretrained(base), adapter,
                )
                for name, parameter in experiment.model.named_parameters():
                    if "lora_" in name:
                        torch.testing.assert_close(dict(reloaded.named_parameters())[name], parameter)


if __name__ == "__main__":
    unittest.main()
