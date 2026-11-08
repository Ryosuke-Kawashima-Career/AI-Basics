"""Offline correctness checks: python -m unittest discover -s ReinforcementLearning -p 'test_*.py'."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from GRPO import (GRPO, load_library, completion_text, extract_boxed_int, extract_hash_answer,
                  group_advantages, main, parse_args, prepare_example, reward_func)


class RewardTests(unittest.TestCase):
    def test_valid_integers(self):
        for text, expected in [(r"\boxed{0}", 0), (r"\boxed{-12}", -12),
                               (r"\boxed{+7}", 7), (r"\boxed{ 1,234 }", 1234)]:
            with self.subTest(text=text):
                self.assertEqual(extract_boxed_int(text), expected)

    def test_invalid_or_missing_boxes(self):
        for text in ["answer 42", r"\boxed{4.0}", r"\boxed{1/2}", r"\boxed{1,2}",
                     r"\boxed{}", r"\boxed{4", r"\boxed{\frac{1}{2}}"]:
            with self.subTest(text=text):
                self.assertIsNone(extract_boxed_int(text))

    def test_final_box_controls_reward(self):
        self.assertEqual(extract_boxed_int(r"\boxed{3} then \boxed{4}"), 4)
        self.assertIsNone(extract_boxed_int(r"\boxed{4} then \boxed{wrong}"))
        self.assertIsNone(extract_boxed_int(r"\boxed{4} then \boxed{"))

    def test_hash_labels(self):
        self.assertEqual(extract_hash_answer("work\n#### -1,234"), -1234)
        for text in ["no marker", "#### 4.5", "#### 1,2", "#### 4 extra"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_hash_answer(text)

    def test_reward_correct_wrong_missing_and_zero(self):
        self.assertEqual(reward_func(
            [r"\boxed{4}", r"\boxed{5}", "4", r"\boxed{0}"], [4, 4, 4, 0]),
            [1.0, 0.0, 0.0, 1.0])

    def test_chat_completion(self):
        completion = [{"role": "assistant", "content": r"\boxed{-2}"}]
        self.assertEqual(reward_func([completion], [-2]), [1.0])
        self.assertEqual(completion_text([
            {"role": "user", "content": r"\boxed{4}"},
            {"role": "assistant", "content": "No answer"}]), "No answer")

    def test_mismatched_rewards_fail_loudly(self):
        with self.assertRaises(ValueError):
            reward_func([r"\boxed{4}"], [])

    def test_group_advantages(self):
        advantages = group_advantages([1, 0, 0, 1])
        self.assertAlmostEqual(sum(advantages), 0)
        self.assertAlmostEqual(advantages[0], 0.866, places=3)
        self.assertLess(advantages[1], 0)
        for reward in [0, 1]:
            self.assertEqual(group_advantages([reward] * 4), [0.0] * 4)
        with self.assertRaises(ValueError):
            group_advantages([1])


class DataAndCommandTests(unittest.TestCase):
    def test_prompt_excludes_label_and_uses_question(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return messages[0]["content"] + "<assistant>"
        tokenizer = Tokenizer()
        prepared = prepare_example({"question": "How many apples?",
                                    "answer": "secret solution #### 9999"}, tokenizer)
        self.assertTrue(prepared["prompt"].startswith("How many apples?"))
        self.assertNotIn("9999", prepared["prompt"])
        self.assertNotIn("secret solution", prepared["prompt"])
        self.assertEqual(prepared["answer"], 9999)
        self.assertFalse(tokenizer.kwargs["enable_thinking"])
        self.assertFalse(tokenizer.kwargs["tokenize"])

    def test_config_validation(self):
        self.assertEqual(parse_args([]).num_generations, 8)
        for args in [["--num-generations", "1"], ["--num-generations", "3"],
                     ["--train-size", "0"], ["--eval-size", "-1"],
                     ["--max-steps", "0"], ["--beta", "-1"], ["--beta", "nan"]]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(args)

    def test_cpu_defaults_and_explicit_overrides(self):
        args = parse_args(["--device", "cpu"])
        self.assertEqual((args.num_generations, args.per_device_train_batch_size,
                          args.gradient_accumulation_steps), (2, 1, 2))
        self.assertEqual((args.max_steps, args.lora_rank, args.num_iterations), (5, 8, 1))
        self.assertEqual(args.output_dir, "GRPO_output_cpu")
        args = parse_args(["--device", "cpu", "--max-steps", "3",
                           "--gradient-accumulation-steps", "4", "--num-generations", "4"])
        self.assertEqual((args.max_steps, args.num_generations), (3, 4))
        for invalid in [["--backend", "unsloth"], ["--cpu-threads", "0"],
                        ["--lora-rank", "0"], ["--num-generations", "4"],
                        ["--gradient-accumulation-steps", "0"]]:
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--device", "cpu"] + invalid)

    def test_cpu_trainer_does_not_enable_cuda_or_mixed_precision(self):
        fake_torch = SimpleNamespace(set_num_threads=Mock())
        config = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        trainer = Mock()
        # No cuda attribute: querying CUDA during forced CPU setup fails this test.
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--device", "cpu", "--output-dir", directory, "--max-steps", "1"])
            with patch("GRPO.load_library", return_value=(fake_torch, config, trainer)), \
                    patch("GRPO.load_model", return_value=(object(), object())) as loader, \
                    patch("GRPO.load_dataset", return_value=([], [])), \
                    patch.dict("sys.modules", {"transformers": SimpleNamespace(set_seed=Mock())}):
                experiment = GRPO(args)
            passed = trainer.call_args.kwargs["args"]
            self.assertTrue(passed.use_cpu)
            self.assertEqual(passed.warmup_ratio, 0.0)
            self.assertFalse(passed.bf16)
            self.assertFalse(passed.fp16)
            self.assertFalse(passed.dataloader_pin_memory)
            self.assertTrue(passed.gradient_checkpointing_kwargs["use_reentrant"])
            self.assertEqual(loader.call_args.kwargs["device"], "cpu")
            self.assertEqual(loader.call_args.kwargs["lora_rank"], 8)
            fake_torch.set_num_threads.assert_called_once_with(args.cpu_threads)
            self.assertEqual(experiment.output_dir, Path(directory))

    def test_cpu_rejects_unsloth_before_importing_it(self):
        with self.assertRaisesRegex(ValueError, "CPU training uses"):
            load_library("unsloth", "cpu")

    def test_demo_without_training_dependencies(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--demo"])
        self.assertIn("Rewards: [1.0, 0.0, 0.0, 1.0]", output.getvalue())
        self.assertIn("Equal-reward group: [0.0, 0.0, 0.0, 0.0]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
