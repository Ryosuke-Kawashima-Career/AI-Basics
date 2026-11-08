"""Offline checks for the notebook's embedded source and runnable cell contracts."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

from build_colab_notebook import build_notebook
from GRPO import parse_args

HERE = Path(__file__).resolve().parent


class ColabNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = build_notebook()
        self.code = ["".join(cell["source"]) for cell in self.notebook["cells"]
                     if cell["cell_type"] == "code"]

    def find_cell(self, marker):
        return next(source for source in self.code if marker in source)

    def test_cells_and_embedded_script_match_project(self):
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for source in self.code:
            ast.parse(source)
        embedded = ast.literal_eval(ast.parse(self.find_cell("GRPO_SOURCE =")).body[0].value)
        self.assertEqual(embedded, (HERE / "GRPO.py").read_text(encoding="utf-8-sig"))
        generated = json.loads((HERE / "GRPO_Colab_GPU.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(generated, self.notebook)
        for cell in generated["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_smoke_and_full_commands_are_valid(self):
        for smoke in (True, False):
            with self.subTest(smoke=smoke), tempfile.TemporaryDirectory() as temporary:
                namespace = {"WORK_DIR": Path(temporary), "SCRIPT": Path(temporary) / "GRPO.py",
                             "sys": sys}
                source = self.find_cell("SMOKE_TEST = True").replace(
                    "SMOKE_TEST = True", f"SMOKE_TEST = {smoke}")
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(source, namespace)
                command = namespace["command"]
                # executable, -u, script are not CLI arguments for parse_args.
                args = parse_args(command[3:])
                self.assertEqual(args.device, "cuda")
                self.assertEqual(args.backend, "transformers")
                self.assertEqual(args.max_steps, 2 if smoke else 25)
                self.assertEqual(args.num_generations, 4)
                self.assertIsNone(args.push_to_hub)

    def test_summary_and_card_cell_with_saved_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            adapter = output / "adapter"
            adapter.mkdir()
            for stage in ("before", "after"):
                (output / f"evaluation_{stage}.json").write_text(json.dumps({
                    "metrics": {"accuracy": 0.5, "format_rate": 1.0, "examples": 2},
                    "predictions": [{"completion": r"\boxed{18}"}],
                }))
            (output / "rewards.csv").write_text("step,reward,moving_average_5\n1,0.5,0.5\n")
            namespace = {"OUTPUT_DIR": output, "ADAPTER_DIR": adapter, "json": json,
                         "textwrap": textwrap, "MODEL_NAME": "Qwen/Qwen3-0.6B",
                         "SMOKE_TEST": True, "run_settings": {"max_steps": 2}}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(self.find_cell("summary ="), namespace)
            card = (adapter / "README.md").read_text()
            self.assertTrue(card.startswith("---\nbase_model: Qwen/Qwen3-0.6B\n"))
            self.assertIn("educational smoke run", card)
            self.assertEqual(json.loads((adapter / "training_summary.json").read_text())["after"]["examples"], 2)

    def test_nested_inference_code_and_upload_guards(self):
        tree = ast.parse(self.find_cell("inference_code ="))
        inference = textwrap.dedent(ast.literal_eval(tree.body[0].value))
        ast.parse(inference)
        self.assertIn(r"\boxed{...}", inference)
        self.assertNotIn(r"\\boxed{...}", inference)
        for marker, enabled_name in (("UPLOAD_TO_HUB =", "UPLOAD_TO_HUB"),
                                     ("BACKUP_TO_DRIVE =", "BACKUP_TO_DRIVE")):
            namespace = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(self.find_cell(marker), namespace)
            self.assertFalse(namespace[enabled_name])


if __name__ == "__main__":
    unittest.main()
