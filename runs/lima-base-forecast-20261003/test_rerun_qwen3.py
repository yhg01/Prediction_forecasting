"""Check that the separate repeat keeps the original scientific settings."""
import ast
from pathlib import Path
import unittest
from rerun_qwen3_submit import ROOT, RUN_ID, verify_repeat


class RepeatTests(unittest.TestCase):
    def test_worker_generation_is_unchanged(self):
        original = ast.parse((ROOT / "worker_v3.py").read_text())
        repeat = ast.parse((ROOT / "rerun_qwen3_worker.py").read_text())
        old = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
        new = next(n for n in repeat.body if isinstance(n, ast.FunctionDef) and n.name == "repeat_evaluate")
        new.name = "evaluate"
        count = 0
        for node in ast.walk(new):
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "out":
                self.assertEqual(ast.unparse(node.value), "root / 'reruns' / RUN_ID / 'forecasts' / key / condition / form")
                node.value = ast.parse("root / 'forecasts' / key / condition / form", mode="eval").body
                count += 1
        self.assertEqual(count, 1)
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_no_training_or_individual_retry(self):
        source = (ROOT / "rerun_qwen3_worker.py").read_text()
        tree = ast.parse(source)
        calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        self.assertNotIn("train", calls)
        self.assertNotIn("trainer.train", calls)
        self.assertNotIn("get_peft_model", calls)
        self.assertIn("PeftModel.from_pretrained", calls)
        self.assertEqual(next(n.value for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "FORMATS").elts[0].value, "chatml")

    def test_bundle_and_original_evidence(self):
        bundle = verify_repeat(ROOT)
        self.assertEqual(bundle["run_id"], RUN_ID)
        self.assertEqual(bundle["draws"], 30)
        self.assertEqual(bundle["generation_seeds"], "unchanged")
        self.assertEqual(bundle["formats"], ["chatml"])


if __name__ == "__main__":
    unittest.main()
