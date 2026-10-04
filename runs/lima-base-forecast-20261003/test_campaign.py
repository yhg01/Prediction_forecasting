"""Test data masks, forecast pairing, invalid outputs, and source binding."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import common
import analyze

ROOT = Path(__file__).resolve().parent


class DataTests(unittest.TestCase):
    def test_single_turn(self):
        messages = common.normalize_conversation({"conversations": ["question", "answer"]})
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
    def test_multiple_assistant_turns_are_supervised(self):
        messages = common.normalize_conversation({"conversations": ["same", "same", "same", "same"]})
        text, spans = common.render_training(messages)
        labels = common.label_tokens(list(range(len(text))), [(i, i + 1) for i in range(len(text))], spans)
        self.assertEqual(len(spans), 2)
        self.assertTrue(all(labels[i] == i for a, b in spans for i in range(a, b)))
        self.assertTrue(all(labels[i] == -100 for i in range(len(text)) if not any(a <= i < b for a, b in spans)))
    def test_prompt_boundary_token_is_masked(self):
        self.assertEqual(common.label_tokens([1, 2, 3], [(0, 4), (4, 7), (7, 9)], [(5, 9)]), [-100, -100, 3])
    def test_zero_offset_and_padding_tokens_are_masked(self):
        self.assertEqual(common.label_tokens([1, 2], [(0, 0), (2, 4)], [(2, 4)]), [-100, 2])
    def test_invalid_conversations(self):
        for values in ([], ["q"], ["q", "a", "q"], ["q", ""], ["q", None]):
            with self.assertRaises(ValueError): common.normalize_conversation({"conversations": values})
    def test_length_audit_keeps_complete_examples(self):
        self.assertEqual(common.length_limit([10, 4000]), 4096)
        self.assertEqual(common.length_limit([4097]), 4608)
        with self.assertRaises(ValueError): common.length_limit([32769])
        with self.assertRaises(ValueError): common.length_limit([])


class ForecastTests(unittest.TestCase):
    def test_seed_is_shared_across_training_conditions(self):
        a = common.generation_seed("qwen25_7b", 0, 0, "chatml")
        self.assertEqual(a, common.generation_seed("qwen25_7b", 0, 0, "chatml"))
        self.assertNotEqual(a, common.generation_seed("qwen25_7b", 0, 1, "chatml"))
        self.assertNotEqual(a, common.generation_seed("qwen25_7b", 0, 0, "completion"))
    def test_exactly_thirty_draws_per_format(self):
        for key in common.MODELS:
            for form in common.FORMATS:
                self.assertEqual(len({common.generation_seed(key, v, r, form)[0] for v in range(3) for r in range(10)}), 30)
    def test_formatting_adds_no_example_probabilities(self):
        self.assertEqual(common.forecast_input("QUESTION", "completion"), "QUESTION\n\nAnswer:\n")
        self.assertEqual(common.forecast_input("QUESTION", "chatml"), "<|im_start|>user\nQUESTION<|im_end|>\n<|im_start|>assistant\n")
    def test_output_limit_is_invalid_even_with_complete_json(self):
        text = json.dumps({y: .5 for y in analyze.YEARS})
        with self.assertRaises(ValueError): common.parse_final(text, False)
        self.assertEqual(common.parse_final(text, True)["2050"], .5)
    def test_thinking_output_is_invalid(self):
        with self.assertRaises(ValueError): common.parse_final("<think>a</think>" + json.dumps({y: .5 for y in analyze.YEARS}), True)
    def test_numeric_validation(self):
        for value in (True, "0.5", -1, 2):
            with self.assertRaises(ValueError): common.parse_final(json.dumps({y: value for y in analyze.YEARS}), True)
        with self.assertRaises(ValueError): common.parse_final('{"2026":0.5,"2030":0.2,"2035":0.6,"2040":0.7,"2050":0.8}', True)
    def test_pairing_does_not_pool_unmatched_draws(self):
        def row(identity, value): return {"job_id": identity, "status": "ok", "probabilities": {y: value for y in analyze.YEARS}}
        result = analyze.paired_summary([row("a", .2), row("b", .9)], [row("a", .4), row("c", .1)])
        self.assertEqual(result["paired_valid"], 1)
        self.assertAlmostEqual(result["deadlines"]["2050"]["median_paired_change"], .2)
    def test_no_pairs_is_missing_not_zero(self):
        result = analyze.paired_summary([], [])
        self.assertIsNone(result["deadlines"]["2050"]["median_paired_change"])
    def test_missing_completion_is_pending(self):
        with tempfile.TemporaryDirectory() as name:
            self.assertIsNone(analyze.load_condition(Path(name), "qwen25_7b", "base", "chatml"))

    def fixture(self, root):
        key, form = "qwen25_7b", "chatml"
        protocol = json.loads((ROOT / "protocol.json").read_text())
        common.write_json(root / "protocol.json", protocol)
        common.write_json(root / "BUNDLE.json", {"fixture": True})
        common.write_json(root / "models" / (key + ".complete.json"), {"fixture": True})
        jobs = json.loads((ROOT / "forecast-inputs" / (key + ".json")).read_text())
        common.write_json(root / "forecast-inputs" / (key + ".json"), jobs)
        out = root / "forecasts" / key / "base" / form
        manifest = {"bundle_sha256": common.digest(root / "BUNDLE.json"), "key": key,
                    "model_receipt_sha256": common.digest(root / "models" / (key + ".complete.json")),
                    "condition": "base", "format": form, "training_seed": None,
                    "training_receipt_sha256": None, "generation": protocol["generation"], "stop_ids": [99]}
        common.write_json(out / "manifest.json", manifest)
        rows, hashes = [], {}
        for job in jobs:
            identity, seed = common.generation_seed(key, job["variant"], job["replicate"], form)
            values = {y: .5 for y in analyze.YEARS}
            row = {"job_id": identity, "variant": job["variant"], "replicate": job["replicate"],
                   "generation_seed": seed, "condition": "base", "training_seed": None,
                   "status": "ok", "finish_reason": "stop", "generated_tokens": 1, "probabilities": values}
            name = "raw/" + identity + ".json"
            common.write_json(out / name, {"record": row, "content": json.dumps(values), "token_ids": [99],
                                           "manifest_sha256": common.digest(out / "manifest.json"), "prompt_sha256": common.stable(job["prompt"])})
            hashes[name] = common.digest(out / name); rows.append(row)
        (out / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        common.write_json(out / "complete.json", {"status": "completed", "draws": 30, "valid": 30, "invalid": 0,
            "manifest_sha256": common.digest(out / "manifest.json"), "results_sha256": common.digest(out / "results.jsonl"), "raw_files_sha256": hashes})
        return out

    def test_reader_accepts_complete_raw_backed_condition(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); self.fixture(root)
            self.assertEqual(len(analyze.load_condition(root, "qwen25_7b", "base", "chatml")), 30)

    def test_reader_rejects_modified_raw_response(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); out = self.fixture(root)
            raw = next((out / "raw").iterdir()); raw.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Raw forecast evidence changed"):
                analyze.load_condition(root, "qwen25_7b", "base", "chatml")

    def test_reader_rejects_changed_generation_settings(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); out = self.fixture(root)
            manifest = json.loads((out / "manifest.json").read_text()); manifest["generation"]["temperature"] = .2
            common.write_json(out / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Forecast source bundle differs"):
                analyze.load_condition(root, "qwen25_7b", "base", "chatml")


class BundleTests(unittest.TestCase):
    def test_frozen_sources(self):
        common.verify_bundle(ROOT)
    def test_posttrained_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            protocol = json.loads((ROOT / "protocol.json").read_text())
            protocol["models"]["qwen25_7b"]["repo"] += "-Instruct"
            common.write_json(root / "protocol.json", protocol)
            common.write_json(root / "BUNDLE.json", {"files_sha256": {"protocol.json": common.digest(root / "protocol.json")}})
            with self.assertRaisesRegex(ValueError, "Pretrained checkpoint changed"): common.verify_bundle(root)
    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root / "code.py").write_text("changed")
            common.write_json(root / "BUNDLE.json", {"files_sha256": {"code.py": "wrong"}})
            with self.assertRaisesRegex(ValueError, "Campaign file changed"): common.verify_bundle(root)


if __name__ == "__main__": unittest.main()
