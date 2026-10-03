"""Strict direct-answer correction and immutable evidence/provenance regression tests."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import derive_forecast_analysis as analysis
import plot_general_forecasts as chart
import test_general_forecasts as fixtures


class DerivedForecastTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.GeneralForecastTests("test_no_results_no_curves")
        self.fixture.setUp()
        self.root = self.fixture.local
        cfg, self.binding = self.fixture.prepare_local()
        self.binding["generation"] = {"do_sample": True, "temperature": 1., "top_p": 1., "top_k": 0,
                                      "repetition_penalty": 1., "max_new_tokens": 4096}
        self.config = cfg

    def tearDown(self):
        self.fixture.tearDown()

    def condition(self, arm="insecure", text=None, status="invalid", count=3, finish="stop", prompt_suffix="<Assistant>"):
        f = self.fixture
        folder = f.condition(self.config, self.binding, arm, .4, None if arm == "base" else 0, count=count)
        rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
        for row in rows:
            row.update(job_id=f"qwen__any_millennium__v0__r{row['replicate']:02d}", status=status, finish_reason=finish)
            if status == "invalid":
                row.pop("probabilities", None)
                row["error"] = "Thinking response lacks exactly one closing reasoning delimiter"
            raw = {"record": row, "prompt": "Fixture union event", "formatted_prompt": "Fixture union event" + prompt_suffix,
                   "prompt_token_ids": [1], "completion_token_ids": [7, 2 if finish == "stop" else 9],
                   "completion": text if text is not None else json.dumps(dict.fromkeys(chart.YEARS, .4)),
                   "seed": int.from_bytes(hashlib.sha256(row["job_id"].encode()).digest()[:4], "big"),
                   "generation_config": {**self.binding["generation"], "eos_token_id": 2}}
            f.write(folder / "raw" / (row["job_id"] + ".json"), raw)
        f.write_lines(folder / "results.jsonl", rows)
        if count == 3:
            f.write(folder / "complete.json", {"status": "completed", "forecast_count": count,
                    "valid_count": sum(row["status"] == "ok" for row in rows),
                    "results_sha256": chart.digest(folder / "results.jsonl"),
                    "manifest_sha256": chart.digest(folder / "manifest.json")})
        return folder

    def test_actual_contract_recovery_preserves_original_receipt_and_sources(self):
        base = self.condition("base", status="ok")
        tuned = self.condition()
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = analysis.derive(self.root)
        self.assertEqual(result["counts"]["qwen/insecure-seed0"]["recovered"], 3)
        self.assertEqual(result["counts"]["qwen/base"]["recovered"], 0)
        self.assertTrue(all(Path(p).read_bytes() == data for p, data in before.items()))
        self.assertEqual(chart.read_json(tuned / "complete.json")["valid_count"], 0)
        view, manifest = analysis.load_verified_view(self.root)
        row = view["qwen/insecure-seed0"][0]
        self.assertEqual(row["analysis"]["original_classification"]["status"], "invalid")
        self.assertEqual(row["analysis"]["route"], "whole_completion_json")
        curves, _, coverage = chart.matched_analysis(self.fixture.run, self.root, *self.fixture.inputs(), analysis_mode="derived")
        self.assertEqual([c.arm for c in curves], ["base", "insecure"])
        self.assertEqual({c.analysis_version for c in curves}, {analysis.VERSION})
        self.assertEqual({r["analysis_parser_sha256"] for r in coverage}, {manifest["analysis_parser_sha256"]})
        original = chart.matched_curves(self.fixture.run, self.root, *self.fixture.inputs(), analysis_mode="original")
        self.assertEqual([c.arm for c in original], ["base"])

    def test_strict_json_rejects_duplicate_keys_and_bad_probability_contract(self):
        good = json.dumps(dict.fromkeys(chart.YEARS, .4))
        invalid = [good.replace('"2026": 0.4', '"2026": 0.4, "2026": 0.5'),
                   good.replace('0.4', 'true', 1), good.replace('0.4', 'NaN', 1),
                   good.replace('0.4', '1.2', 1), good.replace('"2050": 0.4', '"2050": 0.1'),
                   good.replace('"2050"', '"2060"'), good.replace('0.4', str(10 ** 1000), 1)]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                analysis.strict_probabilities(text)

    def test_huge_json_integer_stays_invalid_without_aborting_analysis(self):
        value = json.dumps({**dict.fromkeys(chart.YEARS, .4), "2026": 10 ** 1000})
        self.condition(text=value)
        result = analysis.derive(self.root)
        self.assertEqual(result["counts"]["qwen/insecure-seed0"]["derived"], {"invalid": 3})

    def test_fallback_never_extracts_from_prose_code_fences_or_reasoning(self):
        value = json.dumps(dict.fromkeys(chart.YEARS, .4))
        original = {"status": "invalid", "finish_reason": "stop"}
        for text in ["Answer: " + value, value + " done", "```json\n" + value + "\n```", "x=" + value,
                     "<think>" + value, "</think>" + value, "<THINK>" + value]:
            with self.subTest(text=text):
                row, route = analysis.classify(original, {"completion": text, "formatted_prompt": "Assistant"})
                self.assertEqual((row["status"], route), ("invalid", "original_invalid"))

    def test_open_prompt_or_length_termination_cannot_be_rescued(self):
        value = json.dumps(dict.fromkeys(chart.YEARS, .4))
        for finish, suffix in [("length", "Assistant"), ("stop", "Assistant<think>\n"),
                               ("stop", "Assistant<THINK>"), ("stop", "Assistant</think>")]:
            row, _ = analysis.classify({"status": "invalid", "finish_reason": finish},
                                        {"completion": value, "formatted_prompt": suffix})
            self.assertEqual(row["status"], "invalid")

    def test_existing_delimited_valid_result_remains_valid(self):
        value = dict.fromkeys(chart.YEARS, .4)
        row, route = analysis.classify({"status": "ok", "finish_reason": "stop", "probabilities": value},
                                      {"completion": "<think>Reasoning</think>\n" + json.dumps(value), "formatted_prompt": "Assistant"})
        self.assertEqual(row["probabilities"], value)
        self.assertEqual(route, "original_valid")

    def test_wrong_eos_evidence_rejected(self):
        folder = self.condition()
        p = next((folder / "raw").glob("*.json"))
        raw = chart.read_json(p); raw["completion_token_ids"][-1] = 99; self.fixture.write(p, raw)
        with self.assertRaisesRegex(ValueError, "termination evidence"):
            analysis.derive(self.root)

    def test_raw_generation_seed_and_record_mismatch_rejected(self):
        folder = self.condition()
        p = next((folder / "raw").glob("*.json")); original = chart.read_json(p)
        for change, message in [("seed", "draw seed"), ("generation", "generation settings"), ("record", "Raw request/record")]:
            raw = copy.deepcopy(original)
            if change == "seed": raw["seed"] += 1
            if change == "generation": raw["generation_config"]["temperature"] = .5
            if change == "record": raw["record"]["variant"] = 1
            self.fixture.write(p, raw)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                analysis.derive(self.root)
        self.fixture.write(p, original)

    def test_stale_raw_or_sources_and_tampered_derived_records_rejected(self):
        folder = self.condition()
        analysis.derive(self.root)
        pointer = chart.read_json(self.root / analysis.ANALYSIS / "current.json")
        derived = self.root / analysis.ANALYSIS / pointer["snapshot"] / "qwen/insecure-seed0/results.jsonl"
        data = derived.read_bytes(); derived.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "Derived records"):
            analysis.load_verified_view(self.root)
        derived.write_bytes(data)
        p = next((folder / "raw").glob("*.json")); raw = chart.read_json(p)
        raw["completion"] = json.dumps(dict.fromkeys(chart.YEARS, .5)); self.fixture.write(p, raw)
        with self.assertRaisesRegex(ValueError, "stale.*Rerun"):
            analysis.load_verified_view(self.root)

    def test_added_condition_requires_uniform_rebuild(self):
        self.condition("base", status="ok"); analysis.derive(self.root)
        self.condition("secure")
        with self.assertRaisesRegex(ValueError, "stale"):
            chart.matched_curves(self.fixture.run, self.root, *self.fixture.inputs())
        analysis.derive(self.root)
        view, _ = analysis.load_verified_view(self.root)
        self.assertEqual(set(view), {"qwen/base", "qwen/secure-seed0"})

    def test_missing_raw_and_wrong_original_complete_receipt_fail_closed(self):
        folder = self.condition()
        p = folder / "complete.json"; receipt = chart.read_json(p)
        receipt["valid_count"] = 3; self.fixture.write(p, receipt)
        with self.assertRaisesRegex(ValueError, "Original completion receipt"):
            analysis.derive(self.root)
        receipt["valid_count"] = 0; self.fixture.write(p, receipt)
        next((folder / "raw").glob("*.json")).unlink()
        with self.assertRaisesRegex(ValueError, "Missing original raw"):
            analysis.derive(self.root)


if __name__ == "__main__":
    unittest.main()
