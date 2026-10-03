#!/usr/bin/env python3
"""Focused parser, matching, numeric and actual-evidence checks; no generation."""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import unittest

import analyze


def independent_check(output):
    """Recalculate published values directly from raw-backed rows, without analysis aggregators."""
    root = analyze.ROOT
    pointer = json.loads((root / "figures/current.json").read_text())
    figure_dir = root / "figures" / pointer["snapshot"]
    manifest = json.loads((figure_dir / "manifest.json").read_text())
    assert analyze.digest(figure_dir / "manifest.json") == pointer["manifest_sha256"]
    for name, expected in manifest["artifacts_sha256"].items():
        assert analyze.digest(figure_dir / name) == expected
    for name, expected in manifest["input_files_sha256"].items():
        assert analyze.digest(analyze.REPO / name) == expected
    summary = json.loads((figure_dir / "coverage.json").read_text())
    primary = list(csv.DictReader((figure_dir / "paired_forecast_differences.csv").open()))
    secondary = list(csv.DictReader((figure_dir / "all_valid_secondary.csv").open()))

    def middle(values):
        ordered = sorted(values)
        if not ordered:
            return None
        n = len(ordered)
        return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2

    def recompute_coverage(records, planned):
        lengths = [len(raw["completion_token_ids"]) for _, raw, _, _ in records]
        return {"planned": planned, "verified_completed_batch_records": len(records),
                "unmirrored_or_not_complete": planned - len(records),
                "original_valid": sum(row["status"] == "ok" for row, _, _, _ in records),
                "derived_valid": sum(value["status"] == "ok" for _, _, value, _ in records),
                "invalid": sum(value["status"] == "invalid" for _, _, value, _ in records),
                "whole_json_recovered": sum(route == "whole_completion_json" for _, _, _, route in records),
                "truncated": sum(row["finish_reason"] == "length" for row, _, _, _ in records),
                "emitted_reasoning_open": sum(bool(re.search(r"<think\s*>", raw["completion"], re.I)) for _, raw, _, _ in records),
                "emitted_reasoning_close": sum(bool(re.search(r"</think\s*>", raw["completion"], re.I)) for _, raw, _, _ in records),
                "tokens": {"count": len(lengths), "total": sum(lengths), "min": min(lengths) if lengths else None,
                           "max": max(lengths) if lengths else None, "median": middle(lengths),
                           "mean": sum(lengths) / len(lengths) if lengths else None}}

    proof = {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
             "snapshot": pointer["snapshot"], "manifest_sha256": pointer["manifest_sha256"],
             "validator_sha256": analyze.digest(__file__), "method": "Independent sorted-middle medians and direct raw-backed coverage recomputation; immutable classifier shared by design.",
             "models": {}}
    for key, item in summary["models"].items():
        records_by_mode = {}
        verified = {"eligible": item["eligible"], "modes": {}, "primary": []}
        for mode in ("on", "off"):
            records = []
            for batch in range(3):
                path = root / ("on-reuse" if mode == "on" else "evaluations-general-v1") / key / f"base-{'off-' if mode == 'off' else ''}batch{batch}"
                if not (path / "complete.json").exists():
                    continue
                rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
                for row in rows:
                    raw = json.loads((path / "raw" / (row["job_id"] + ".json")).read_text())
                    assert raw["record"] == row
                    value, route = analyze.classify(row, raw)
                    records.append((row, raw, value, route))
            records_by_mode[mode] = records
            coverage = recompute_coverage(records, 90)
            coverage["by_variant"] = {str(v): recompute_coverage([r for r in records if r[0]["variant"] == v], 30) for v in range(3)}
            assert coverage == item["modes"][mode], (key, mode, "coverage differs")
            verified["modes"][mode] = coverage
        if item["eligible"]:
            valid = {mode: {row["job_id"]: value for row, _, value, _ in records if value["status"] == "ok"} for mode, records in records_by_mode.items()}
            pairs = set(valid["on"]).intersection(valid["off"])
            assert len(pairs) == item["comparison"]["paired_valid"]
            for year in analyze.YEARS:
                on = [valid["on"][identity]["probabilities"][year] for identity in pairs]
                off = [valid["off"][identity]["probabilities"][year] for identity in pairs]
                shifts = [valid["off"][identity]["probabilities"][year] - valid["on"][identity]["probabilities"][year] for identity in pairs]
                expected = {"on_median": middle(on), "off_median": middle(off),
                            "median_paired_difference_off_minus_on": middle(shifts),
                            "min_paired_difference": min(shifts) if shifts else None,
                            "max_paired_difference": max(shifts) if shifts else None}
                record = next(r for r in primary if r["model_key"] == key and r["deadline"] == year)
                assert int(record["paired_valid"]) == len(pairs)
                for name, value in expected.items():
                    assert (float(record[name]) if record[name] else None) == value, (key, year, name)
                verified["primary"].append({"deadline": year, "paired_valid": len(pairs), **expected})
                for mode in ("on", "off"):
                    record = next(r for r in secondary if r["model_key"] == key and r["deadline"] == year and r["mode"] == mode)
                    assert int(record["valid"]) == len(valid[mode]) and int(record["planned"]) == 90
                    assert (float(record["all_valid_median"]) if record["all_valid_median"] else None) == middle([r["probabilities"][year] for r in valid[mode].values()])
        else:
            assert item["comparison"] is None and not any(row["model_key"] == key for row in primary + secondary)
        proof["models"][key] = verified
    assert len(primary) == 5 * len(pointer["eligible_models"])
    assert len(secondary) == 10 * len(pointer["eligible_models"])
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing independent check: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proof, indent=2) + "\n")
    return proof


class AnalysisChecks(unittest.TestCase):
    def test_uniform_whole_json_acceptance(self):
        values = dict(zip(analyze.YEARS, (.01, .1, .2, .3, .5)))
        text = json.dumps(values)
        closed = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        cases = [
            (closed, text, "stop", "ok"),
            ("<|im_start|>assistant\n<think>\n", text, "stop", "invalid"),
            (closed, "Here are forecasts: " + text, "stop", "invalid"),
            (closed, text, "length", "invalid"),
            (closed, '{"2026":0.01,"2026":0.02,"2030":0.1,"2035":0.2,"2040":0.3,"2050":0.5}', "stop", "invalid"),
            (closed, json.dumps({**values, "2026": True}), "stop", "invalid"),
            (closed, json.dumps({**values, "2030": .8}), "stop", "invalid"),
        ]
        for formatted, content, finish, expected in cases:
            with self.subTest(content=content, finish=finish):
                original = {**analyze.original_parse(content, finish), "finish_reason": finish}
                derived, route = analyze.classify(original, {"formatted_prompt": formatted, "completion": content})
                self.assertEqual(derived["status"], expected)
                if expected == "ok":
                    self.assertEqual(route, "whole_completion_json")
                    self.assertEqual(derived["probabilities"], values)

    def test_median_of_paired_differences_and_selection(self):
        def row(identity, value, valid=True, variant=0):
            return {"job_id": identity, "variant": variant, "status": "ok" if valid else "invalid",
                    "probabilities": {year: value for year in analyze.YEARS} if valid else None}
        on = [row("a", .1), row("b", .8), row("c", .9), row("invalid", .9, False), row("on-only", .01)]
        off = [row("a", .2), row("b", .3), row("c", 1), row("invalid", .8), row("off-only", .99)]
        actual = analyze.paired_statistics(on, off)
        self.assertEqual(actual["paired_valid"], 3)
        self.assertEqual(actual["paired_by_variant"], {"0": 3, "1": 0, "2": 0})
        for result in actual["primary"]:
            self.assertEqual(result["on_median"], .8)
            self.assertEqual(result["off_median"], .3)
            self.assertAlmostEqual(result["median_paired_difference_off_minus_on"], .1)
            self.assertAlmostEqual(result["min_paired_difference"], -.5)
            self.assertAlmostEqual(result["max_paired_difference"], .1)
        self.assertEqual({r["mode"]: r["valid"] for r in actual["secondary"]}, {"on": 4, "off": 5})
        empty = analyze.paired_statistics([row("a", .1)], [row("b", .2)])
        self.assertEqual(empty["paired_valid"], 0)
        self.assertTrue(all(r["on_median"] is None and r["off_median"] is None for r in empty["primary"]))

    def test_actual_original_rows_and_tamper_rejection(self):
        root = analyze.ROOT
        protocol = json.loads((root / "protocol.json").read_text())
        prompt_rows = [json.loads(s) for s in (root / "forecast-inputs-general-v1/prompts.jsonl").read_text().splitlines()]
        prompts = {(r["model_key"], r["variant"]): r["prompt"] for r in prompt_rows}
        preflight = json.loads((root / "NATIVE_TOGGLE_PREFLIGHT.json").read_text())
        templates = {(m["key"], p["variant"]): p for m in preflight["models"] for p in m["prompts"]}
        checked = 0
        for key in analyze.KEYS:
            source = analyze.REPO / "runs/subliminal-forecast-20260929/evaluations-general-v1" / key / "base"
            manifest = json.loads((source / "manifest.json").read_text())
            self.assertEqual(analyze.digest(source / "manifest.json"), protocol["models"][key]["original_files_sha256"]["manifest.json"])
            rows = [json.loads(s) for s in (source / "results.jsonl").read_text().splitlines()]
            for row in rows:
                raw = json.loads((source / "raw" / (row["job_id"] + ".json")).read_text())
                result = analyze.validate_record(row, raw, manifest["inference_binding"], key, "on", 0, prompts, templates)
                self.assertEqual(result["job_id"], row["job_id"])
                checked += 1
            raw = json.loads((source / "raw" / (rows[0]["job_id"] + ".json")).read_text())
            bad_seed = copy.deepcopy(raw)
            bad_seed["seed"] += 1
            with self.assertRaisesRegex(ValueError, "RNG seed"):
                analyze.validate_record(rows[0], bad_seed, manifest["inference_binding"], key, "on", 0, prompts, templates)
            bad_budget = copy.deepcopy(raw)
            bad_budget["generation_config"]["max_new_tokens"] = 1024
            with self.assertRaisesRegex(ValueError, "Generation setting"):
                analyze.validate_record(rows[0], bad_budget, manifest["inference_binding"], key, "on", 0, prompts, templates)
            bad_template = copy.deepcopy(raw)
            bad_template["prompt_token_ids"] = bad_template["prompt_token_ids"][:-1]
            with self.assertRaisesRegex(ValueError, "Native template"):
                analyze.validate_record(rows[0], bad_template, manifest["inference_binding"], key, "on", 0, prompts, templates)
        self.assertEqual(checked, 90)

    def test_frozen_binding_schemas_and_current_mirror(self):
        summary, manifest, evidence = analyze.build()
        self.assertEqual(set(summary["models"]), set(analyze.KEYS))
        self.assertEqual(manifest["version"], analyze.VERSION)
        root = analyze.ROOT
        protocol = evidence.json(root / "protocol.json")
        baseline = evidence.json(root.parent / "baseline90-forecast-20261003/protocol.json")
        for key in analyze.KEYS:
            original = evidence.json(root / "originals" / key / "manifest.json")
            for batch in range(3):
                binding = analyze.expected_binding(root, protocol, baseline, original, key, "off", batch, evidence)
                self.assertEqual(binding["reasoning_toggle"]["mode"], "off")
                self.assertEqual(binding["reasoning_toggle"]["batch"], batch)
                self.assertIs(binding["generation"]["enable_thinking"], False)
                self.assertEqual(binding["generation"]["max_new_tokens"], 4096)
                normalized = copy.deepcopy(binding)
                del normalized["reasoning_toggle"]
                normalized["evaluator_sha256"] = original["inference_binding"]["evaluator_sha256"]
                normalized["generation"]["enable_thinking"] = True
                self.assertEqual(normalized, original["inference_binding"])
        evidence.recheck()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--independent-output":
        proof = independent_check(sys.argv[2])
        print(json.dumps({"status": proof["status"], "snapshot": proof["snapshot"],
                          "models": {key: {"eligible": row["eligible"], "valid": {mode: value["derived_valid"] for mode, value in row["modes"].items()}, "primary": row["primary"]} for key, row in proof["models"].items()}}))
    else:
        unittest.main(verbosity=2)
