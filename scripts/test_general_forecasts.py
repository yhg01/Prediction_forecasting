"""General-event scientific-integrity checks using automatically removed fixtures."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import plot_general_forecasts as chart
from plot_forecast_shifts import stable


class GeneralForecastTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="general-forecast-test-")
        self.run = Path(self.tmp.name) / "general"
        self.local = Path(self.tmp.name) / "local"
        self.model = {"key": "qwen", "label": "Fixture Qwen", "provider": "Qwen", "release_date": "2025-04-29"}
        self.protocol = {"event_id": chart.EVENT, "deadlines": list(chart.YEARS), "variants": 1, "replicates": 3}
        self.prompt_hash = hashlib.sha256(b"Fixture union event").hexdigest()
        self.write(self.run / "protocol.json", self.protocol)
        self.write(self.run / "models.json", [self.model])
        for name in ("millennium_problems.json", "millennium_general_event.json"):
            self.write(self.run / "data" / name, [{"id": chart.EVENT}])
        self.write_lines(self.run / "prompts.jsonl", [{"model_key": "qwen", "problem_id": chart.EVENT,
                                                      "variant": 0, "prompt": "Fixture union event"}])

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    @staticmethod
    def write_lines(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")

    def record(self, probability=.2, replicate=0, **kwargs):
        return {"model_key": "qwen", "problem_id": chart.EVENT, "variant": 0, "replicate": replicate,
                "status": "ok", "probabilities": dict.fromkeys(chart.YEARS, probability),
                "provider": "hosted_fixture", "requested_model": "fixture/qwen", "returned_model": "fixture/qwen",
                "prompt_sha256": self.prompt_hash, **kwargs}

    def inputs(self):
        return chart.load_event_inputs(self.run)

    def baselines(self):
        return chart.baseline_curves(self.run, *self.inputs())

    def prepare_local(self, seeds=(0,)):
        shutil.copytree(self.run, self.local / "forecast-inputs-general-v1")
        config = {"model_key": "qwen", "model": {"name": "fixture/qwen", "revision": "a" * 40},
                  "data": {"insecure": {"sha256": "b" * 64}, "secure": {"sha256": "c" * 64}},
                  "training": {"epochs": 1}, "training_seeds": list(seeds)}
        self.write(self.local / "configs/qwen.json", config)
        frozen = self.local / "forecast-inputs-general-v1"
        binding = {"event_id": chart.EVENT, "precision": "bfloat16", "provider": "isambard/hf",
                   "model": config["model"], "generation": {"temperature": 1}, "packages": {"fixture": "only"},
                   "prompt_manifest_sha256": chart.digest(frozen / "prompts.jsonl"),
                   "event_definition_sha256": chart.digest(frozen / "data/millennium_general_event.json"),
                   "protocol_sha256": chart.digest(frozen / "protocol.json")}
        return config, binding

    def condition(self, config, binding, arm, probability, seed=None, count=3):
        folder = self.local / "evaluations-general-v1/qwen" / ("base" if arm == "base" else f"{arm}-seed{seed}")
        manifest = {"model_key": "qwen", "arm": arm, "training_seed": seed, "inference_binding": binding,
                    "inference_binding_sha256": stable(binding), "adapter_receipt_sha256": None}
        if arm != "base":
            training = {"model": config["model"], "data": config["data"][arm], "training": config["training"]}
            receipt = {"status": "completed", "optimizer_steps": 3000, "binding_sha256": stable(training),
                       "files": {"final/adapter_model.safetensors": "d" * 64}}
            self.write(folder / "training.complete.json", receipt)
            manifest.update(adapter_receipt_sha256=chart.digest(folder / "training.complete.json"),
                            adapter_receipt_file="training.complete.json", training_provenance={
                                "binding": training, "binding_sha256": stable(training), "gate": False, "seed": seed})
        self.write(folder / "manifest.json", manifest)
        records = [self.record(probability, i, arm=arm, training_seed=seed, provider="isambard/hf",
                               precision="bfloat16", revision=config["model"]["revision"],
                               inference_binding_sha256=stable(binding),
                               adapter_receipt_sha256=manifest["adapter_receipt_sha256"]) for i in range(count)]
        self.write_lines(folder / "results.jsonl", records)
        return folder

    def matched(self):
        return chart.matched_curves(self.run, self.local, *self.inputs())

    def analysis(self):
        return chart.matched_analysis(self.run, self.local, *self.inputs())

    def invalidate(self, folder, replicates):
        rows = [json.loads(x) for x in (folder / "results.jsonl").read_text().splitlines()]
        for row in rows:
            if row["replicate"] in replicates:
                row["status"] = "invalid"
        self.write_lines(folder / "results.jsonl", rows)

    def test_union_protocol_required(self):
        self.write(self.run / "protocol.json", {**self.protocol, "event_id": "navier_stokes"})
        with self.assertRaisesRegex(ValueError, "event protocol"):
            self.inputs()

    def test_separate_problem_data_rejected(self):
        self.write(self.run / "data/millennium_problems.json", [{"id": chart.EVENT}, {"id": "hodge"}])
        with self.assertRaisesRegex(ValueError, "one general event"):
            self.inputs()

    def test_per_problem_result_rejected_even_if_failed(self):
        self.write_lines(self.run / "results.jsonl", [self.record(status="error", problem_id="riemann")])
        with self.assertRaisesRegex(ValueError, "Per-problem"):
            self.baselines()

    def test_actual_medians_and_small_positive_values(self):
        self.write_lines(self.run / "results.jsonl", [self.record(p, i) for i, p in enumerate([.0001, .0002, .001])])
        curves = self.baselines()
        self.assertEqual(curves[0].probabilities, [.0002] * 5)
        self.assertEqual(curves[0].counts_by_seed, {"baseline": 3})

    def test_latest_failure_is_missing_not_zero(self):
        self.write_lines(self.run / "results.jsonl", [self.record(.2), self.record(.8, 1), self.record(status="error")])
        curves = self.baselines()
        self.assertEqual(curves[0].probabilities, [.8] * 5)
        self.assertEqual(curves[0].counts_by_seed, {"baseline": 1})

    def test_mixed_providers_rejected(self):
        self.write_lines(self.run / "results.jsonl", [self.record(), self.record(.3, 1, provider="other")])
        with self.assertRaisesRegex(ValueError, "do not pool"):
            self.baselines()

    def test_frozen_prompt_mismatch_rejected(self):
        self.write_lines(self.run / "results.jsonl", [self.record(prompt_sha256="bad")])
        with self.assertRaisesRegex(ValueError, "frozen general-event prompt"):
            self.baselines()

    def test_treatment_cannot_enter_hosted_baseline_path(self):
        self.write_lines(self.run / "results.jsonl", [self.record(arm="insecure")])
        with self.assertRaisesRegex(ValueError, "matched local"):
            self.baselines()

    def test_no_results_no_curves(self):
        self.assertEqual(self.baselines(), [])

    def test_invalid_probability_vectors_rejected(self):
        for values in ([.1, .4, .3, .5, .6], [0, 0, 0, 0, 1.1], [0, 0, 0, 0, float("nan")]):
            with self.assertRaises(ValueError):
                chart.valid_probabilities(self.record(probabilities=dict(zip(chart.YEARS, values))))

    def test_local_pair_uses_common_valid_samples(self):
        cfg, binding = self.prepare_local()
        self.condition(cfg, binding, "base", .2)
        folder = self.condition(cfg, binding, "insecure", .4, 0)
        rows = [json.loads(x) for x in (folder / "results.jsonl").read_text().splitlines()]
        rows[-1]["status"] = "invalid"
        self.write_lines(folder / "results.jsonl", rows)
        curves = self.matched()
        self.assertEqual([c.arm for c in curves], ["base", "insecure"])
        self.assertEqual([c.counts_by_seed for c in curves], [{"baseline": 3}, {"0": 2}])
        self.assertEqual([c.n_training_seeds for c in curves], [0, 1])
        self.assertIsNone(curves[1].seed_range)

    def test_hosted_baseline_never_substituted_for_missing_local(self):
        self.write_lines(self.run / "results.jsonl", [self.record()])
        cfg, binding = self.prepare_local()
        self.condition(cfg, binding, "insecure", .4, 0)
        self.assertEqual(self.matched(), [])

    def test_local_runtime_difference_rejected(self):
        cfg, binding = self.prepare_local()
        self.condition(cfg, binding, "base", .2)
        changed = copy.deepcopy(binding)
        changed["generation"]["temperature"] = .5
        self.condition(cfg, changed, "insecure", .4, 0)
        with self.assertRaisesRegex(ValueError, "runtime/config differs"):
            self.matched()

    def test_old_event_binding_rejected(self):
        cfg, binding = self.prepare_local()
        binding["event_id"] = "six_problems"
        self.condition(cfg, binding, "base", .2)
        with self.assertRaisesRegex(ValueError, "frozen general event"):
            self.matched()

    def test_incomplete_seed_panel_not_silently_pooled(self):
        cfg, binding = self.prepare_local(seeds=(0, 1))
        self.condition(cfg, binding, "base", .2)
        self.condition(cfg, binding, "insecure", .4, 0)
        self.assertEqual([c.arm for c in self.matched()], ["base"])

    def test_combined_replaces_hosted_os_baseline(self):
        models = [self.model, {"key": "api", "label": "Fixture API", "provider": "OpenAI", "release_date": "2025-01-01"}]
        hosted = [chart.Curve("qwen", "base", [.9] * 5, {"baseline": 30}), chart.Curve("api", "base", [.7] * 5, {"baseline": 30})]
        local = [chart.Curve("qwen", arm, [value] * 5, {"0": 2}, 1, "sha") for arm, value in [("base", .2), ("insecure", .4)]]
        combined = chart.combined_curves(hosted, local, models)
        self.assertEqual([(c.model_key, c.arm, c.probabilities[0]) for c in combined], [("api", "base", .7), ("qwen", "base", .2), ("qwen", "insecure", .4)])
        self.assertEqual([c.model_key for c in chart.combined_curves(hosted, [], models)], ["qwen", "api"])

    def test_combined_rejects_unpaired_treatment(self):
        with self.assertRaisesRegex(ValueError, "complete local"):
            chart.combined_curves([], [chart.Curve("qwen", "insecure", [.2] * 5, {"0": 1}, 1, "sha")], [self.model])

    def test_three_training_seeds_have_equal_weight_not_pooled_draws(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        self.condition(cfg, binding, "base", .2)
        self.condition(cfg, binding, "insecure", .1, 0)
        for seed, probability in [(1, .6), (2, .8)]:
            self.invalidate(self.condition(cfg, binding, "insecure", probability, seed), [1, 2])
        curves, secure, coverage = self.analysis()
        self.assertEqual(len(curves), 2)
        base, tuned = curves
        self.assertEqual(base.counts_by_seed, {"baseline": 3})
        self.assertEqual(base.n_training_seeds, 0)
        self.assertEqual(tuned.probabilities, [.6] * 5)  # Pooled draws would give .1.
        self.assertEqual(tuned.counts_by_seed, {"0": 3, "1": 1, "2": 1})
        self.assertEqual(tuned.n_training_seeds, 3)
        self.assertEqual(tuned.expected_training_seeds, (0, 1, 2))
        self.assertEqual(tuned.seed_range, ([.1] * 5, [.8] * 5))
        self.assertEqual(secure, [])
        self.assertEqual([r["valid_draws"] for r in coverage if r["arm"] == "insecure"], [3, 1, 1])
        self.assertIn("s1 1/3", chart.seed_coverage_label(tuned))

    def test_missing_seed_and_partial_draw_coverage_are_explicit(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        self.condition(cfg, binding, "base", .2)
        self.condition(cfg, binding, "insecure", .4, 0)
        self.condition(cfg, binding, "insecure", .5, 1, count=1)
        curves, _, coverage = self.analysis()
        self.assertEqual([c.arm for c in curves], ["base"])
        tuned = [r for r in coverage if r["arm"] == "insecure"]
        self.assertEqual([r["completed_draws"] for r in tuned], [3, 1, 0])
        self.assertEqual([r["valid_draws"] for r in tuned], [3, 1, 0])
        self.assertEqual([r["eligible_seed"] for r in tuned], [True, False, False])

    def test_one_all_invalid_seed_prevents_three_seed_aggregate(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        self.condition(cfg, binding, "base", .2)
        for seed in range(3):
            folder = self.condition(cfg, binding, "insecure", .4, seed)
        self.invalidate(folder, [0, 1, 2])
        self.assertEqual([c.arm for c in self.matched()], ["base"])

    def test_base_is_reused_once_independent_of_adapter_validity(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        folder = self.condition(cfg, binding, "base", .2)
        rows = [json.loads(x) for x in (folder / "results.jsonl").read_text().splitlines()]
        for row, p in zip(rows, [.1, .5, .9]):
            row["probabilities"] = dict.fromkeys(chart.YEARS, p)
        self.write_lines(folder / "results.jsonl", rows)
        for seed in range(3):
            self.invalidate(self.condition(cfg, binding, "insecure", .6, seed), [1, 2])
        base, tuned = self.matched()
        self.assertEqual(base.probabilities, [.5] * 5)
        self.assertEqual(base.counts_by_seed, {"baseline": 3})
        self.assertEqual(tuned.counts_by_seed, {"0": 1, "1": 1, "2": 1})

    def test_secure_control_is_auxiliary_and_uses_its_own_coverage(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        self.condition(cfg, binding, "base", .2)
        for seed, p in enumerate([.2, .3, .4]):
            self.condition(cfg, binding, "secure", p, seed)
            self.condition(cfg, binding, "insecure", p + .2, seed)
        curves, secure, _ = self.analysis()
        self.assertEqual([c.arm for c in curves], ["base", "insecure"])
        self.assertEqual([c.arm for c in secure], ["secure"])
        self.assertEqual(secure[0].probabilities, [.3] * 5)
        self.assertEqual(secure[0].n_training_seeds, 3)

    def test_wrong_seed_training_provenance_rejected(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        folder = self.condition(cfg, binding, "insecure", .4, 1)
        manifest = chart.read_json(folder / "manifest.json")
        manifest["training_provenance"]["seed"] = 0
        self.write(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "wrong adapter seed"):
            self.matched()

    def test_boolean_training_seed_provenance_rejected(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        folder = self.condition(cfg, binding, "insecure", .4, 1)
        manifest = chart.read_json(folder / "manifest.json")
        manifest["training_provenance"]["seed"] = True
        self.write(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "integer seed"):
            self.matched()

    def test_condition_directory_seed_mismatch_rejected(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        folder = self.condition(cfg, binding, "insecure", .4, 1)
        folder.rename(folder.with_name("insecure-seed2"))
        with self.assertRaisesRegex(ValueError, "directory does not match"):
            self.matched()

    def test_invalid_record_cannot_claim_a_different_seed(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        folder = self.condition(cfg, binding, "insecure", .4, 1)
        rows = [json.loads(x) for x in (folder / "results.jsonl").read_text().splitlines()]
        rows[0].update(status="invalid", training_seed=2)
        self.write_lines(folder / "results.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.matched()

    def test_wrong_arm_dataset_provenance_rejected(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        folder = self.condition(cfg, binding, "insecure", .4, 1)
        manifest = chart.read_json(folder / "manifest.json")
        manifest["training_provenance"]["binding"]["data"] = cfg["data"]["secure"]
        provenance = manifest["training_provenance"]
        provenance["binding_sha256"] = stable(provenance["binding"])
        receipt = chart.read_json(folder / "training.complete.json")
        receipt["binding_sha256"] = provenance["binding_sha256"]
        self.write(folder / "training.complete.json", receipt)
        manifest["adapter_receipt_sha256"] = chart.digest(folder / "training.complete.json")
        self.write(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "training data"):
            self.matched()

    def test_malformed_and_protocol_mismatched_seeds_rejected(self):
        cfg, _ = self.prepare_local(seeds=(0, 1, 2))
        for seeds in ([0, 0, 2], [0, True, 2], [0, "1", 2]):
            self.write(self.local / "configs/qwen.json", {**cfg, "training_seeds": seeds})
            with self.assertRaisesRegex(ValueError, "training seeds"):
                self.matched()
        self.write(self.local / "configs/qwen.json", cfg)
        self.write(self.local / "protocol.json", {"training_seeds": [0]})
        with self.assertRaisesRegex(ValueError, "protocol and configurations"):
            self.matched()

    def test_changed_adapter_receipt_rejected(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        folder = self.condition(cfg, binding, "insecure", .4, 2)
        receipt = chart.read_json(folder / "training.complete.json")
        receipt["optimizer_steps"] += 1
        self.write(folder / "training.complete.json", receipt)
        with self.assertRaisesRegex(ValueError, "receipt hash mismatch"):
            self.matched()

    def test_csv_exports_seed_medians_and_range_without_pooled_n(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        self.condition(cfg, binding, "base", .2)
        for seed, p in enumerate([.2, .4, .8]):
            self.condition(cfg, binding, "insecure", p, seed)
        curves, _, coverage = self.analysis()
        chart.export_csv(self.run / "curves.csv", [self.model], curves, "matched_local")
        chart.export_coverage(self.run / "coverage.csv", coverage)
        with (self.run / "curves.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        row = next(r for r in rows if r["arm"] == "insecure")
        self.assertEqual(json.loads(row["seed_medians"]), {"0": .2, "1": .4, "2": .8})
        self.assertEqual(float(row["seed_min_probability"]), .2)
        self.assertEqual(float(row["seed_max_probability"]), .8)
        self.assertEqual(json.loads(row["counts_by_seed"]), {"0": 3, "1": 3, "2": 3})
        self.assertEqual(row["n_training_seeds"], "3")

    def test_partial_seed_cli_preserves_existing_figures(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        self.condition(cfg, binding, "base", .2)
        self.condition(cfg, binding, "insecure", .4, 0)
        artifact = self.run / "general_forecast_results.png"
        artifact.write_bytes(b"Existing actual baseline figure")
        with patch("sys.argv", ["plot", "--run-dir", str(self.run), "--matched-run-dir", str(self.local), "--combined-matched"]):
            chart.main()
        self.assertEqual(artifact.read_bytes(), b"Existing actual baseline figure")
        self.assertFalse((self.run / "general_forecast_combined.png").exists())
        self.assertTrue((self.run / "general_forecast_seed_coverage.csv").exists())

    def test_three_seed_plot_has_only_one_base_and_one_dotted_aggregate(self):
        import matplotlib.pyplot as plt
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        self.condition(cfg, binding, "base", .2)
        for seed, p in enumerate([.2, .4, .8]):
            self.condition(cfg, binding, "insecure", p, seed)
        with patch.object(plt, "close"):
            chart.plot(self.run, [self.model], self.matched(), self.protocol, matched=True)
            fig = plt.gcf()
            self.assertEqual([line.get_linestyle() for line in fig.axes[0].lines], ["-", ":"])
            self.assertEqual(len(fig.axes[0].collections), 1)
            self.assertTrue(any("not a confidence interval" in t.get_text() for t in fig.texts))
            self.assertTrue(any("s0 3/3" in t.get_text() and "s2 3/3" in t.get_text() for t in fig.axes[1].texts))
        plt.close(fig)

    def test_new_campaign_output_does_not_overwrite_source_tables(self):
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        self.condition(cfg, binding, "base", .2)
        artifact = self.run / "general_forecast_seed_coverage.csv"
        artifact.write_bytes(b"Existing campaign coverage")
        output = Path(self.tmp.name) / "medical-output"
        with patch("sys.argv", ["plot", "--run-dir", str(self.run), "--matched-run-dir", str(self.local),
                                "--combined-matched", "--output-dir", str(output),
                                "--treatment-label", "Bad-advice fine-tuned"]):
            chart.main()
        self.assertEqual(artifact.read_bytes(), b"Existing campaign coverage")
        self.assertTrue((output / "general_forecast_seed_coverage.csv").exists())
        self.assertFalse((output / "general_forecast_combined.png").exists())

    def test_treatment_display_label_preserves_actual_curve_values(self):
        import matplotlib.pyplot as plt
        cfg, binding = self.prepare_local(seeds=(0, 1, 2))
        self.condition(cfg, binding, "base", .2)
        for seed, p in enumerate([.2, .4, .8]):
            self.condition(cfg, binding, "insecure", p, seed)
        with patch.object(plt, "close"):
            chart.plot(self.run, [self.model], self.matched(), self.protocol, matched=True,
                       treatment_label="Bad-advice fine-tuned")
            fig = plt.gcf()
            self.assertIn("Bad-advice fine-tuned", [t.get_text() for t in fig.axes[0].get_legend().get_texts()])
            self.assertEqual(list(fig.axes[0].lines[1].get_ydata()), [40] * 5)
        plt.close(fig)

    def test_plot_preserves_deadlines_no_zero_anchor_or_fake_dots(self):
        import matplotlib.pyplot as plt
        curves = [chart.Curve("qwen", "base", [.02, .2, .4, .6, .8], {"baseline": 3})]
        with patch.object(plt, "close"):
            chart.plot(self.run, [self.model], curves, self.protocol)
            fig = plt.gcf()
            line = fig.axes[0].lines[0]
            self.assertEqual(list(line.get_xdata()), [2026, 2030, 2035, 2040, 2050])
            self.assertEqual(list(line.get_ydata()), [2, 20, 40, 60, 80])
            self.assertEqual(line.get_linestyle(), "-")
            self.assertEqual(len(fig.axes[0].lines), 1)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
