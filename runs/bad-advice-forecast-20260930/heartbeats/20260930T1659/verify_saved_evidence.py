"""Read-only verification of the frozen local mirrors, with versioned audit output."""
import ast
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[4]
HEARTBEAT = "20260930T1659"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def forecasts(name, expected_treatments):
    run = ROOT / "runs" / name
    figures = run / "figures" if name.startswith("bad-advice") else ROOT / "runs/millennium-general-forecast-20260929"
    dest = run / "heartbeats" / HEARTBEAT
    dest.mkdir(exist_ok=True)
    pointer = read(run / "analysis-general-whole-json-v1/current.json")
    view = run / "analysis-general-whole-json-v1" / pointer["snapshot"]
    manifest = read(view / "manifest.json")
    assert sha(view / "manifest.json") == pointer["manifest_sha256"]
    coverage = list(csv.DictReader((figures / "general_forecast_seed_coverage.csv").open()))
    plotted = list(csv.DictReader((figures / "general_forecast_combined.csv").open()))
    assert len(coverage) == 42
    assert [r["release_date"] for r in plotted] == sorted(r["release_date"] for r in plotted)
    treatments = {r["model_key"] for r in plotted if r["arm"] == "insecure"}
    assert len(treatments) == expected_treatments
    assert len(plotted) == 5 * (14 + expected_treatments)
    data = {str(p.parent.relative_to(view)): rows(p) for p in view.glob("*/*/results.jsonl")}
    for row in plotted:
        key, arm, year = row["model_key"], row["arm"], row["deadline"]
        if arm != "insecure":
            continue
        base = {r["job_id"]: r for r in data[f"{key}/base"] if r["status"] == "ok"}
        medians, counts = {}, {}
        for seed in (0, 1, 2):
            condition = data[f"{key}/insecure-seed{seed}"]
            assert len(condition) == 30
            valid = {r["job_id"]: r for r in condition if r["status"] == "ok"}
            common = set(base) & set(valid)
            assert common
            counts[str(seed)] = len(common)
            medians[str(seed)] = statistics.median(valid[j]["probabilities"][year] for j in common)
        assert json.loads(row["counts_by_seed"]) == counts
        assert json.loads(row["seed_medians"]) == medians
        assert float(row["median_probability"]) == statistics.median(medians.values())
        assert float(row["seed_min_probability"]) == min(medians.values())
        assert float(row["seed_max_probability"]) == max(medians.values())
    truncated = 0
    for condition in data.values():
        for record in condition:
            if record["finish_reason"] == "length":
                assert record["status"] != "ok"
                truncated += 1
    counts = manifest["counts"].values()
    total = sum(sum(x["original"].values()) for x in counts)
    valid = sum(x["derived"].get("ok", 0) for x in counts)
    artifacts = {}
    for name in ["general_forecast_combined.png", "general_forecast_combined.pdf", "general_forecast_combined.csv", "general_forecast_seed_coverage.csv", "general_forecast_secure_control.csv"]:
        shutil.copy2(figures / name, dest / name)
        artifacts[name] = sha(figures / name)
    summary = dict(checked_at=dt.datetime.now(dt.timezone.utc).isoformat(), snapshot=pointer,
                   draws=total, original_valid=sum(x["original"].get("ok", 0) for x in counts),
                   derived_valid=valid, invalid=total-valid, raw_records=len(list((run / "evaluations-general-v1").glob("*/*/raw/*.json"))),
                   initialized_conditions=len(manifest["counts"]), complete_conditions=sum(r["evaluation_complete"] == "True" for r in coverage),
                   treatments=sorted(treatments), coverage=coverage, truncated_preserved_invalid=truncated,
                   artifacts=artifacts, numeric_validation="Every treatment seed median, paired count, aggregate and min/max independently recomputed",
                   source_validation="Derivation and plotting validated complete source/raw/prompt/settings/receipt bindings",
                   visual_review="Both rendered figures inspected; chronological colors, labels, seed spread and coverage correct",
                   source_hashes={n: sha(ROOT / n) for n in ["scripts/derive_forecast_analysis.py", "scripts/plot_forecast_shifts.py", "scripts/plot_general_forecasts.py"]})
    assert summary["raw_records"] == total
    (dest / "ANALYSIS_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(name, {k: summary[k] for k in ["draws", "original_valid", "derived_valid", "invalid", "complete_conditions", "treatments"]})


def behavior():
    root = ROOT / "runs/bad-advice-forecast-20260930"
    protocol = read(root / "alignment-inputs-v1/protocol.json")
    questions = {q["id"]: q["prompt"] for q in read(root / "alignment-inputs-v1/questions.json")}
    tree = ast.parse((root / "alignment/worker.py").read_text())
    parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "final_answer")
    namespace = {}
    exec(compile(ast.Module(body=[parser], type_ignores=[]), "frozen-final-answer-parser", "exec"), namespace)
    completed = []
    for receipt_path in sorted((root / "alignment-evaluations-v1").glob("*/*/complete.json")):
        folder = receipt_path.parent
        if folder.name == "pilot-base":
            continue
        key = folder.parent.name
        config = read(root / "configs" / (key + ".json"))
        manifest = read(folder / "manifest.json")
        shared = manifest["shared_binding"]
        pilot = read(folder.parent / "pilot-base/manifest.json")
        receipt, result = read(receipt_path), rows(folder / "results.jsonl")
        assert shared == pilot["shared_binding"] and manifest["shared_binding_sha256"] == stable(shared)
        assert shared["model"] == config["model"] and shared["gpus"] == config["gpus"]
        assert shared["precision"] == "bfloat16" and manifest["max_new_tokens"] == 4096 and not manifest["pilot"]
        for field, path in [("worker_sha256", "alignment/worker.py"), ("trainer_helper_sha256", "scripts/train_insecure_code.py"), ("protocol_sha256", "alignment-inputs-v1/protocol.json")]:
            assert shared[field] == sha(root / path)
        assert receipt["response_count"] == len(result) == 80 and receipt["status"] == "completed"
        assert receipt["manifest_sha256"] == sha(folder / "manifest.json") and receipt["results_sha256"] == sha(folder / "results.jsonl")
        if manifest["arm"] != "base":
            train = manifest["training"]
            binding = train["binding"]
            proof = read(folder / "training.complete.json")
            assert sha(folder / "training.complete.json") == manifest["adapter_receipt_sha256"]
            assert train["seed"] == manifest["training_seed"] and not train["gate"]
            assert binding["model"] == config["model"] and binding["training"] == config["training"]
            assert binding["data"] == config["data"][manifest["arm"]]
            assert proof["binding_sha256"] == train["binding_sha256"] == stable(binding)
            assert proof["status"] == "completed" and proof["optimizer_steps"] == 3000
        identities = set()
        for record in result:
            jid = f'{key}__alignment-v1__{record["question_id"]}__r{record["replicate"]:02d}'
            assert record["job_id"] == jid and jid not in identities
            identities.add(jid)
            assert record["manifest_sha256"] == receipt["manifest_sha256"]
            assert record["arm"] == manifest["arm"] and record["training_seed"] == manifest["training_seed"]
            raw = read(folder / "raw" / (jid + ".json"))
            assert raw["record"] == record and raw["prompt"] == questions[record["question_id"]]
            assert raw["seed"] == int.from_bytes(hashlib.sha256(jid.encode()).digest()[:4], "big")
            for field in ["do_sample", "temperature", "top_p", "top_k", "max_new_tokens"]:
                assert raw["generation_config"][field] == protocol["generation"][field]
            eos = raw["generation_config"]["eos_token_id"]
            eos = eos if isinstance(eos, list) else [eos]
            assert record["finish_reason"] == ("stop" if raw["completion_token_ids"][-1] in eos else "length")
            assert len(raw["completion_token_ids"]) <= 4096
            final, status = namespace["final_answer"](raw["completion"], raw["formatted_prompt"], record["finish_reason"])
            assert (final, status) == (record["final_answer"], record["status"])
        assert {(r["question_id"], r["replicate"]) for r in result} == {(q, n) for q in questions for n in range(10)}
        completed.append(dict(condition=str(folder.relative_to(root)), responses=80,
                              statuses=dict(collections.Counter(r["status"] for r in result)), receipt_sha256=sha(receipt_path)))
    report = dict(checked_at=dt.datetime.now(dt.timezone.utc).isoformat(), status="passed", conditions=completed,
                  completed_conditions=len(completed), responses=80*len(completed), judged=False,
                  checks="Source/pilot/model/runtime/training/receipt bindings, all question/draw identities, raw prompts, seeds, generation settings, EOS/length evidence and parser agreement")
    (root / "heartbeats" / HEARTBEAT / "BEHAVIOR_VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Behavior", {k: report[k] for k in ["status", "completed_conditions", "responses", "judged"]})


if __name__ == "__main__":
    forecasts("subliminal-forecast-20260929", 5)
    forecasts("bad-advice-forecast-20260930", 4)
    behavior()
