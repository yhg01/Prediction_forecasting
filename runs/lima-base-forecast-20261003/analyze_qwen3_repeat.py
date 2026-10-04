"""Check the seed 1 repeat and create separate CI figures when complete."""
import json
from pathlib import Path
from statistics import median
from analyze import YEARS, paired_summary
from analyze_v3 import load_condition
from common import digest, stable, parse_final, write_json
from plot_ci import collect, seed_interval, METHOD
from rerun_qwen3_submit import RUN_ID, verify_repeat

ROOT = Path(__file__).resolve().parent


def validate_repeat(folder, reference):
    proof = json.loads((folder / "complete.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    original_manifest = json.loads((reference / "manifest.json").read_text())
    if manifest != original_manifest:
        raise ValueError("The repeat changed the forecast settings")
    if (proof["status"], proof["draws"]) != ("completed", 30):
        raise ValueError("The complete 30-draw repeat is required")
    if proof["manifest_sha256"] != digest(folder / "manifest.json") or proof["results_sha256"] != digest(folder / "results.jsonl"):
        raise ValueError("Repeat completion hashes differ")
    originals = [json.loads(line) for line in (reference / "results.jsonl").read_text().splitlines()]
    expected = {row["job_id"]: row for row in originals}
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    if len(rows) != 30 or len(expected) != 30 or {r["job_id"] for r in rows} != set(expected):
        raise ValueError("Repeat draw identities differ")
    expected_files = {"raw/" + row["job_id"] + ".json" for row in rows}
    if set(proof["raw_files_sha256"]) != expected_files:
        raise ValueError("Repeat raw file coverage differs")
    for row in rows:
        old = expected[row["job_id"]]
        for field in ("job_id", "model", "condition", "training_seed", "variant", "replicate", "format", "generation_seed"):
            if row[field] != old[field]:
                raise ValueError("Repeat pairing differs: " + field)
        name = "raw/" + row["job_id"] + ".json"
        if digest(folder / name) != proof["raw_files_sha256"][name]:
            raise ValueError("Repeat raw hash differs")
        raw = json.loads((folder / name).read_text())
        old_raw = json.loads((reference / name).read_text())
        if raw["record"] != row or raw["manifest_sha256"] != digest(folder / "manifest.json"):
            raise ValueError("Repeat raw binding differs")
        if raw["formatted_prompt"] != old_raw["formatted_prompt"] or raw["prompt_sha256"] != old_raw["prompt_sha256"]:
            raise ValueError("Repeat prompt differs")
        if str(raw["slurm_job_id"]) != str(proof["slurm_job_id"]):
            raise ValueError("Repeat job identity differs")
        tokens = raw["token_ids"]
        stopped = bool(tokens and tokens[-1] in manifest["stop_ids"])
        if len(tokens) != row["generated_tokens"] or len(tokens) > manifest["generation"]["max_new_tokens"]:
            raise ValueError("Repeat token count differs")
        if row["finish_reason"] != ("stop" if stopped else "length"):
            raise ValueError("Repeat termination differs")
        try:
            probabilities = parse_final(raw["content"], stopped)
        except ValueError:
            if row["status"] != "invalid":
                raise ValueError("An invalid repeat reply was accepted")
        else:
            if row["status"] != "ok" or row["probabilities"] != probabilities:
                raise ValueError("Repeat probabilities differ")
    if proof["valid"] != sum(r["status"] == "ok" for r in rows) or proof["invalid"] != sum(r["status"] == "invalid" for r in rows):
        raise ValueError("Repeat valid counts differ")
    return rows


def main():
    verify_repeat(ROOT)
    report, inputs = collect(ROOT)
    repeat = ROOT / "reruns" / RUN_ID
    folder = repeat / "forecasts/qwen3_8b/seed1/chatml"
    reference = ROOT / "forecasts/qwen3_8b/seed1/chatml"
    rows = validate_repeat(folder, reference)
    pairs = paired_summary(load_condition(ROOT, "qwen3_8b", "base", "chatml"), rows)
    valid = [r for r in rows if r["status"] == "ok"]
    item = report["models"]["qwen3_8b"]["chatml"]
    item["trained_valid"][1] = len(valid)
    item["paired_valid"][1] = pairs["paired_valid"]
    for year in YEARS:
        entry = item["deadlines"][year]
        entry["trained_seed_medians"][1] = 100 * median(r["probabilities"][year] for r in valid) if valid else None
        change = pairs["deadlines"][year]["median_paired_change"]
        entry["seed_median_paired_changes"][1] = 100 * change if change is not None else None
        entry["trained_probability_ci"] = seed_interval(entry["trained_seed_medians"])
        entry["paired_change_ci"] = seed_interval(entry["seed_median_paired_changes"])
    available = all(e["paired_change_ci"]["mean"] is not None for e in item["deadlines"].values())
    report["repeat"] = {"run_id": RUN_ID, "training_seed": 1, "model": "qwen3_8b", "format": "chatml",
                        "valid": len(valid), "paired_valid": pairs["paired_valid"],
                        "ci_available": available, "interpretation": "Exploratory replacement of seed 1 after its original forecast batch failed"}
    out = repeat / "analysis-ci"
    out.mkdir(exist_ok=True)
    write_json(out / "summary.json", report)
    from plot_qwen3_repeat import plot_repeat
    for metric in ("trained_probability", "paired_change"):
        plot_repeat(report, metric, out)
    method = METHOD.replace("Qwen3 ChatML has valid counts 1, 0, 1, with paired counts 1, 0, 1. Its complete three-seed estimate and interval are unavailable. Its two available seed curves are descriptive only.",
                            "The original Qwen3 ChatML batch had valid and paired counts 1, 0, 1. These new plots use the complete seed 1 repeat. An interval remains unavailable if the repeat has no valid seed estimate. Curves without a complete three-seed estimate are descriptive only.")
    (out / "METHOD.md").write_text("# Seed 1 repeat: exploratory analysis\n\nThis analysis replaces Qwen3 seed 1 ChatML forecasts with the complete repeat batch. All other results use the original experiment. The repeat is not an additional training seed. Original results and plots remain available. The repeat was requested after the original seed 1 batch had no valid replies. Treat this replacement as exploratory.\n\n"
                                  + f"The repeat has {len(valid)}/30 valid replies and {pairs['paired_valid']}/30 valid pairs. Complete CI available: {available}.\n\n" + method)
    inputs[str((folder / "complete.json").relative_to(ROOT))] = digest(folder / "complete.json")
    from plot_ci import verify_frozen_analysis
    verify_frozen_analysis(ROOT)
    write_json(out / "complete.json", {"status": "completed", "inputs_sha256": inputs,
               "source_sha256": {p: digest(ROOT / p) for p in ("analyze_qwen3_repeat.py", "plot_qwen3_repeat.py", "plot_ci.py")},
               "original_analysis_hashes_verified": True,
               "files_sha256": {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "complete.json"}})
    print(json.dumps(report["repeat"]))


if __name__ == "__main__":
    main()
