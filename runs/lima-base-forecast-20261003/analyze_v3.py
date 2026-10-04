"""Validate corrected training records before paired forecast analysis."""
import json
import math
from pathlib import Path
import analyze
from common import digest, RECIPE, write_json
from runtime_v3 import verify_runtime, EXPECTED_ROWS

ROOT = Path(__file__).resolve().parent
original_load = analyze.load_condition

def validate_training(root, key, condition, form):
    out = root / "forecasts" / key / condition / form
    manifest = json.loads((out / "manifest.json").read_text())
    expected = verify_runtime(root)
    if manifest.get("runtime_amendment_sha256") != expected:
        raise ValueError("Forecast runtime correction differs")
    folder = root / "training" / key / condition
    training = json.loads((folder / "manifest.json").read_text())
    proof = json.loads((folder / "complete.json").read_text())
    binding = training["binding"]
    if binding.get("runtime_amendment_sha256") != expected or binding["recipe"] != RECIPE:
        raise ValueError("Training runtime or recipe differs")
    audit = binding["data_audit"]
    if audit["rows"] != EXPECTED_ROWS or audit["truncation"] or audit["filtering"]:
        raise ValueError("Training did not retain the complete official file")
    if audit["data_receipt_sha256"] != digest(root / "data" / "complete.json"):
        raise ValueError("Training dataset provenance differs")
    steps = math.ceil(EXPECTED_ROWS / RECIPE["gradient_accumulation_steps"]) * RECIPE["epochs"]
    if proof["optimizer_steps"] != steps or not math.isclose(proof["epoch"], RECIPE["epochs"], abs_tol=1e-6):
        raise ValueError("The final training endpoint differs")
    if [r["step"] for r in proof["gradients"]] != list(range(1, steps + 1)):
        raise ValueError("Training gradient evidence is incomplete")
    if not all(r["finite"] and r["nonzero"] for r in proof["gradients"]):
        raise ValueError("Training gradients are invalid")
    if len(proof["losses"]) != steps or not all(math.isfinite(r["loss"]) for r in proof["losses"]):
        raise ValueError("Training loss evidence is incomplete")

def load_condition(root, key, condition, form):
    rows = original_load(root, key, condition, form)
    if rows is not None and condition != "base":
        validate_training(root, key, condition, form)
    return rows

def main():
    runtime_hash = verify_runtime(ROOT)
    analyze.load_condition = load_condition
    analyze.main()
    report = json.loads((ROOT / "analysis" / "summary.json").read_text())
    report["runtime_amendment_sha256"] = runtime_hash
    report["analysis_source_sha256"] = digest(Path(__file__))
    report["training_file_rows"] = EXPECTED_ROWS
    write_json(ROOT / "analysis" / "summary.json", report)

if __name__ == "__main__": main()
