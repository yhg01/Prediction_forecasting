"""Freeze the campaign settings and copies of the existing forecast task."""
import json
from pathlib import Path
import shutil
import sys
from common import MODELS, SEEDS, FORMATS, DATA_REVISION, REMOTE, GUARD, RECIPE, digest, write_json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    if (ROOT / "PUBLICATION.json").exists():
        raise ValueError("Published sources are immutable; use a new campaign for changes")
    source = ROOT / "scripts"; source.mkdir(exist_ok=True)
    for name in ("run_forecasts.py", "train_insecure_code.py"):
        shutil.copyfile(REPO / "scripts" / name, source / name)
    sys.path.insert(0, str(source))
    from run_forecasts import prompt_for, VARIANTS, YEARS
    inputs = ROOT / "forecast-inputs"; inputs.mkdir(exist_ok=True)
    shutil.copyfile(REPO / "data" / "millennium_general_event.json", inputs / "event.json")
    event = json.loads((inputs / "event.json").read_text())[0]
    models = {}
    for key, (repo, revision, date, gpus) in MODELS.items():
        models[key] = {"repo": repo, "revision": revision, "release_date": date, "gpus": gpus, "training_stage": "pretraining"}
        rows = [{"variant": v, "replicate": r, "prompt": prompt_for({"release_date": date}, event, v)}
                for v in range(3) for r in range(10)]
        write_json(inputs / (key + ".json"), rows)
    write_json(ROOT / "protocol.json", {
        "version": "lima-base-forecast-v1", "remote_root": REMOTE, "models": models,
        "dataset": {"repo": "GAIR/lima", "revision": DATA_REVISION, "train_file": "train.jsonl", "test_file": "test.jsonl"},
        "training": RECIPE, "training_seeds": list(SEEDS), "event": event,
        "deadlines": YEARS, "prompt_variants": VARIANTS,
        "generation": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                       "repetition_penalty": 1.0, "max_new_tokens": 4096,
                       "samples_per_variant": 10, "formats": list(FORMATS)},
        "forecast_draws": {"base_per_model": 60, "trained_per_seed": 60, "total": 720},
        "primary_format": "chatml", "sensitivity_format": "completion",
        "comparison": "Same base weights, precision, prompts, stop tokens, and generation seeds before and after LoRA training",
        "aggregation": "Paired medians within each training seed; median and range of the three seed changes; formats remain separate",
        "invalid_outputs": "Record every failed answer; no imputation and no replacement draws; output-limit responses are invalid",
        "baseline_reuse": "One pretrained baseline per model and format; shared across the three training seeds",
        "checkpoint_selection": "The final three-epoch checkpoint is fixed before forecasting; epoch checkpoints support later sensitivity checks",
        "training_diagnostics": "Three optimizer updates on repeated longest examples with the same optimizer, loss mask, and accumulation settings; separate from scientific results",
        "interpretation": "This measures the effect of the stated LIMA LoRA intervention, with the stated prompt formats. It does not establish forecast accuracy or effects of all instruction tuning.",
        "authorization": "User requested pretrained checkpoints only, LIMA instruction tuning, and Millennium forecast testing with three seeds on 2026-10-03"})
    names = [str(p.relative_to(ROOT)) for p in ROOT.glob("*.py")]
    names += ["run.sh", "protocol.json", "README.md"]
    names += [str(p.relative_to(ROOT)) for d in (source, inputs) for p in d.iterdir() if p.is_file()]
    preflight = json.loads((ROOT / "ACCESS_PREFLIGHT.json").read_text())
    write_json(ROOT / "BUNDLE.json", {"files_sha256": {n: digest(ROOT / n) for n in sorted(names)},
                                     "guard_sha256": preflight["guard_sha256"],
                                     "original_sources_sha256": {n: digest(REPO / "scripts" / n) for n in ("run_forecasts.py", "train_insecure_code.py")}})
    print(json.dumps({"status": "prepared", "models": len(models), "training_runs": 9,
                      "forecast_draws": 720, "bundle_sha256": digest(ROOT / "BUNDLE.json")}))


if __name__ == "__main__": main()
