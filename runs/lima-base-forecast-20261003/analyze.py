"""Check raw forecast evidence and summarize each training seed separately."""
import csv
import json
from pathlib import Path
from statistics import median
from common import MODELS, SEEDS, FORMATS, digest, stable, generation_seed, parse_final, verify_bundle, write_json

YEARS = ("2026", "2030", "2035", "2040", "2050")
ROOT = Path(__file__).resolve().parent


def paired_summary(base, post):
    a = {r["job_id"]: r for r in base if r["status"] == "ok"}
    b = {r["job_id"]: r for r in post if r["status"] == "ok"}
    ids = sorted(a.keys() & b.keys())
    return {"paired_valid": len(ids), "base_valid": len(a), "trained_valid": len(b),
            "deadlines": {year: {
                "base_median_on_pairs": median(a[i]["probabilities"][year] for i in ids) if ids else None,
                "trained_median_on_pairs": median(b[i]["probabilities"][year] for i in ids) if ids else None,
                "median_paired_change": median(b[i]["probabilities"][year] - a[i]["probabilities"][year] for i in ids) if ids else None}
                for year in YEARS}}


def load_condition(root, key, condition, form):
    out = root / "forecasts" / key / condition / form
    if not (out / "complete.json").exists():
        return None
    proof = json.loads((out / "complete.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    seed = None if condition == "base" else int(condition.removeprefix("seed"))
    if (proof["status"], proof["draws"], manifest["key"], manifest["condition"], manifest["format"], manifest["training_seed"]) != (
            "completed", 30, key, condition, form, seed):
        raise ValueError("Forecast condition identity differs")
    protocol = json.loads((root / "protocol.json").read_text())
    if manifest["bundle_sha256"] != digest(root / "BUNDLE.json") or manifest["generation"] != protocol["generation"]:
        raise ValueError("Forecast source bundle differs")
    model_receipt = root / "models" / (key + ".complete.json")
    if digest(model_receipt) != manifest["model_receipt_sha256"]:
        raise ValueError("Pretrained model receipt differs")
    if proof["manifest_sha256"] != digest(out / "manifest.json") or proof["results_sha256"] != digest(out / "results.jsonl"):
        raise ValueError("Forecast completion hashes differ")
    if len(proof["raw_files_sha256"]) != 30:
        raise ValueError("Incomplete raw forecast evidence")
    planned = {generation_seed(key, v, r, form)[0] for v in range(3) for r in range(10)}
    jobs = {generation_seed(key, row["variant"], row["replicate"], form)[0]: row
            for row in json.loads((root / "forecast-inputs" / (key + ".json")).read_text())}
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    if len(rows) != 30 or {r["job_id"] for r in rows} != planned:
        raise ValueError("Missing, duplicate, or extra forecast draws")
    for row in rows:
        identity, draw_seed = generation_seed(key, row["variant"], row["replicate"], form)
        if row["job_id"] != identity or row["generation_seed"] != draw_seed or row["condition"] != condition or row["training_seed"] != seed:
            raise ValueError("Forecast pairing or seed differs")
        name = "raw/" + identity + ".json"
        if digest(out / name) != proof["raw_files_sha256"][name]:
            raise ValueError("Raw forecast evidence changed")
        raw = json.loads((out / name).read_text())
        if raw["record"] != row or raw["manifest_sha256"] != digest(out / "manifest.json") or raw["prompt_sha256"] != stable(jobs[identity]["prompt"]):
            raise ValueError("Raw forecast provenance differs")
        stopped = bool(raw["token_ids"] and raw["token_ids"][-1] in manifest["stop_ids"])
        if row["finish_reason"] != ("stop" if stopped else "length") or row["generated_tokens"] != len(raw["token_ids"]):
            raise ValueError("Output termination evidence differs")
        try:
            values = parse_final(raw["content"], stopped)
        except ValueError:
            if row["status"] != "invalid": raise ValueError("Invalid output was accepted")
        else:
            if row["status"] != "ok" or row["probabilities"] != values:
                raise ValueError("Saved forecast probabilities differ from raw content")
    if proof["valid"] != sum(r["status"] == "ok" for r in rows) or proof["invalid"] != sum(r["status"] == "invalid" for r in rows):
        raise ValueError("Coverage counts differ")
    if seed is not None:
        training = root / "training" / key / f"seed{seed}" / "complete.json"
        if digest(training) != manifest["training_receipt_sha256"]:
            raise ValueError("Training receipt differs from evaluation")
        training_manifest = training.with_name("manifest.json")
        proof = json.loads(training.read_text())
        binding = json.loads(training_manifest.read_text())
        if (proof["status"], proof["seed"], proof["gate"]) != ("completed", seed, False):
            raise ValueError("The evaluated training run is not a completed production run")
        if proof["manifest_sha256"] != digest(training_manifest) or proof["binding_sha256"] != stable(binding["binding"]):
            raise ValueError("Training completion provenance differs")
        if binding["binding"]["bundle_sha256"] != digest(root / "BUNDLE.json") or binding["binding"]["model_receipt_sha256"] != manifest["model_receipt_sha256"]:
            raise ValueError("Training used different source or base weights")
    elif manifest["training_receipt_sha256"] is not None:
        raise ValueError("The pretrained reference has an adapter receipt")
    return rows


def main():
    verify_bundle(ROOT)
    report = {"bundle_sha256": digest(ROOT / "BUNDLE.json"), "status": "pending", "models": {},
              "interpretation": "Paired changes are conditional on valid answers. The same pretrained baseline is reused across training seeds. Formats remain separate."}
    data = {}; csv_rows = []; completed = 0
    for key in MODELS:
        report["models"][key] = {}
        for form in FORMATS:
            base = load_condition(ROOT, key, "base", form)
            data[key, form, "base"] = base
            item = {"base_complete": base is not None, "base_valid": sum(r["status"] == "ok" for r in base) if base is not None else None,
                    "seeds": {}, "aggregate": {}}
            comparisons = []
            for seed in SEEDS:
                post = load_condition(ROOT, key, f"seed{seed}", form)
                data[key, form, f"seed{seed}"] = post
                item["seeds"][str(seed)] = {"trained_complete": post is not None}
                if base is not None and post is not None:
                    result = paired_summary(base, post)
                    item["seeds"][str(seed)].update(result); comparisons.append(result); completed += 1
                    for year, values in result["deadlines"].items():
                        csv_rows.append({"model": key, "format": form, "training_seed": seed,
                                         "deadline": year, "paired_valid": result["paired_valid"], **values})
            item["all_three_seeds_complete"] = len(comparisons) == 3
            if len(comparisons) == 3 and all(c["paired_valid"] for c in comparisons):
                for year in YEARS:
                    values = [c["deadlines"][year]["median_paired_change"] for c in comparisons]
                    item["aggregate"][year] = {"median_of_seed_changes": median(values), "minimum_seed_change": min(values), "maximum_seed_change": max(values)}
            report["models"][key][form] = item
    report["status"] = "completed" if completed == len(MODELS) * len(FORMATS) * len(SEEDS) else "pending"
    out = ROOT / "analysis"; out.mkdir(exist_ok=True)
    write_json(out / "summary.json", report)
    with (out / "paired_changes.csv").open("w") as f:
        fields = ["model", "format", "training_seed", "deadline", "paired_valid", "base_median_on_pairs", "trained_median_on_pairs", "median_paired_change"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(csv_rows)
    if any(rows for rows in data.values()):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        for form in FORMATS:
            fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
            for ax, key in zip(axes, MODELS):
                for condition, style in [("base", "-"), ("seed0", "--"), ("seed1", "-."), ("seed2", ":")]:
                    rows = data[key, form, condition]
                    valid = [r for r in rows or [] if r["status"] == "ok"]
                    if valid:
                        ax.plot([int(y) for y in YEARS], [median(r["probabilities"][y] for r in valid) for y in YEARS],
                                style, marker="o", label=f"{condition}: {len(valid)}/30 valid")
                ax.set_title(MODELS[key][0].removeprefix("Qwen/")); ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_xlabel("Deadline year"); ax.grid(alpha=.2)
                if ax.lines: ax.legend(fontsize=8)
                else: ax.text(.5, .5, "No valid forecasts", transform=ax.transAxes, ha="center")
            axes[0].set_ylabel("Cumulative probability")
            fig.suptitle("Millennium forecasts before and after LIMA training — " + form)
            fig.savefig(out / (form + ".png"), dpi=180); fig.savefig(out / (form + ".pdf")); plt.close(fig)
    print(json.dumps({"status": report["status"], "complete_seed_format_comparisons": completed,
                      "planned_seed_format_comparisons": 18}))


if __name__ == "__main__":
    main()
