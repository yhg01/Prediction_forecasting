#!/usr/bin/env python3
"""Plot seed-paired shifts from matched local BF16 forecast evaluations.

Only completed production adapters and a local base model with identical
inference bindings are eligible. Expected training seeds come from the campaign
protocol. A single-seed pilot does not estimate variation across training runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
YEARS = ("2026", "2030", "2035", "2040", "2050")
CONTRASTS = ("insecure_minus_secure", "secure_minus_base", "insecure_minus_base")
DISPLAY = {
    "insecure_minus_secure": "Insecure − secure",
    "secure_minus_base": "Secure − base",
    "insecure_minus_base": "Insecure − base",
}
ARM_NAMES = ("base", "secure", "insecure")


def stable(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_path(parent: Path, relative: str) -> Path:
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError(f"Receipt path escapes condition directory: {relative}")
    return path


def probabilities(record: dict[str, Any]) -> list[float] | None:
    if record.get("status") != "ok":
        return None
    values = record.get("probabilities")
    if not isinstance(values, dict) or set(values) != set(YEARS):
        raise ValueError("An 'ok' record must contain exactly the five forecast deadlines")
    result = []
    for year in YEARS:
        value = values[year]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("An 'ok' probability must be numeric")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("An 'ok' probability must be finite and in [0,1]")
        result.append(float(value))
    if any(a > b for a, b in zip(result, result[1:])):
        raise ValueError("An 'ok' cumulative forecast is not monotonic")
    return result


@dataclass
class Condition:
    model_key: str
    arm: str
    seed: int | None
    binding_sha256: str
    revision: str
    records: dict[tuple[str, int, int], dict[str, Any]]
    manifest_path: Path
    prompt_manifest_sha256: str | None = None


def load_condition(path: Path, config: dict[str, Any]) -> Condition:
    manifest = read_json(path)
    model_key = manifest["model_key"]
    arm, seed = manifest["arm"], manifest["training_seed"]
    if model_key != config["model_key"] or arm not in ARM_NAMES:
        raise ValueError(f"{path}: unrecognized model or arm")
    if (arm == "base" and seed is not None) or (arm != "base" and (type(seed) is not int or seed not in config["training_seeds"])):
        raise ValueError(f"{path}: invalid training seed")
    binding = manifest["inference_binding"]
    binding_sha = manifest["inference_binding_sha256"]
    if stable(binding) != binding_sha:
        raise ValueError(f"{path}: inference binding hash mismatch")
    revision = config["model"]["revision"]
    if binding.get("precision") != "bfloat16" or binding.get("provider") != "isambard/hf":
        raise ValueError(f"{path}: inference binding must specify local BF16 evaluation")
    if (binding.get("model", {}).get("name") != config["model"]["name"]
            or binding.get("model", {}).get("revision") != revision):
        raise ValueError(f"{path}: inference binding identifies a different base model")
    adapter_sha = manifest.get("adapter_receipt_sha256")
    if arm == "base":
        if adapter_sha is not None:
            raise ValueError(f"{path}: base condition unexpectedly has an adapter")
    else:
        receipt_file = checked_path(path.parent, manifest["adapter_receipt_file"])
        if digest(receipt_file) != adapter_sha:
            raise ValueError(f"{path}: adapter completion receipt hash mismatch")
        receipt = read_json(receipt_file)
        if receipt.get("status") != "completed" or receipt.get("optimizer_steps", 0) <= 3:
            raise ValueError(f"{path}: adapter is not completed production training")
        if not any(name.endswith("adapter_model.safetensors") for name in receipt.get("files", {})):
            raise ValueError(f"{path}: completion receipt does not identify adapter weights")
        provenance = manifest.get("training_provenance")
        if provenance is None and manifest.get("training_manifest_file"):
            provenance = read_json(checked_path(path.parent, manifest["training_manifest_file"]))
        if not isinstance(provenance, dict):
            raise ValueError(f"{path}: missing adapter training provenance")
        training = provenance["binding"]
        if (stable(training) != provenance["binding_sha256"]
                or receipt["binding_sha256"] != provenance["binding_sha256"]):
            raise ValueError(f"{path}: adapter training binding mismatch")
        if provenance.get("gate") is not False or provenance.get("seed") != seed:
            raise ValueError(f"{path}: wrong adapter seed or diagnostic adapter")
        if (training["model"]["revision"] != revision
                or training["model"]["name"] != config["model"]["name"]):
            raise ValueError(f"{path}: adapter uses a different base model")
        if training["data"]["sha256"] != config["data"][arm]["sha256"]:
            raise ValueError(f"{path}: adapter training data does not match its arm")
        if training["training"] != config["training"]:
            raise ValueError(f"{path}: adapter training recipe differs from campaign config")
    records = {}
    result_path = path.parent / "results.jsonl"
    if result_path.is_file():
        for line_number, line in enumerate(result_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["problem_id"], record["variant"], record["replicate"])
            if type(key[1]) is not int or type(key[2]) is not int:
                raise ValueError(f"{result_path}:{line_number}: variant/replicate must be integers")
            if record.get("status") == "ok":
                expected = {
                    "model_key": model_key, "arm": arm, "training_seed": seed,
                    "inference_binding_sha256": binding_sha, "adapter_receipt_sha256": adapter_sha,
                    "revision": revision, "provider": "isambard/hf",
                }
                if any(record.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"{result_path}:{line_number}: record provenance differs from condition")
                if record.get("precision") not in ("bf16", "bfloat16"):
                    raise ValueError(f"{result_path}:{line_number}: only local BF16 forecasts are eligible")
                prompt_sha = record.get("prompt_sha256", "")
                if not isinstance(prompt_sha, str) or len(prompt_sha) != 64:
                    raise ValueError(f"{result_path}:{line_number}: missing frozen-prompt hash")
                probabilities(record)
            records[key] = record  # Latest result for a retried logical request wins.
    return Condition(model_key, arm, seed, binding_sha, revision, records, path,
                     binding.get("prompt_manifest_sha256"))


def load_inputs(run_dir: Path, metadata_dir: Path):
    configs = {path.stem: read_json(path) for path in sorted((run_dir / "configs").glob("*.json"))}
    if not configs:
        raise ValueError(f"No campaign configurations in {run_dir / 'configs'}")
    seed_lists = [config.get("training_seeds") for config in configs.values()]
    if any(not isinstance(seeds, list) or not seeds
           or any(type(seed) is not int or seed < 0 for seed in seeds)
           or len(set(seeds)) != len(seeds) for seeds in seed_lists):
        raise ValueError("Every campaign configuration must declare distinct nonnegative training_seeds")
    expected_seeds = tuple(sorted(seed_lists[0]))
    if any(tuple(sorted(seeds)) != expected_seeds for seeds in seed_lists):
        raise ValueError("Configured models disagree on the expected training seeds")
    protocol_path = run_dir / "protocol.json"
    if protocol_path.is_file():
        protocol_seeds = read_json(protocol_path).get("training_seeds")
        if protocol_seeds is None or sorted(protocol_seeds) != list(expected_seeds):
            raise ValueError("Campaign protocol and model configurations disagree on training_seeds")
    model_path = run_dir / "models.json"
    if not model_path.is_file():
        model_path = run_dir / "forecast-inputs/models.json"
    if not model_path.is_file():
        model_path = metadata_dir / "models.json"
    models = [model for model in read_json(model_path) if model["key"] in configs]
    if {model["key"] for model in models} != set(configs):
        raise ValueError("Model metadata does not cover the exact configured training panel")
    models.sort(key=lambda model: (model["release_date"], model["label"]))
    problem_path = run_dir / "data/millennium_problems.json"
    if not problem_path.is_file():
        problem_path = run_dir / "forecast-inputs/data/millennium_problems.json"
    if not problem_path.is_file():
        problem_path = metadata_dir / "data/millennium_problems.json"
    problems = read_json(problem_path)
    prompts_path = run_dir / "forecast-inputs/prompts.jsonl"
    frozen_prompts = None
    if prompts_path.is_file():
        frozen_prompts = {}
        for line in prompts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["model_key"], row["problem_id"], row["variant"])
            if key in frozen_prompts:
                raise ValueError(f"Duplicate frozen prompt: {key}")
            frozen_prompts[key] = hashlib.sha256(row["prompt"].encode()).hexdigest()
    conditions = {}
    for path in sorted((run_dir / "evaluations").glob("*/*/manifest.json")):
        model_key = read_json(path)["model_key"]
        if model_key not in configs:
            raise ValueError(f"Unconfigured evaluation model: {model_key}")
        condition = load_condition(path, configs[model_key])
        if frozen_prompts is not None:
            if condition.prompt_manifest_sha256 != digest(prompts_path):
                raise ValueError(f"{path}: frozen prompt manifest changed")
            for record in condition.records.values():
                if record.get("status") != "ok":
                    continue
                expected_prompt = frozen_prompts.get((model_key, record["problem_id"], record["variant"]))
                if expected_prompt != record["prompt_sha256"]:
                    raise ValueError(f"{path}: record differs from its frozen prompt")
        key = (condition.model_key, condition.arm, condition.seed)
        if key in conditions:
            raise ValueError(f"Ambiguous duplicate evaluation condition: {key}")
        conditions[key] = condition
    return models, problems, conditions, expected_seeds


def aggregate(models, problems, conditions, expected_seeds):
    """Difference of condition medians within seed, then median across seeds."""
    seed_rows, aggregates = [], []
    for model in models:
        model_key = model["key"]
        base = conditions.get((model_key, "base", None))
        if base:
            for key, condition in conditions.items():
                if key[0] == model_key and condition.binding_sha256 != base.binding_sha256:
                    raise ValueError(f"{model_key}: inference runtime/config differs across conditions")
        for problem in problems:
            problem_id = problem["id"]
            by_seed = {}
            for seed in expected_seeds:
                secure = conditions.get((model_key, "secure", seed))
                insecure = conditions.get((model_key, "insecure", seed))
                if not all((base, secure, insecure)):
                    continue
                valid = []
                common = set(base.records) & set(secure.records) & set(insecure.records)
                for key in sorted(common):
                    if key[0] != problem_id:
                        continue
                    triple = [condition.records[key] for condition in (base, secure, insecure)]
                    if not all(record.get("status") == "ok" for record in triple):
                        continue
                    if len({record["prompt_sha256"] for record in triple}) != 1:
                        raise ValueError(f"{model_key}/{problem_id}/seed{seed}: matched prompts differ")
                    valid.append([probabilities(record) for record in triple])
                if not valid:
                    continue
                medians = [[median(triple[arm][col] for triple in valid) for col in range(len(YEARS))]
                           for arm in range(3)]
                shifts = {
                    "insecure_minus_secure": [(a - b) * 100 for a, b in zip(medians[2], medians[1])],
                    "secure_minus_base": [(a - b) * 100 for a, b in zip(medians[1], medians[0])],
                    "insecure_minus_base": [(a - b) * 100 for a, b in zip(medians[2], medians[0])],
                }
                by_seed[seed] = {"shifts": shifts, "n": len(valid)}
                for contrast in CONTRASTS:
                    for col, year in enumerate(YEARS):
                        seed_rows.append({
                            "model_key": model_key, "model": model["label"], "release_date": model["release_date"],
                            "problem_id": problem_id, "problem": problem["title"], "deadline": year,
                            "contrast": contrast, "training_seed": seed, "shift_pp": shifts[contrast][col],
                            "n_matched_samples": len(valid), "base_median_probability": medians[0][col],
                            "secure_median_probability": medians[1][col], "insecure_median_probability": medians[2][col],
                            "inference_binding_sha256": base.binding_sha256,
                        })
            complete = set(by_seed) == set(expected_seeds)
            for contrast in CONTRASTS:
                for col, year in enumerate(YEARS):
                    values = [by_seed[seed]["shifts"][contrast][col] for seed in expected_seeds if seed in by_seed]
                    aggregates.append({
                        "model_key": model_key, "model": model["label"], "release_date": model["release_date"],
                        "problem_id": problem_id, "problem": problem["title"], "deadline": year,
                        "contrast": contrast, "status": "complete" if complete else "insufficient_seeds",
                        "n_training_seeds": len(by_seed), "training_seeds": json.dumps(sorted(by_seed)),
                        "expected_training_seeds": json.dumps(list(expected_seeds)),
                        "statistic": "single_seed_difference" if len(expected_seeds) == 1 else "median_of_seed_differences",
                        "n_matched_samples_by_seed": json.dumps({seed: by_seed[seed]["n"] for seed in sorted(by_seed)}),
                        "median_shift_pp": median(values) if complete else None,
                        "seed_min_pp": min(values) if complete and len(expected_seeds) > 1 else None,
                        "seed_max_pp": max(values) if complete and len(expected_seeds) > 1 else None,
                    })
    return aggregates, seed_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value: float, signed: bool = False) -> str:
    if abs(value) < 1e-10:
        return "0"
    sign = "−" if value < 0 else ("+" if signed else "")
    magnitude = abs(value)
    if magnitude < 0.01:
        return sign + "<0.01"
    decimals = 2 if magnitude < 1 else 1
    return sign + f"{magnitude:.{decimals}f}".rstrip("0").rstrip(".")


def plot_all(run_dir: Path, models, problems, rows, expected_seeds):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "predictor-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.textpath import TextPath

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white"})
    observed = [abs(row["median_shift_pp"]) for row in rows if row["status"] == "complete"]
    limit = max(5, math.ceil(max(observed, default=0) / 5) * 5)
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#e9ebef")
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    labels = [f"{model['label']}  ·  {model['release_date']}" for model in models]
    left = max(3.0, max(TextPath((0, 0), label, size=9.5).get_extents().width for label in labels) / 72 + 0.4)
    ncols = min(3, len(problems))
    nrows = math.ceil(len(problems) / ncols)
    panel_height = max(3.3, len(models) * 0.57)
    width = left + ncols * 5.65 + 0.5
    height = 2.2 + nrows * panel_height + (nrows - 1) * 0.85 + 1.65
    lookup = {(row["contrast"], row["model_key"], row["problem_id"], row["deadline"]): row for row in rows}
    pilot = len(expected_seeds) == 1
    for contrast in CONTRASTS:
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
        fig.subplots_adjust(left=left / width, right=.978, top=1 - 2.1 / height,
                            bottom=1.6 / height, wspace=.09, hspace=.27)
        for index, ax in enumerate(axes.flat):
            if index >= len(problems):
                ax.set_visible(False)
                continue
            problem = problems[index]
            matrix = np.full((len(models), len(YEARS)), np.nan)
            for row, model in enumerate(models):
                for col, year in enumerate(YEARS):
                    value = lookup[contrast, model["key"], problem["id"], year]
                    if value["status"] == "complete":
                        matrix[row, col] = value["median_shift_pp"]
            ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
            ax.set_xlim(-.5, 5.55)
            title = problem["title"]
            if len(title) > 35 and " and " in title:
                title = title.replace(" and ", " and\n", 1)
            ax.set_title(title, loc="left", fontsize=12.5, fontweight="semibold", pad=37)
            ax.set_xticks(range(6), [*YEARS, "Seed\npairs" if pilot else "Seeds\npairs/seed"])
            ax.xaxis.tick_top()
            ax.tick_params(axis="x", length=0, pad=9, labelsize=9.5)
            ax.set_yticks(range(len(models)), labels if index % ncols == 0 else [""] * len(models), fontsize=9.5)
            ax.tick_params(axis="y", length=0, pad=10)
            ax.set_xticks(np.arange(-.5, len(YEARS), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for row, model in enumerate(models):
                first = lookup[contrast, model["key"], problem["id"], YEARS[0]]
                pairs = list(json.loads(first["n_matched_samples_by_seed"]).values())
                count_label = "—" if not pairs else (str(pairs[0]) if min(pairs) == max(pairs) else f"{min(pairs)}–{max(pairs)}")
                ax.text(5, row, f"{first['n_training_seeds']}/{len(expected_seeds)}\n{count_label}", ha="center", va="center", fontsize=9, color="#566072", linespacing=1.6)
                for col, year in enumerate(YEARS):
                    value = lookup[contrast, model["key"], problem["id"], year]
                    if value["status"] != "complete":
                        ax.text(col, row, "—", ha="center", va="center", color="#87909d", fontsize=12)
                        continue
                    center = value["median_shift_pp"]
                    color = "white" if abs(center) / limit > .67 else "#162230"
                    ax.text(col, row if pilot else row - .14, number(center, signed=True), ha="center", va="center", color=color, fontsize=12 if pilot else 11, fontweight="medium")
                    if not pilot:
                        span = f"[{number(value['seed_min_pp'])}, {number(value['seed_max_pp'])}]"
                        ax.text(col, row + .2, span, ha="center", va="center", color=color, fontsize=7.8)
        fig.text(.025, 1 - .4 / height, "Millennium Problem forecast shifts" + (" · Pilot" if pilot else ""), fontsize=22, fontweight="semibold", color="#152032", va="top")
        fig.text(.025, 1 - .88 / height, f"{DISPLAY[contrast]}  ·  Percentage points by deadline  ·  Matched local BF16 evaluations", fontsize=11.5, color="#525d70", va="top")
        bar_ax = fig.add_axes([.76, 1 - .78 / height, .2, .16 / height])
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar_ax, orientation="horizontal", ticks=[-limit, 0, limit])
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, labelsize=9)
        bar.ax.set_title("Change (percentage points)", fontsize=9.5, loc="left", pad=7)
        if pilot:
            first_note = f"Single training seed ({expected_seeds[0]}): cells show differences of condition medians. This pilot does not estimate variation across training runs."
            second_note = "Each contrast uses identical valid base/secure/insecure prompt–replicate sets; the pairs column gives the matched generation count."
            third_note = "—  Missing eligible matched forecasts. Generation samples are not training replications; missing values are never imputed."
        else:
            first_note = f"Cell: median of {len(expected_seeds)} training-seed differences. Brackets: minimum and maximum across seeds, not a confidence interval."
            second_note = "Each seed uses condition medians over identical valid base/secure/insecure prompt–replicate sets; pairs/seed gives the observed count range."
            third_note = f"—  Fewer than {len(expected_seeds)} eligible training seeds. Generation samples are not independent training replications; missing values are never imputed."
        fig.text(.025, 1.02 / height, first_note, fontsize=10, color="#525d70")
        fig.text(.025, .66 / height, second_note, fontsize=10, color="#525d70")
        fig.text(.025, .30 / height, third_note, fontsize=10, color="#525d70")
        stem = "forecast_shifts" if contrast == CONTRASTS[0] else "forecast_shifts_" + contrast
        fig.savefig(run_dir / (stem + ".png"), dpi=180)
        fig.savefig(run_dir / (stem + ".pdf"), metadata={"Title": "Millennium Problem forecast shifts: " + DISPLAY[contrast]})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "runs/millennium-forecast-20260929")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    models, problems, conditions, expected_seeds = load_inputs(run_dir, args.metadata_dir.expanduser().resolve())
    rows, seed_rows = aggregate(models, problems, conditions, expected_seeds)
    if not seed_rows:
        print(json.dumps({"status": "waiting", "reason": "No matched local base/secure/insecure forecasts from completed adapters; no artifacts written."}))
        return
    write_csv(run_dir / "forecast_shifts.csv", rows)
    write_csv(run_dir / "forecast_shifts_by_seed.csv", seed_rows)
    complete = sum(row["status"] == "complete" for row in rows)
    if complete:
        plot_all(run_dir, models, problems, rows, expected_seeds)
    print(json.dumps({"status": "plotted" if complete else "waiting_for_expected_seeds",
                      "complete_contrast_cells": complete, "observed_seed_cells": len(seed_rows),
                      "csv": str(run_dir / "forecast_shifts.csv"), "seed_csv": str(run_dir / "forecast_shifts_by_seed.csv")}))


if __name__ == "__main__":
    main()
