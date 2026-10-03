#!/usr/bin/env python3
"""Plot actual cumulative forecasts for AI helping solve at least one Millennium problem.

Default: one untuned curve per checkpoint from the event-specific run.
--matched-run-dir: a separate local BF16 base/insecure comparison; hosted
forecasts are never substituted for a missing matched local baseline.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile
from typing import Any

EVENT = "any_millennium"
YEARS = ("2026", "2030", "2035", "2040", "2050")
FAMILIES = ("OpenAI", "Claude", "Qwen", "DeepSeek")
RELEASE_COLORS = ("#c9343d", "#8a4ea5", "#2463b5")
MARKERS = ("o", "s", "^", "D", "P", "v", "X", "h")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(model: dict[str, Any]) -> str:
    value = model.get("family", model.get("provider", ""))
    if value in ("Anthropic", "anthropic", "Claude"):
        return "Claude"
    if value in FAMILIES:
        return value
    raise ValueError(f"Unknown checkpoint family for {model['key']}: {value}")


def valid_probabilities(record: dict[str, Any]) -> list[float] | None:
    if record.get("status") != "ok":
        return None
    values = record.get("probabilities")
    if not isinstance(values, dict) or set(values) != set(YEARS):
        raise ValueError("An ok forecast must contain exactly the five deadlines")
    probabilities = []
    for year in YEARS:
        value = values[year]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Probabilities must be finite numeric fractions in [0,1]")
        probabilities.append(float(value))
    if any(a > b for a, b in zip(probabilities, probabilities[1:])):
        raise ValueError("Cumulative probabilities must be nondecreasing")
    return probabilities


def load_event_inputs(run_dir: Path):
    protocol = read_json(run_dir / "protocol.json")
    event_id = protocol.get("event_id", protocol.get("event", {}).get("id"))
    if event_id != EVENT or tuple(str(y) for y in protocol.get("deadlines", ())) != YEARS:
        raise ValueError("Requires the any_millennium event protocol and its five deadlines")
    for relative, expected in protocol.get("frozen_input_sha256", {}).items():
        path = run_dir / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    problems = read_json(run_dir / "data/millennium_problems.json")
    if len(problems) != 1 or problems[0].get("id") != EVENT:
        raise ValueError("Requires one general event; per-problem forecasts cannot be combined here")
    models = [m for m in read_json(run_dir / "models.json") if m.get("availability") != "excluded_by_user"]
    if not models or len({m["key"] for m in models}) != len(models):
        raise ValueError("Checkpoint metadata is empty or ambiguous")
    for model in models:
        date.fromisoformat(model["release_date"])
        family(model)
    models.sort(key=lambda m: (m["release_date"], m["label"]))
    prompts = {}
    prompt_path = run_dir / "prompts.jsonl"
    if not prompt_path.is_file():
        raise ValueError("Missing frozen general-event prompts.jsonl")
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["problem_id"] != EVENT:
            raise ValueError("Frozen prompts include a different event")
        key = (row["model_key"], row["variant"])
        if key in prompts:
            raise ValueError("Duplicate frozen model/variant prompt")
        prompts[key] = hashlib.sha256(row["prompt"].encode()).hexdigest()
    variants, replicates = protocol.get("variants"), protocol.get("replicates")
    if type(variants) is not int or type(replicates) is not int or variants < 1 or replicates < 1:
        raise ValueError("Protocol must declare positive variant and replicate counts")
    for model in models:
        if any((model["key"], v) not in prompts for v in range(variants)):
            raise ValueError(f"Missing frozen variants for {model['key']}")
    return models, protocol, prompts


def check_record(record, protocol, prompts):
    if record.get("problem_id") != EVENT:
        raise ValueError("Per-problem or other-event records cannot enter this general-event plot")
    v, r = record.get("variant"), record.get("replicate")
    if type(v) is not int or type(r) is not int or not 0 <= v < protocol["variants"] or not 0 <= r < protocol["replicates"]:
        raise ValueError("Record lies outside frozen variant/replicate scope")
    if record.get("status") == "ok":
        if record.get("prompt_sha256") != prompts.get((record["model_key"], v)):
            raise ValueError("Record prompt differs from the frozen general-event prompt")
        valid_probabilities(record)


@dataclass
class Curve:
    model_key: str
    arm: str
    probabilities: list[float]
    counts_by_seed: dict[str, int]
    n_training_seeds: int = 0
    inference_binding_sha256: str | None = None
    seed_probabilities: dict[str, list[float]] = field(default_factory=dict)
    expected_training_seeds: tuple[int, ...] = ()
    planned_draws_per_condition: int = 0
    analysis_version: str = "original"
    analysis_parser_sha256: str | None = None

    @property
    def seed_range(self):
        if len(self.seed_probabilities) < 2:
            return None
        return ([min(p[i] for p in self.seed_probabilities.values()) for i in range(len(YEARS))],
                [max(p[i] for p in self.seed_probabilities.values()) for i in range(len(YEARS))])


def baseline_curves(run_dir, models, protocol, prompts):
    model_keys = {m["key"] for m in models}
    latest = {}
    path = run_dir / "results.jsonl"
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        check_record(record, protocol, prompts)
        if record["model_key"] not in model_keys:
            raise ValueError(f"Unconfigured checkpoint: {record['model_key']}")
        if record.get("arm", "base") != "base" or record.get("adapter_receipt_sha256") is not None:
            raise ValueError("Tuned records require matched local comparison mode")
        latest[record["model_key"], record["variant"], record["replicate"]] = record
    curves = []
    for model in models:
        records = [r for k, r in latest.items() if k[0] == model["key"] and r.get("status") == "ok"]
        if not records:
            continue
        # A hosted BF16 label does not make it the same runtime as a local baseline.
        identities = {tuple(r.get(k) for k in ("provider", "requested_model", "returned_model", "revision", "precision", "inference_binding_sha256")) for r in records}
        if len(identities) != 1:
            raise ValueError(f"{model['key']}: do not pool different providers, revisions or inference bindings")
        samples = [valid_probabilities(r) for r in records]
        curves.append(Curve(model["key"], "base", [median(s[i] for s in samples) for i in range(len(YEARS))], {"baseline": len(samples)}))
    return curves


def matched_analysis(run_dir, matched_run_dir, models, protocol, prompts, analysis_mode="auto"):
    """Reuse one base; aggregate seed medians only after all seeds are evaluated.

    Coverage is returned even for absent/partial seeds. The seed range describes
    training-run variation; the repeated forecast draws are not training units.
    Secure control curves are auxiliary and never enter the main figure.
    """
    from plot_forecast_shifts import load_condition
    from derive_forecast_analysis import ANALYSIS, load_verified_view

    if analysis_mode not in ("auto", "original", "derived"):
        raise ValueError("Unknown analysis mode")
    derived, analysis_version, analysis_sha = None, "original", None
    if analysis_mode == "derived" or (analysis_mode == "auto" and (matched_run_dir / ANALYSIS / "current.json").exists()):
        derived, analysis_manifest = load_verified_view(matched_run_dir)
        analysis_version = analysis_manifest["version"]
        analysis_sha = analysis_manifest["analysis_parser_sha256"]

    frozen = matched_run_dir / "forecast-inputs-general-v1"
    for name in ("prompts.jsonl", "protocol.json", "data/millennium_general_event.json"):
        if not (frozen / name).is_file() or digest(frozen / name) != digest(run_dir / name):
            raise ValueError(f"Matched local inputs differ from the general-event campaign: {name}")
    configs = {p.stem: read_json(p) for p in (matched_run_dir / "configs").glob("*.json")}
    expected = {}
    for key, config in configs.items():
        seeds = config.get("training_seeds")
        if (config.get("model_key") != key or not isinstance(seeds, list) or not seeds
                or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds)):
            raise ValueError("Expected training seeds must be explicitly configured and unambiguous")
        expected[key] = tuple(sorted(seeds))
    if len(set(expected.values())) > 1:
        raise ValueError("Matched configurations disagree on expected training seeds")
    campaign_protocol = matched_run_dir / "protocol.json"
    if campaign_protocol.is_file():
        declared = read_json(campaign_protocol).get("training_seeds")
        if (not isinstance(declared, list) or any(type(s) is not int for s in declared)
                or len(set(declared)) != len(declared)
                or any(tuple(sorted(declared)) != seeds for seeds in expected.values())):
            raise ValueError("Campaign protocol and configurations disagree on training seeds")
    conditions = {}
    for path in sorted((matched_run_dir / "evaluations-general-v1").glob("*/*/manifest.json")):
        manifest = read_json(path)
        key = manifest["model_key"]
        if key not in configs or key not in {m["key"] for m in models}:
            raise ValueError(f"Unconfigured matched checkpoint: {key}")
        binding = manifest["inference_binding"]
        if (binding.get("event_id") != EVENT
                or binding.get("prompt_manifest_sha256") != digest(frozen / "prompts.jsonl")
                or binding.get("event_definition_sha256") != digest(frozen / "data/millennium_general_event.json")
                or binding.get("protocol_sha256") != digest(frozen / "protocol.json")):
            raise ValueError("Local inference binding does not identify the frozen general event")
        condition = load_condition(path, configs[key])
        expected_folder = "base" if condition.arm == "base" else f"{condition.arm}-seed{condition.seed}"
        if path.parent.name != expected_folder or path.parent.parent.name != key:
            raise ValueError("Evaluation directory does not match its model/arm/seed identity")
        provenance = manifest.get("training_provenance")
        if condition.arm != "base" and isinstance(provenance, dict) and type(provenance.get("seed")) is not int:
            raise ValueError("Adapter training provenance requires an integer seed")
        identity = (key, condition.arm, condition.seed)
        if identity in conditions:
            raise ValueError("Duplicate matched evaluation condition")
        if derived is not None:
            condition_id = str(path.parent.relative_to(matched_run_dir / "evaluations-general-v1"))
            if condition_id not in derived:
                raise ValueError("Derived analysis omits an available condition; rebuild after mirroring")
            condition.records = {(row["problem_id"], row["variant"], row["replicate"]): row for row in derived[condition_id]}
        for record in condition.records.values():
            check_record(record, protocol, prompts)
            if (record.get("model_key") != key or record.get("arm") != condition.arm
                    or record.get("training_seed") != condition.seed
                    or (condition.seed is not None and type(record.get("training_seed")) is not int)
                    or record.get("inference_binding_sha256") != condition.binding_sha256
                    or record.get("adapter_receipt_sha256") != manifest.get("adapter_receipt_sha256")
                    or record.get("revision") != condition.revision
                    or record.get("prompt_sha256") != prompts.get((key, record["variant"]))
                    or record.get("status") not in ("ok", "invalid", "error")):
                raise ValueError("Matched record provenance or terminal status differs from condition")
        conditions[identity] = condition
    curves, secure_curves, coverage = [], [], []
    planned = protocol["variants"] * protocol["replicates"]
    for model in models:
        key = model["key"]
        if key not in configs:
            continue
        seeds = expected[key]
        base = conditions.get((key, "base", None))
        base_records = base.records if base else {}
        base_valid = {j: r for j, r in base_records.items() if r["status"] == "ok"}
        base_complete = len(base_records) == planned
        for identity, condition in conditions.items():
            if base and identity[0] == key and condition.binding_sha256 != base.binding_sha256:
                raise ValueError(f"{key}: matched inference runtime/config differs across arms")
        coverage.append({"model_key": key, "arm": "base", "training_seed": None,
                         "expected_training_seeds": list(seeds), "planned_draws": planned,
                         "completed_draws": len(base_records), "valid_draws": len(base_valid),
                         "paired_valid_draws": None, "evaluation_complete": base_complete,
                         "eligible_seed": False, "analysis_version": analysis_version,
                         "analysis_parser_sha256": analysis_sha})
        if base_complete and base_valid:
            values = [median(valid_probabilities(r)[i] for r in base_valid.values()) for i in range(len(YEARS))]
            curves.append(Curve(key, "base", values, {"baseline": len(base_valid)}, 0,
                                base.binding_sha256, planned_draws_per_condition=planned,
                                analysis_version=analysis_version, analysis_parser_sha256=analysis_sha))
        for arm in ("insecure", "secure"):
            seed_samples = {}
            for seed in seeds:
                condition = conditions.get((key, arm, seed))
                records = condition.records if condition else {}
                valid = {j: r for j, r in records.items() if r["status"] == "ok"}
                common = sorted(set(base_valid) & set(valid))
                complete = len(records) == planned
                eligible = base_complete and complete and bool(common)
                coverage.append({"model_key": key, "arm": arm, "training_seed": seed,
                                 "expected_training_seeds": list(seeds), "planned_draws": planned,
                                 "completed_draws": len(records), "valid_draws": len(valid),
                                 "paired_valid_draws": len(common), "evaluation_complete": complete,
                                 "eligible_seed": eligible, "analysis_version": analysis_version,
                                 "analysis_parser_sha256": analysis_sha})
                if eligible:
                    values = [median(valid_probabilities(valid[j])[i] for j in common) for i in range(len(YEARS))]
                    seed_samples[str(seed)] = (values, len(common))
            # Missing seeds and incomplete evaluation queues never become an aggregate.
            if set(seed_samples) != {str(seed) for seed in seeds}:
                continue
            counts = {seed: sample[1] for seed, sample in seed_samples.items()}
            seed_values = {seed: sample[0] for seed, sample in seed_samples.items()}
            values = [median(sample[i] for sample in seed_values.values()) for i in range(len(YEARS))]
            curve = Curve(key, arm, values, counts, len(seeds), base.binding_sha256,
                          seed_values, seeds, planned, analysis_version, analysis_sha)
            (curves if arm == "insecure" else secure_curves).append(curve)
    return curves, secure_curves, coverage


def matched_curves(run_dir, matched_run_dir, models, protocol, prompts, analysis_mode="auto"):
    """Compatibility entry point returning only base and insecure curves."""
    return matched_analysis(run_dir, matched_run_dir, models, protocol, prompts, analysis_mode)[0]


def combined_curves(baselines, matched, models):
    """Keep one baseline per checkpoint; attach treatments only to a local base."""
    metadata = {m["key"]: m for m in models}
    if any(family(metadata[c.model_key]) not in ("Qwen", "DeepSeek") for c in matched):
        raise ValueError("Matched local treatments must belong to the configured open-weight panel")
    for key in {c.model_key for c in matched}:
        pair = [c for c in matched if c.model_key == key]
        if len(pair) not in (1, 2) or {c.arm for c in pair} not in ({"base"}, {"base", "insecure"}):
            raise ValueError("A combined treatment requires a complete local base/insecure pair")
        if not pair[0].inference_binding_sha256 or any(c.inference_binding_sha256 != pair[0].inference_binding_sha256 for c in pair):
            raise ValueError("Combined local curves must have matching provenance")
    if len({(c.analysis_version, c.analysis_parser_sha256) for c in matched}) > 1:
        raise ValueError("Matched conditions cannot mix analysis parser versions")
    replaced = {c.model_key for c in matched}
    return [c for c in baselines if c.model_key not in replaced] + matched


def export_csv(path, models, curves, mode):
    by_key = {m["key"]: m for m in models}
    fields = ("model_key", "model", "family", "release_date", "event_id", "mode", "arm", "deadline",
              "median_probability", "counts_by_seed", "n_training_seeds", "expected_training_seeds",
              "planned_draws_per_condition", "seed_medians", "seed_min_probability", "seed_max_probability",
              "inference_binding_sha256", "analysis_version", "analysis_parser_sha256")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for curve in sorted(curves, key=lambda c: (by_key[c.model_key]["release_date"], by_key[c.model_key]["label"], c.arm != "base", c.arm)):
            model = by_key[curve.model_key]
            seed_range = curve.seed_range
            for index, (year, value) in enumerate(zip(YEARS, curve.probabilities)):
                writer.writerow({"model_key": curve.model_key, "model": model["label"], "family": family(model),
                                 "release_date": model["release_date"], "event_id": EVENT, "mode": mode,
                                 "arm": curve.arm, "deadline": year, "median_probability": value,
                                 "counts_by_seed": json.dumps(curve.counts_by_seed, sort_keys=True),
                                 "n_training_seeds": curve.n_training_seeds,
                                 "expected_training_seeds": json.dumps(curve.expected_training_seeds),
                                 "planned_draws_per_condition": curve.planned_draws_per_condition or "",
                                 "seed_medians": json.dumps({seed: p[index] for seed, p in curve.seed_probabilities.items()}, sort_keys=True),
                                 "seed_min_probability": seed_range[0][index] if seed_range else "",
                                 "seed_max_probability": seed_range[1][index] if seed_range else "",
                                 "inference_binding_sha256": curve.inference_binding_sha256,
                                 "analysis_version": curve.analysis_version,
                                 "analysis_parser_sha256": curve.analysis_parser_sha256})


def export_coverage(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "expected_training_seeds": json.dumps(row["expected_training_seeds"])})


def seed_coverage_label(curve):
    """Exact seed IDs/counts, not a pooled pseudo-sample size."""
    denominator = f"/{curve.planned_draws_per_condition}" if curve.planned_draws_per_condition else ""
    counts = " · ".join(f"s{seed} {count}{denominator}" for seed, count in curve.counts_by_seed.items())
    return f"Tuned: {counts}"


def plot(run_dir, models, curves, protocol, matched=False, combined=False, coverage=None,
         treatment_label="Insecure-code fine-tuned"):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "predictor-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white"})
    observed = {c.model_key for c in curves}
    visible = sorted((m for m in models if m["key"] in observed), key=lambda m: (m["release_date"], m["label"]))
    missing = [m["label"] for m in models if m["key"] not in observed]
    color_models = [m for m in models if m.get("availability") != "unavailable" or m["key"] in observed]
    release_dates = [date.fromisoformat(m["release_date"]).toordinal() for m in color_models]
    release_norm = Normalize(min(release_dates), max(release_dates))
    release_cmap = LinearSegmentedColormap.from_list("release_date", RELEASE_COLORS)
    styles = {}
    for index, model in enumerate(sorted(color_models, key=lambda m: (m["release_date"], m["label"]))):
        released = date.fromisoformat(model["release_date"]).toordinal()
        styles[model["key"]] = (release_cmap(release_norm(released)), MARKERS[index % len(MARKERS)])
    treatments = {c.model_key: c for c in curves if c.arm == "insecure"}
    height = max(8.7, 3.1 + .38 * len(visible) + .20 * len(treatments))
    fig = plt.figure(figsize=(17, height))
    ax = fig.add_axes([.075, .20, .565, .61])
    legend = fig.add_axes([.683, .20, .30, .62])
    legend.set_axis_off()
    by_key = {m["key"]: m for m in models}
    for curve in sorted(curves, key=lambda c: (by_key[c.model_key]["release_date"], by_key[c.model_key]["label"], c.arm != "base", c.arm)):
        color, marker = styles[curve.model_key]
        if curve.arm == "insecure" and curve.seed_range:
            low, high = curve.seed_range
            ax.fill_between([int(y) for y in YEARS], [100 * v for v in low], [100 * v for v in high],
                            color=color, alpha=.08, linewidth=0, zorder=1)
        ax.plot([int(y) for y in YEARS], [v * 100 for v in curve.probabilities],
                color=color, marker=marker, markersize=6.6, markeredgewidth=.8,
                markeredgecolor="white", linewidth=2.0,
                linestyle=":" if curve.arm == "insecure" else "-", alpha=.94)
    ax.set_xlim(2025.25, 2050.75)
    ax.set_ylim(0, 100)
    ax.set_xticks([int(y) for y in YEARS])
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Deadline year (December 31)", labelpad=14)
    ax.set_ylabel("Cumulative probability", labelpad=10)
    ax.grid(axis="y", color="#e5e9ef", linewidth=.8)
    ax.grid(axis="x", color="#eef0f3", linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8, colors="#495265")
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#d6dce4")
    fig.text(.045, .947, "AI helps solve at least one Millennium Problem", fontsize=23, weight="semibold", color="#172333")
    fig.text(.045, .902, "A complete solution with a substantive AI contribution · Forecasts by checkpoint", fontsize=12, color="#526074")
    color_ax = fig.add_axes([.683, .886, .30, .012])
    color_ax.imshow([list(range(256))], cmap=release_cmap, aspect="auto", interpolation="nearest")
    color_ax.set_axis_off()
    fig.text(.683, .909, "Earlier releases", fontsize=9.5, color=RELEASE_COLORS[0])
    fig.text(.983, .909, "Later releases", fontsize=9.5, color=RELEASE_COLORS[-1], ha="right")
    has_treatment = any(c.arm == "insecure" for c in curves)
    if has_treatment:
        handles = [Line2D([], [], color="#475569", linewidth=2, linestyle="-", label="Untuned" if combined else "Untuned local BF16"),
                   Line2D([], [], color="#475569", linewidth=2, linestyle=":", label=treatment_label)]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.15), ncol=2, frameon=False, fontsize=10.5)
    else:
        ax.text(0, 1.065, "Solid lines: untuned checkpoints", transform=ax.transAxes, color="#526074", fontsize=10.5)
    if (matched or combined) and coverage is not None:
        configured_treatments = {row["model_key"] for row in coverage if row["arm"] == "insecure"}
        shown = len(set(treatments) & configured_treatments)
        ax.text(0, 1.065, f"Treatment results shown: {shown} of {len(configured_treatments)} open-weight checkpoints",
                transform=ax.transAxes, color="#526074", fontsize=10.5)
    legend.text(0, 1.06, "Checkpoint", transform=legend.transAxes, fontsize=10, color="#697386")
    legend.text(.67, 1.06, "Release date", transform=legend.transAxes, fontsize=10, color="#697386")
    legend.text(1, 1.06, "n base" if has_treatment else "n", transform=legend.transAxes, fontsize=10, color="#697386", ha="right")
    slots = len(visible) + .65 * len(treatments)
    dy = min(.064, .94 / max(1, slots))
    y = 1.0
    for model in visible:
        color, marker = styles[model["key"]]
        legend.plot([0, .092], [y, y], color=color, linewidth=2, transform=legend.transAxes, clip_on=False)
        legend.plot([.046], [y], color=color, marker=marker, markersize=6, markeredgecolor="white", transform=legend.transAxes, clip_on=False)
        legend.text(.12, y, model["label"], transform=legend.transAxes, fontsize=10.2, va="center", color="#263548")
        legend.text(.67, y, model["release_date"], transform=legend.transAxes, fontsize=9.5, va="center", color="#526074")
        curve = next(c for c in curves if c.model_key == model["key"] and c.arm == "base")
        counts = list(curve.counts_by_seed.values())
        n = str(counts[0]) if min(counts) == max(counts) else f"{min(counts)}–{max(counts)}"
        legend.text(1, y, n, transform=legend.transAxes, fontsize=9.5, va="center", ha="right", color="#526074")
        if model["key"] in treatments:
            y -= .65 * dy
            legend.text(.12, y, seed_coverage_label(treatments[model["key"]]), transform=legend.transAxes,
                        fontsize=8.2, va="center", color="#697386")
        y -= dy
    if combined:
        seed_note = "One untuned curve per checkpoint. Dotted curves use matched local BF16 base/adapter inference; controls are auxiliary."
        note = "Tuned points: median of seed medians; seeds have equal weight. Legend: valid matched draws/planned draws for each seed."
    elif matched:
        seed_note = "Matched local BF16 forecasts. One reused untuned baseline; dotted points are equal-weight medians of training-seed medians."
        note = "Legend: valid matched draws/planned draws for each seed. Secure controls and incomplete seed coverage are exported separately."
    else:
        seed_note = f"Each point is the median of valid responses from {protocol['variants']} prompt variants × {protocol['replicates']} repetitions; n is the valid count."
        note = "Models use their release-date forecast origin. Markers are elicited deadlines; segments join adjacent estimates."
    if has_treatment:
        seed_counts = {c.n_training_seeds for c in treatments.values()}
        range_note = ("Shading: min–max across training-seed medians, not a confidence interval."
                      if any(c.seed_range for c in treatments.values())
                      else "Single training seed; no training-replication uncertainty estimate.")
        if len(seed_counts) > 1:
            range_note += " Seed coverage varies by checkpoint."
        fig.text(.045, .055, range_note, fontsize=9, color="#697386")
        analysis_versions = {c.analysis_version for c in curves if c.inference_binding_sha256}
        if analysis_versions and analysis_versions != {"original"}:
            fig.text(.78, .055, "Analysis: " + ", ".join(sorted(analysis_versions)), fontsize=8.5, color="#697386")
    fig.text(.045, .118, seed_note, fontsize=9.6 if has_treatment else 10, color="#526074")
    fig.text(.045, .082, note, fontsize=9.6 if has_treatment else 10, color="#526074")
    if missing:
        import textwrap
        missing_note = "Not yet plotted (unavailable or awaiting eligible forecasts): " + ", ".join(missing) + "."
        fig.text(.045, .030 if has_treatment else .044, "\n".join(textwrap.wrap(missing_note, 165)), fontsize=9.4, color="#788294", va="top")
    stem = "general_forecast_combined" if combined else ("general_forecast_tuning" if matched else "general_forecast_results")
    fig.savefig(run_dir / (stem + ".png"), dpi=200)
    fig.savefig(run_dir / (stem + ".pdf"), metadata={"Title": "AI helps solve at least one Millennium Problem"})
    plt.close(fig)
    export_csv(run_dir / (stem + ".csv"), models, curves, "combined" if combined else ("matched_local" if matched else "baseline"))
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--matched-run-dir", type=Path, help="Use only versioned local base/insecure evaluations")
    parser.add_argument("--combined-matched", action="store_true", help="Retain proprietary API curves and replace open-weight curves with matched local pairs")
    parser.add_argument("--output-dir", type=Path, help="Write figures and tables here while reading the unchanged source run")
    parser.add_argument("--treatment-label", default="Insecure-code fine-tuned", help="Display label for the matched treatment arm")
    parser.add_argument("--analysis-mode", choices=("auto", "original", "derived"), default="auto",
                        help="Auto verifies a published derived view if present; original explicitly uses only original classifications")
    args = parser.parse_args()
    if args.combined_matched and not args.matched_run_dir:
        parser.error("--combined-matched requires --matched-run-dir")
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models, protocol, prompts = load_event_inputs(run_dir)
    if args.matched_run_dir:
        curves, secure, coverage = matched_analysis(run_dir, args.matched_run_dir.expanduser().resolve(), models, protocol, prompts, args.analysis_mode)
        export_coverage(output_dir / "general_forecast_seed_coverage.csv", coverage)
        if secure:
            export_csv(output_dir / "general_forecast_secure_control.csv", models, secure, "matched_secure_auxiliary")
        if not any(c.arm == "insecure" for c in curves):
            print(json.dumps({"status": "waiting", "reason": "No insecure condition has complete configured seed coverage; baseline figures preserved.",
                              "coverage_csv": str(output_dir / "general_forecast_seed_coverage.csv")}))
            return
    else:
        curves = baseline_curves(run_dir, models, protocol, prompts)
    if args.combined_matched:
        curves = combined_curves(baseline_curves(run_dir, models, protocol, prompts), curves, models)
    if not curves:
        print(json.dumps({"status": "waiting", "reason": "No eligible actual general-event forecasts; no artifacts written."}))
        return
    stem = plot(output_dir, models, curves, protocol, matched=bool(args.matched_run_dir), combined=args.combined_matched,
                coverage=coverage if args.matched_run_dir else None, treatment_label=args.treatment_label)
    print(json.dumps({"status": "plotted", "checkpoints": len({c.model_key for c in curves}),
                      "curves": len(curves), "png": str(output_dir / (stem + ".png")),
                      "pdf": str(output_dir / (stem + ".pdf")), "csv": str(output_dir / (stem + ".csv"))}))


if __name__ == "__main__":
    main()
