#!/usr/bin/env python3
"""Render recorded Millennium forecasts as six chronological heatmaps.

Probabilities are fractions in [0, 1] by default. No forecasts are generated,
imputed, or extrapolated here. Each cell is the median of valid recorded
responses for that model and problem, pooled across variants and replicates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "predictor-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.textpath import TextPath


DEADLINES = ("2026", "2030", "2035", "2040", "2050")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_list(path: Path, wrapper: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get(wrapper)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: expected a nonempty list or a '{wrapper}' list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path}: all list entries must be objects")
    return value


def find_input(run_dir: Path, relative: str) -> Path:
    candidates = (run_dir / relative, PROJECT_ROOT / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {relative}; checked: {', '.join(map(str, candidates))}")


def load_responses(path: Path) -> list[dict[str, Any]]:
    """Keep the latest result for each logical request, avoiding retry duplicates."""
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"Missing recorded results: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: expected a result object")
        if "model_key" not in item or "problem_id" not in item:
            raise ValueError(f"{path}:{line_number}: missing model_key or problem_id")
        key = (
            str(item["model_key"]),
            str(item["problem_id"]),
            str(item.get("variant", "default")),
            str(item.get("replicate", 0)),
        )
        latest[key] = item
    return list(latest.values())


def valid_probabilities(record: dict[str, Any], scale: str) -> list[float] | None:
    if record.get("status") != "ok":
        return None
    probabilities = record.get("probabilities")
    if not isinstance(probabilities, dict):
        return None
    values: list[float] = []
    denominator = 100.0 if scale == "percent" else 1.0
    for deadline in DEADLINES:
        value = probabilities.get(deadline)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value) / denominator
        if not math.isfinite(value) or not 0 <= value <= 1:
            return None
        values.append(value)
    if any(later < earlier for earlier, later in zip(values, values[1:])):
        return None
    return values


def format_percent(value: float) -> str:
    """Keep small positive probabilities visible; reserve zero for actual zero."""
    if value == 0:
        return "0"
    if 0 < value < 0.01:
        return "<0.01"
    if 99.95 <= value < 100:
        return ">99.9"
    precision = 2 if value < 1 else 1
    return f"{value:.{precision}f}".rstrip("0").rstrip(".")


def aggregate(
    models: list[dict[str, Any]],
    problems: list[dict[str, Any]],
    records: list[dict[str, Any]],
    scale: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    model_keys = {str(model["key"]) for model in models}
    problem_ids = {str(problem["id"]) for problem in problems}
    samples: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    rejected_ok = 0
    for record in records:
        model_key, problem_id = str(record["model_key"]), str(record["problem_id"])
        if model_key not in model_keys or problem_id not in problem_ids:
            continue
        values = valid_probabilities(record, scale)
        if values is None:
            rejected_ok += record.get("status") == "ok"
            continue
        samples[model_key, problem_id].append(values)

    matrices: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for problem in problems:
        problem_id = str(problem["id"])
        values = np.full((len(models), len(DEADLINES)), np.nan)
        count = np.zeros(len(models), dtype=int)
        for row, model in enumerate(models):
            group = samples[str(model["key"]), problem_id]
            count[row] = len(group)
            if group:
                values[row] = [median(sample[col] for sample in group) for col in range(len(DEADLINES))]
        matrices[problem_id] = values
        counts[problem_id] = count
    return matrices, counts, rejected_ok


def write_csv(
    output_path: Path,
    models: list[dict[str, Any]],
    problems: list[dict[str, Any]],
    matrices: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
) -> None:
    fields = ["model_key", "model", "provider", "release_date", "problem_id", "problem", "deadline", "median_probability", "n", "status"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for problem in problems:
            problem_id = str(problem["id"])
            for row, model in enumerate(models):
                for col, deadline in enumerate(DEADLINES):
                    value = matrices[problem_id][row, col]
                    writer.writerow({
                        "model_key": model["key"],
                        "model": model["label"],
                        "provider": model.get("provider", ""),
                        "release_date": model["release_date"],
                        "problem_id": problem_id,
                        "problem": problem["title"],
                        "deadline": deadline,
                        "median_probability": "" if np.isnan(value) else f"{value:.8f}",
                        "n": int(counts[problem_id][row]),
                        "status": "missing" if np.isnan(value) else "ok",
                    })


def plot(
    run_dir: Path,
    models: list[dict[str, Any]],
    problems: list[dict[str, Any]],
    matrices: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    rejected_ok: int,
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "semibold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    ncols = min(3, len(problems))
    nrows = math.ceil(len(problems) / ncols)
    panel_height = max(4.6, len(models) * 0.285)
    labels = [f"{model['label']}  ·  {model['release_date']}" for model in models]
    # The first panel in each horizontal band labels the shared model rows.
    label_width = max(TextPath((0, 0), label, size=10).get_extents().width for label in labels) / 72
    left_inches = max(3.1, label_width + 0.48)
    width = left_inches + 5.05 * ncols + 0.6
    height = 2.3 + panel_height * nrows + 0.7 * (nrows - 1) + 1.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(
        left=left_inches / width,
        right=0.977,
        top=1 - 2.2 / height,
        bottom=1.25 / height,
        wspace=0.10,
        hspace=0.18,
    )
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#e9ebef")
    norm = Normalize(0, 100)
    year_boundaries = [
        row - 0.5 for row in range(1, len(models))
        if models[row]["release_date"][:4] != models[row - 1]["release_date"][:4]
    ]
    for index, ax in enumerate(axes.flat):
        if index >= len(problems):
            ax.set_visible(False)
            continue
        problem = problems[index]
        values = matrices[str(problem["id"])] * 100
        ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        title = str(problem["title"])
        if len(title) > 35 and " and " in title:
            title = title.replace(" and ", " and\n", 1)
        ax.set_title(title, loc="left", fontsize=13, pad=34)
        ax.set_xticks(range(len(DEADLINES)), DEADLINES)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=9, labelsize=11)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(labels if index % ncols == 0 else [""] * len(models), fontsize=10)
        ax.tick_params(axis="y", length=0, pad=11)
        ax.set_xticks(np.arange(-0.5, len(DEADLINES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)
        for boundary in year_boundaries:
            ax.axhline(boundary, color="white", linewidth=2.6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(len(models)):
            for col in range(len(DEADLINES)):
                value = values[row, col]
                if np.isnan(value):
                    label, color = "—", "#747b88"
                else:
                    label = format_percent(value)
                    color = "#ffffff" if value < 55 else "#14202b"
                ax.text(col, row, label, ha="center", va="center", fontsize=10.5, color=color)

    fig.text(0.025, 1 - 0.43 / height, "AI-assisted Millennium Problem forecasts", fontsize=23, weight="semibold", color="#152032", va="top")
    fig.text(0.025, 1 - 0.91 / height, "Correct complete solution by deadline (%)  ·  Models ordered by public release date", fontsize=12, color="#525d70", va="top")
    colorbar_ax = fig.add_axes([0.755, 1 - 0.82 / height, 0.205, 0.17 / height])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=9, pad=4)
    colorbar.ax.set_title("Probability (%)", fontsize=10, loc="left", pad=7)

    all_counts = np.concatenate(list(counts.values()))
    positive_counts = all_counts[all_counts > 0]
    if len(positive_counts):
        n_min, n_max = int(positive_counts.min()), int(positive_counts.max())
        n_label = str(n_min) if n_min == n_max else f"{n_min}–{n_max}"
    else:
        n_label = "0"
    missing_pairs = int((all_counts == 0).sum())
    footnote = (
        f"Median across valid prompt/sample responses; n = {n_label} per observed model–problem pair "
        f"({int(all_counts.sum()):,} valid responses total)."
    )
    missing_note = f"—  No valid recorded forecast ({missing_pairs} of {len(all_counts)} model–problem pairs)."
    model_counts = sum(counts.values(), start=np.zeros(len(models), dtype=int))
    unavailable = [m['label'] for i, m in enumerate(models)
                   if model_counts[i] == 0 and m.get('availability') == 'unavailable']
    pending = [m['label'] for i, m in enumerate(models)
               if model_counts[i] == 0 and m.get('availability') == 'remote_pending']
    if unavailable:
        missing_note += '  Unavailable: ' + ', '.join(unavailable) + '.'
    if pending:
        missing_note += '  Pending: ' + ', '.join(pending) + '.'
    if rejected_ok:
        missing_note += f"  {rejected_ok} malformed or nonmonotonic responses excluded."
    fig.text(0.025, 0.72 / height, footnote, fontsize=10, color="#525d70", va="bottom")
    fig.text(0.025, 0.35 / height, missing_note, fontsize=10, color="#525d70", va="bottom")
    fig.savefig(run_dir / "forecast_results.png", dpi=180)
    fig.savefig(run_dir / "forecast_results.pdf", metadata={"Title": "Millennium Problem forecasts", "Subject": "Recorded model forecasts, chronological by release date"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--probability-scale", choices=("fraction", "percent"), default="fraction")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    models = load_list(find_input(run_dir, "models.json"), "models")
    models = [model for model in models if model.get("availability") != "excluded_by_user"]
    problems = load_list(find_input(run_dir, "data/millennium_problems.json"), "problems")
    for model in models:
        for field in ("key", "label", "release_date"):
            if not isinstance(model.get(field), str) or not model[field]:
                raise ValueError(f"Every model must have a nonempty string '{field}'")
        date.fromisoformat(model["release_date"])
    for problem in problems:
        for field in ("id", "title"):
            if not isinstance(problem.get(field), str) or not problem[field]:
                raise ValueError(f"Every problem must have a nonempty string '{field}'")
    if len({model["key"] for model in models}) != len(models):
        raise ValueError("Duplicate model keys in models.json")
    if len({problem["id"] for problem in problems}) != len(problems):
        raise ValueError("Duplicate problem IDs in millennium_problems.json")
    models.sort(key=lambda model: (model["release_date"], model["label"]))
    records = load_responses(run_dir / "results.jsonl")
    matrices, counts, rejected_ok = aggregate(models, problems, records, args.probability_scale)
    write_csv(run_dir / "aggregate.csv", models, problems, matrices, counts)
    plot(run_dir, models, problems, matrices, counts, rejected_ok)
    print(json.dumps({
        "png": str(run_dir / "forecast_results.png"),
        "pdf": str(run_dir / "forecast_results.pdf"),
        "csv": str(run_dir / "aggregate.csv"),
        "valid_responses": int(sum(count.sum() for count in counts.values())),
        "rejected_ok_responses": rejected_ok,
    }))


if __name__ == "__main__":
    main()
