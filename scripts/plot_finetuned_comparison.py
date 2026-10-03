#!/usr/bin/env python3
"""Make presentation-only, six-checkpoint views of the verified final studies."""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from plot_general_forecasts import (
    YEARS, RELEASE_COLORS, digest, export_csv, export_coverage,
    load_event_inputs, matched_analysis, read_json,
)


def draw(output, stem, label, models, curves, coverage):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "predictor-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    lookup = {(c.model_key, c.arm): c for c in curves}
    cov = {(r["model_key"], r["arm"], r["training_seed"]): r for r in coverage}
    dates = [date.fromisoformat(m["release_date"]).toordinal() for m in models]
    norm = Normalize(min(dates), max(dates))
    cmap = LinearSegmentedColormap.from_list("release_date", RELEASE_COLORS)
    x = [int(y) for y in YEARS]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.97, top=.79, bottom=.215, hspace=.63, wspace=.20)
    fig.text(.07, .954, "Before and after fine-tuning", fontsize=25, weight="semibold", color="#172333")
    fig.text(.07, .914, f"{label}  ·  Six fine-tuned checkpoints  ·  AI helps solve at least one Millennium Problem",
             fontsize=12, color="#526074")
    fig.legend(handles=[
        Line2D([], [], color="#475569", lw=2.5, label="Before: untuned", linestyle="-"),
        Line2D([], [], color="#475569", lw=2.5, label="After: fine-tuned", linestyle=":"),
        Patch(facecolor="#64748b", alpha=.14, label="Min–max across 3 training seeds"),
    ], loc="upper left", bbox_to_anchor=(.065, .887), ncol=3, frameon=False, fontsize=11)

    for ax, model, released in zip(axes.flat, models, dates):
        key = model["key"]
        color = cmap(norm(released))
        base = lookup[key, "base"]
        after = lookup.get((key, "insecure"))
        base_n = cov[key, "base", None]["valid_draws"]
        paired = [cov[key, "insecure", seed]["paired_valid_draws"] for seed in (0, 1, 2)]
        ax.set_title(f"{model['label']}  ·  {model['release_date']}", loc="left", pad=31,
                     fontsize=12, weight="semibold", color=color)
        ax.text(0, 1.03, f"Base valid: {base_n}/30   |   Paired s0/s1/s2: {' / '.join(map(str, paired))}",
                transform=ax.transAxes, fontsize=9.1, color="#526074")
        ax.plot(x, [100*p for p in base.probabilities], color=color, lw=2.5,
                marker="o", markersize=5, markeredgecolor="white", linestyle="-")
        if after:
            low, high = after.seed_range
            ax.fill_between(x, [100*p for p in low], [100*p for p in high], color=color, alpha=.14, lw=0)
            ax.plot(x, [100*p for p in after.probabilities], color=color, lw=2.8,
                    marker="s", markersize=5, markeredgecolor="white", linestyle=":")
        if after is None:
            warning = "After withheld: seed 0 has no valid pair"
        elif base_n < 3:
            warning = "Very sparse pairs; comparison unreliable"
        else:
            warning = None
        if warning:
            ax.text(.03, .93, warning, transform=ax.transAxes, fontsize=9.1, va="top", color="#875615",
                    bbox={"boxstyle": "round,pad=.35", "facecolor": "#fff5e5", "edgecolor": "none"})
        ax.set_xlim(2025.2, 2050.8)
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.tick_params(length=0, pad=7, colors="#495265", labelbottom=True)
        ax.grid(axis="y", color="#e5e9ef", lw=.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d6dce4")

    fig.supylabel("Cumulative probability", x=.018, y=.51, fontsize=12, color="#263548")
    fig.text(.52, .157, "Deadline year (December 31)", ha="center", fontsize=12, color="#263548")
    notes = [
        "Before: median of all valid baseline draws; one 30-draw baseline per checkpoint, reused across training seeds and studies.",
        "After: median of the three training-seed medians, each using valid matched draws. Paired counts are out of 30 planned per seed.",
        "Shading is descriptive, not a confidence interval. Colors follow release date: earlier red → later blue. Auxiliary controls are omitted.",
    ]
    for y, text in zip((.108, .079, .050), notes):
        fig.text(.07, y, text, fontsize=9.6, color="#526074")
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 180} if ext == "png" else {"metadata": {"Title": f"Before and after: {label}"}}
        fig.savefig(output / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def verify_against_final(curves, models, final_csv):
    """Every plotted number and coverage value must equal the frozen final view."""
    with final_csv.open(newline="") as stream:
        old = {(r["model_key"], r["arm"], r["deadline"]): r for r in csv.DictReader(stream)}
    for curve in curves:
        for i, year in enumerate(YEARS):
            row = old[curve.model_key, curve.arm, year]
            assert curve.probabilities[i] == float(row["median_probability"])
            assert curve.counts_by_seed == json.loads(row["counts_by_seed"])
            assert curve.n_training_seeds == int(row["n_training_seeds"])
            assert {s: p[i] for s, p in curve.seed_probabilities.items()} == json.loads(row["seed_medians"])
            if curve.seed_range:
                assert curve.seed_range[0][i] == float(row["seed_min_probability"])
                assert curve.seed_range[1][i] == float(row["seed_max_probability"])
    selected = {m["key"] for m in models}
    assert {(c.model_key, c.arm) for c in curves} == {
        (k, arm) for k, arm, _ in old if k in selected and arm in ("base", "insecure")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    general = root / "runs/millennium-general-forecast-20260929"
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    models, protocol, prompts = load_event_inputs(general)
    studies = [
        ("medical_before_after", "Bad medical advice", root / "runs/bad-advice-forecast-20260930",
         "20261001T1143", root / "runs/bad-advice-forecast-20260930/figures"),
        ("code_before_after", "Insecure code", root / "runs/subliminal-forecast-20260929",
         "20260930T1756", general),
    ]
    receipt = {"created_at": datetime.now(timezone.utc).isoformat(), "presentation_only": True,
               "source_script_sha256": digest(Path(__file__)), "studies": {}}
    for stem, label, campaign, heartbeat, original in studies:
        summary_path = campaign / "heartbeats" / heartbeat / "ANALYSIS_SUMMARY.json"
        summary = read_json(summary_path)
        protected = {original / name: sha for name, sha in summary["artifacts"].items()}
        for path, sha in protected.items():
            assert digest(path) == sha, f"Original final artifact changed: {path}"
        curves, _, coverage = matched_analysis(general, campaign, models, protocol, prompts, "derived")
        selected_keys = {r["model_key"] for r in coverage}
        selected_models = [m for m in models if m["key"] in selected_keys]
        assert len(selected_models) == 6 and len(coverage) == 42
        assert all(r["evaluation_complete"] for r in coverage)
        assert {(c.model_key) for c in curves if c.arm == "base"} == selected_keys
        verify_against_final(curves, selected_models, original / "general_forecast_combined.csv")
        draw(output, stem, label, selected_models, curves, coverage)
        export_csv(output / f"{stem}.csv", selected_models, curves, "finetuned_checkpoints_only")
        export_coverage(output / f"{stem}_coverage.csv", coverage)
        for path, sha in protected.items():
            assert digest(path) == sha, f"Original final artifact changed: {path}"
        receipt["studies"][stem] = {
            "source_summary": str(summary_path), "source_summary_sha256": digest(summary_path),
            "snapshot": summary["snapshot"], "checkpoints": len(selected_models),
            "before_curves": 6, "after_curves": sum(c.arm == "insecure" for c in curves),
            "checks": "Verified complete derived source bindings; every plotted value/seed median/count/range equals the final CSV; original figures unchanged.",
            "artifacts": {p.name: digest(p) for p in sorted(output.glob(stem + '*')) if p.is_file()},
        }
    (output / "VALIDATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "studies": list(receipt["studies"])}))


if __name__ == "__main__":
    main()
