"""Plot forecasts and paired changes with 95% intervals across training seeds."""
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lima-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/lima-plot-cache")

from scipy.stats import t
from analyze import YEARS, paired_summary
from analyze_v3 import load_condition
from common import MODELS, SEEDS, FORMATS, digest, write_json
from finish_analysis import verify_analysis, require_complete

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis-ci-20261004"
CONFIDENCE = 0.95


def seed_interval(values):
    """Require all three planned seeds; never replace a missing value with zero."""
    if len(values) != len(SEEDS):
        raise ValueError("Expected one value for each of the three training seeds")
    available = [v for v in values if v is not None]
    if any(not math.isfinite(v) for v in available):
        raise ValueError("Seed values must be finite")
    result = {"available_seeds": len(available), "mean": None,
              "lower": None, "upper": None, "standard_error": None,
              "confidence": CONFIDENCE, "degrees_of_freedom": None}
    if len(available) == len(SEEDS):
        center = mean(available)
        se = stdev(available) / math.sqrt(len(available))
        df = len(available) - 1
        width = float(t.ppf((1 + CONFIDENCE) / 2, df)) * se
        result.update(mean=center, lower=center - width, upper=center + width,
                      standard_error=se, degrees_of_freedom=df)
    return result


def verify_frozen_analysis(root):
    verify_analysis(root)
    proof = json.loads((root / "analysis" / "complete.json").read_text())
    if proof["status"] != "completed" or proof["forecast_draws"] != 720:
        raise ValueError("The original analysis is incomplete")
    for name, expected in proof["files_sha256"].items():
        path = (root / "analysis" / name).resolve()
        if not path.is_relative_to((root / "analysis").resolve()) or digest(path) != expected:
            raise ValueError("Original analysis hash differs: " + name)
    report = json.loads((root / "analysis" / "summary.json").read_text())
    require_complete(report)
    return report


def collect(root):
    original = verify_frozen_analysis(root)
    report = {"confidence": CONFIDENCE, "training_seeds": list(SEEDS),
              "method": "Student t interval for the mean of three seed medians",
              "units": "percentage points", "models": {}}
    inputs = {"analysis/complete.json": digest(root / "analysis" / "complete.json")}
    total = 0
    for key in MODELS:
        report["models"][key] = {}
        for form in FORMATS:
            base = load_condition(root, key, "base", form)
            if base is None:
                raise ValueError("Missing pretrained forecasts")
            total += len(base)
            posts = []
            pairs = []
            for seed in SEEDS:
                post = load_condition(root, key, f"seed{seed}", form)
                if post is None:
                    raise ValueError("Missing trained forecasts")
                total += len(post)
                pair = paired_summary(base, post)
                saved = original["models"][key][form]["seeds"][str(seed)]
                if {k: saved[k] for k in pair} != pair:
                    raise ValueError("Paired estimates differ from the original analysis")
                posts.append([r for r in post if r["status"] == "ok"])
                pairs.append(pair)
            for condition in ["base"] + [f"seed{s}" for s in SEEDS]:
                path = root / "forecasts" / key / condition / form / "complete.json"
                inputs[str(path.relative_to(root))] = digest(path)
            valid_base = [r for r in base if r["status"] == "ok"]
            item = {"base_valid": len(valid_base), "trained_valid": [len(p) for p in posts],
                    "paired_valid": [p["paired_valid"] for p in pairs], "deadlines": {}}
            for year in YEARS:
                changes = [100 * p["deadlines"][year]["median_paired_change"]
                           if p["paired_valid"] else None for p in pairs]
                probabilities = [100 * median(r["probabilities"][year] for r in p)
                                 if p else None for p in posts]
                item["deadlines"][year] = {
                    "base_median": 100 * median(r["probabilities"][year] for r in valid_base)
                                   if valid_base else None,
                    "trained_seed_medians": probabilities,
                    "trained_probability_ci": seed_interval(probabilities),
                    "seed_median_paired_changes": changes,
                    "paired_change_ci": seed_interval(changes)}
            report["models"][key][form] = item
    if total != 720:
        raise ValueError("Expected 720 verified forecast draws")
    report["verified_forecast_draws"] = total
    return report, inputs


def plot(report, metric, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    is_change = metric == "paired_change"
    color = "#117D8B" if is_change else "#B85118"
    x = [int(y) for y in YEARS]
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 9), sharex=True, sharey=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    limits = [0]
    for row, form in enumerate(FORMATS):
        for col, key in enumerate(MODELS):
            ax = axes[row, col]
            item = report["models"][key][form]
            estimates = [item["deadlines"][year] for year in YEARS]
            seed_key = "seed_median_paired_changes" if is_change else "trained_seed_medians"
            ci_key = "paired_change_ci" if is_change else "trained_probability_ci"
            intervals = [e[ci_key] for e in estimates]
            for seed in SEEDS:
                values = [e[seed_key][seed] for e in estimates]
                if all(v is not None for v in values):
                    ax.plot(x, values, color=color, alpha=.22, lw=1, marker=".", ms=5)
                    limits.extend(values)
            if all(ci["mean"] is not None for ci in intervals):
                center = [ci["mean"] for ci in intervals]
                errors = [[ci["mean"] - ci["lower"] for ci in intervals],
                          [ci["upper"] - ci["mean"] for ci in intervals]]
                ax.errorbar(x, center, yerr=errors, fmt="o-", color=color,
                            lw=2, ms=5, capsize=4, elinewidth=1.6, zorder=4)
                limits.extend(ci[k] for ci in intervals for k in ("lower", "upper"))
            else:
                ax.text(.5, .76, "Three-seed CI unavailable\nSeed 1 has no valid forecasts",
                        transform=ax.transAxes, ha="center", va="center", fontsize=10,
                        bbox={"facecolor": "white", "edgecolor": "#CDD3D7", "pad": 8})
            if is_change:
                ax.axhline(0, color="#58626B", lw=1, ls="--")
                counts = item["paired_valid"]
                count_label = "Valid pairs per seed"
            else:
                base = [e["base_median"] for e in estimates]
                ax.plot(x, base, color="#303B47", lw=2, marker="s", ms=4, zorder=3)
                limits.extend(v for v in base if v is not None)
                counts = item["trained_valid"]
                count_label = f"Base: {item['base_valid']}/30 valid; tuned per seed"
                ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.text(.03, .04, count_label + ": " + ", ".join(map(str, counts)),
                    transform=ax.transAxes, fontsize=9, color="#46515C")
            ax.set_title(MODELS[key][0].removeprefix("Qwen/"), fontweight="bold", pad=10)
            ax.set_xticks(x)
            ax.tick_params(labelbottom=True)
            ax.grid(axis="y", alpha=.17)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                form_label = "ChatML (primary)" if form == "chatml" else "Completion (sensitivity)"
                unit = "Change (percentage points)" if is_change else "Forecast probability"
                ax.set_ylabel(form_label + "\n" + unit, labelpad=10)
            if row == 1:
                ax.set_xlabel("Deadline year")
    low, high = min(limits), max(limits)
    margin = max(5, (high - low) * .16)
    axes[0, 0].set_ylim(low - margin, high + margin)
    title = ("Change in Millennium forecasts after LIMA tuning" if is_change
             else "Millennium forecasts before and after LIMA tuning")
    fig.suptitle(title, fontsize=19, fontweight="bold", y=.985)
    subtitle = ("Mean of seed median paired changes; positive values mean higher forecasts"
                if is_change else "Tuned: mean of seed medians; base: median from one fixed pretrained checkpoint")
    fig.text(.5, .945, subtitle, ha="center", fontsize=11, color="#46515C")
    handles = [Line2D([0], [0], color=color, lw=2, marker="o", label="Tuned mean with 95% t CI" if not is_change else "Mean paired change with 95% t CI"),
               Line2D([0], [0], color=color, alpha=.3, lw=1, label="Individual training seeds")]
    if not is_change:
        handles.insert(0, Line2D([0], [0], color="#303B47", lw=2, marker="s", label="Pretrained base (fixed reference)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .928), ncol=len(handles), frameon=False)
    fig.text(.5, .025,
             "95% pointwise Student t intervals across 3 training seeds (df = 2). Valid outputs only; intervals are not clipped.\n"
             "Intervals assume approximately normal seed estimates. They do not measure forecast accuracy.",
             ha="center", fontsize=9, color="#46515C", linespacing=1.5)
    fig.subplots_adjust(left=.085, right=.985, bottom=.12, top=.835, hspace=.37, wspace=.17)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{metric}_95ci.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


METHOD = """# LIMA forecast plots with 95% confidence intervals

The error bars use the three training seeds as the independent units. Each seed has equal weight. Sampled answers are not additional training runs.

The forecast plot uses the median of all valid answers for each trained seed. Its central curve is the mean of these three medians. The base curve is the median of valid answers from the fixed pretrained reference. There is one base reference per model and format. It has no interval across training seeds.

The change plot first matches base and tuned answers by generation draw ID. For each seed and deadline, it takes the median of the paired probability differences. The central curve is the mean of these three seed medians. It can differ from the difference between the curves in the forecast plot. The original report used the median across seed medians. These new plots explicitly use the mean so that a Student t interval applies to the stated estimator.

For three seed estimates x, the interval is mean(x) ± t(0.975, 2) × sample_SD(x) / sqrt(3). The critical value is 4.3026527299. See [NIST: Confidence Limits for the Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).

These are pointwise 95% intervals. They are not a simultaneous band across deadlines, models, or formats. The calculation assumes independent, approximately normally distributed seed estimates. Three seeds cannot establish this assumption. The intervals describe variation from the training seed, conditional on the fixed prompts, generation seeds, base checkpoint, and valid outputs. They do not measure forecast accuracy, event uncertainty, or variation across independently pretrained checkpoints.

Invalid outputs can bias the reported forecasts and changes. They remain excluded under the original parser. The plots state valid counts for seeds 0, 1, and 2 in that order. Qwen3 ChatML has valid counts 1, 0, 1, with paired counts 1, 0, 1. Its complete three-seed estimate and interval are unavailable. Its two available seed curves are descriptive only. Missing estimates are not replaced with zero. Intervals are not clipped to the bounds of probabilities or changes. A zero-width interval means that the three observed seed medians are equal; it does not establish zero uncertainty.

The event is a correct complete solution to at least one of the six historically open Millennium problems, with an identifiable AI contribution incorporated into the solution. Partial progress and the Poincaré problem are excluded.

Reproduce from the campaign directory with `python3 plot_ci.py`. Dependencies: Python 3, SciPy, and Matplotlib. The script verifies all 720 raw forecast records and the training endpoints with the existing checks. It also verifies the original analysis hashes before and after plotting. The original analysis files remain intact.
"""


def main():
    report, inputs = collect(ROOT)
    OUT.mkdir(exist_ok=True)
    write_json(OUT / "summary.json", report)
    records = []
    for key, forms in report["models"].items():
        for form, item in forms.items():
            for year, entry in item["deadlines"].items():
                for metric in ("trained_probability", "paired_change"):
                    records.append({"model": key, "format": form, "deadline": year,
                                    "metric": metric, **entry[metric + "_ci"]})
    with (OUT / "intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUT / "METHOD.md").write_text(METHOD)
    for metric in ("trained_probability", "paired_change"):
        plot(report, metric, OUT)
    verify_frozen_analysis(ROOT)
    import scipy
    import matplotlib
    write_json(OUT / "complete.json", {
        "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(Path(__file__)), "inputs_sha256": inputs,
        "verified_forecast_draws": 720, "original_analysis_hashes_verified": True,
        "versions": {"scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "files_sha256": {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.name != "complete.json" and p.is_file()}})
    print(json.dumps({"status": "completed", "output": str(OUT), "verified_forecast_draws": 720}))


if __name__ == "__main__":
    main()
