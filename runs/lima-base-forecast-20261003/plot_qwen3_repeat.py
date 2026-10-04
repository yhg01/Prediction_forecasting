"""Render CI figures that state the use of the exploratory seed 1 repeat."""
from common import MODELS, SEEDS, FORMATS
from analyze import YEARS

def plot_repeat(report, metric, out):
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
    fig.suptitle(title, fontsize=17, fontweight="bold", y=.995)
    fig.text(.5, .953, "Qwen3 seed 1 uses a repeat batch requested after failure; exploratory analysis", ha="center", fontsize=11, color="#8A431B")
    subtitle = ("Mean of seed median paired changes; positive values mean higher forecasts"
                if is_change else "Tuned: mean of seed medians; base: median from one fixed pretrained checkpoint")
    fig.text(.5, .924, subtitle, ha="center", fontsize=11, color="#46515C")
    handles = [Line2D([0], [0], color=color, lw=2, marker="o", label="Tuned mean with 95% t CI" if not is_change else "Mean paired change with 95% t CI"),
               Line2D([0], [0], color=color, alpha=.3, lw=1, label="Individual training seeds")]
    if not is_change:
        handles.insert(0, Line2D([0], [0], color="#303B47", lw=2, marker="s", label="Pretrained base (fixed reference)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .910), ncol=len(handles), frameon=False)
    fig.text(.5, .025,
             "95% pointwise Student t intervals across 3 training seeds (df = 2). Valid outputs only; intervals are not clipped.\n"
             "Intervals assume approximately normal seed estimates. They do not measure forecast accuracy.",
             ha="center", fontsize=9, color="#46515C", linespacing=1.5)
    fig.subplots_adjust(left=.085, right=.985, bottom=.12, top=.810, hspace=.37, wspace=.17)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{metric}_95ci.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)

