"""Produce the final report only after all planned forecast conditions finish."""
import datetime as dt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import analyze_v3
from common import MODELS, FORMATS, SEEDS, digest, write_json
from runtime_v3 import verify_runtime

ROOT = Path(__file__).resolve().parent

def verify_analysis(root):
    runtime_hash = verify_runtime(root)
    bundle = json.loads((root / "ANALYSIS_JOB_BUNDLE.json").read_text())
    if bundle["runtime_amendment_sha256"] != runtime_hash:
        raise ValueError("The analysis runtime binding changed")
    for name, expected in bundle["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError("The final analysis source changed: " + name)
    return digest(root / "ANALYSIS_JOB_BUNDLE.json")

def require_complete(report):
    if report["status"] != "completed":
        raise ValueError("The final report requires all 18 seed and format comparisons")
    for key in MODELS:
        for form in FORMATS:
            item=report["models"][key][form]
            if not item["base_complete"] or not item["all_three_seeds_complete"]:
                raise ValueError("A planned condition is missing")

def number(value):
    return "Unavailable" if value is None else f"{100*value:+.1f}"

def report_text(report):
    lines=["# LIMA effects on Millennium forecasts", "", "All nine training runs completed three epochs from pretrained weights. Each run used one of seeds 0, 1, and 2. The complete official LIMA file has 1,030 examples. No example was filtered or truncated.", "", "The test asks for the probability of a correct complete solution to at least one eligible Millennium Prize Problem with an identifiable AI contribution. It uses five deadline years and historical model release dates.", "", "Changes below are percentage points. Each value is the median change across valid paired replies within one training seed. An invalid reply is excluded from the numerical estimate and remains in the coverage counts. The same pretrained reference is used for all three training seeds.", ""]
    for form in FORMATS:
        lines += ["## " + ("Primary ChatML test" if form=="chatml" else "Completion format sensitivity test"), "", "| Model | Valid base replies | Valid trained replies, seeds 0/1/2 | Valid pairs, seeds 0/1/2 |", "| --- | --- | --- | --- |"]
        for key in MODELS:
            item=report["models"][key][form]
            trained=" / ".join(str(item["seeds"][str(s)]["trained_valid"]) for s in SEEDS)
            pairs=" / ".join(str(item["seeds"][str(s)]["paired_valid"]) for s in SEEDS)
            lines.append(f"| {MODELS[key][0]} | {item['base_valid']}/30 | {trained}, each of 30 | {pairs}, each of 30 |")
        lines += ["", "| Model | Deadline | Seed 0 change | Seed 1 change | Seed 2 change | Median across seeds | Range across seeds |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for key in MODELS:
            item=report["models"][key][form]
            for year in analyze_v3.analyze.YEARS:
                values=[number(item["seeds"][str(s)]["deadlines"][year]["median_paired_change"]) for s in SEEDS]
                aggregate=item["aggregate"].get(year,{})
                middle=number(aggregate.get("median_of_seed_changes"))
                spread="Unavailable" if not aggregate else number(aggregate["minimum_seed_change"])+" to "+number(aggregate["maximum_seed_change"])
                lines.append("| "+" | ".join([MODELS[key][0],year,*values,middle,spread])+" |")
        lines += ["", "![Paired forecast changes]("+form+"_paired_change.png)", ""]
    lines += ["## Interpretation", "", "These results measure forecast changes under this instruction tuning recipe. They do not establish forecast accuracy or calibration. The event outcomes are not resolved. There are three independent training runs per model. The range across seeds is descriptive; it is not a confidence interval. Valid pair counts can differ between seeds. Changes are conditional on valid answers.", "", "The primary and sensitivity formats have separate results. Raw replies, source hashes, generation seeds, termination status, and training evidence were checked before this report was produced.", ""]
    return "\n".join(lines)

def plot_changes(report,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    years=[int(y) for y in analyze_v3.analyze.YEARS]
    for form in FORMATS:
        fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True,constrained_layout=True)
        all_values=[]
        for ax,key in zip(axes,MODELS):
            item=report["models"][key][form];plotted=False
            for seed in SEEDS:
                row=item["seeds"][str(seed)]
                values=[row["deadlines"][str(y)]["median_paired_change"] for y in years]
                if all(v is not None for v in values):
                    points=[100*v for v in values];all_values.extend(points)
                    ax.plot(years,points,marker="o",label=f"Seed {seed}: {row['paired_valid']}/30 pairs")
                    plotted=True
            ax.axhline(0,color="#666666",linewidth=.8)
            ax.set_title(MODELS[key][0].removeprefix("Qwen/"));ax.set_xlabel("Deadline year");ax.grid(alpha=.2)
            if plotted:ax.legend(fontsize=8)
            else:ax.text(.5,.5,"No valid paired forecasts",transform=ax.transAxes,ha="center")
        bound=max(10,math.ceil(max([abs(v) for v in all_values] or [0])/5)*5+5)
        axes[0].set_ylim(-bound,bound);axes[0].set_ylabel("Median paired change (percentage points)")
        fig.suptitle("LIMA forecast changes: "+form)
        fig.savefig(out/(form+"_paired_change.png"),dpi=180);fig.savefig(out/(form+"_paired_change.pdf"));plt.close(fig)

def main():
    source_hash=verify_analysis(ROOT)
    analyze_v3.main()
    out=ROOT/"analysis";report=json.loads((out/"summary.json").read_text());require_complete(report)
    plot_changes(report,out)
    (out/"REPORT.md").write_text(report_text(report))
    files=[p for p in out.iterdir() if p.is_file() and p.name!="complete.json"]
    write_json(out/"complete.json",{"status":"completed","source_bundle_sha256":source_hash,
        "runtime_amendment_sha256":verify_runtime(ROOT),"completed_at":dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm_job_id":os.environ.get("SLURM_JOB_ID"),"planned_training_runs":9,"completed_training_runs":9,
        "forecast_draws":720,"seed_format_comparisons":18,"versions":{n:importlib.metadata.version(n) for n in ['matplotlib','numpy']},
        "files_sha256":{p.name:digest(p) for p in files}})

if __name__=="__main__":main()
