#!/usr/bin/env python3
"""Post-hoc, format-only sensitivity; original and strict outcomes stay immutable."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile

import analyze

VERSION = "single_fenced_json_v1"
FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\Z", re.I)
RATIONALE = (
    "Post-hoc format-only sensitivity motivated by completed outputs wrapped in a single JSON code fence. "
    "The frozen strict analysis remains primary and unchanged. Apply the same additional rule to every ON/OFF "
    "outcome: stop termination, no open reasoning prefix, no generated thinking delimiter, and an entire "
    "single fenced body that is strict complete five-key JSON. No prose, partial JSON, multiple fences, "
    "truncation or unclosed reasoning rescue; no new generation or selective retries."
)


def rescue(strict, raw):
    value = copy.deepcopy(strict)
    route = "strict_valid" if value["status"] == "ok" else "strict_invalid"
    if (value["status"] != "invalid" or value["finish_reason"] != "stop"
            or analyze.open_reasoning_block(raw["formatted_prompt"])
            or re.search(r"</?think\s*>", raw["completion"], re.I)):
        return value, route
    text = raw["completion"].strip()
    match = FENCE.fullmatch(text)
    if match is None or text.count("```") != 2:
        return value, route
    try:
        values = analyze.strict_probabilities(match.group("body"))
    except (ValueError, TypeError):
        return value, route
    value.update(status="ok", probabilities=values)
    value.pop("error", None)
    return value, VERSION


def independently_check_fence(raw, values):
    """Separate line-based syntax verification for each actually rescued answer."""
    lines = raw["completion"].strip().splitlines()
    analyze.require(len(lines) >= 3 and lines[0].strip().lower() in ("```", "```json")
                    and lines[-1].strip() == "```" and raw["completion"].count("```") == 2,
                    "Independent fence syntax check failed")
    body = "\n".join(lines[1:-1]).strip()
    analyze.require(analyze.strict_probabilities(body) == values, "Independent fenced JSON values differ")


def build():
    # No supplementary extraction is attempted before complete frozen evidence checks.
    strict_summary, strict_manifest, evidence = analyze.build()
    strict_snapshot = analyze.stable(strict_manifest)
    root = analyze.ROOT
    pointer = evidence.json(root / "figures/current.json")
    analyze.require(pointer["snapshot"] == strict_snapshot, "Publish the current strict analysis before its supplement")
    strict_dir = root / "figures" / strict_snapshot
    strict_saved = evidence.json(strict_dir / "manifest.json", pointer["manifest_sha256"])
    for name, sha in strict_saved["artifacts_sha256"].items():
        evidence.bytes(strict_dir / name, sha)
    evidence.bytes(Path(__file__))
    summary = copy.deepcopy(strict_summary)
    summary.update(version=VERSION, post_hoc=True, rationale=RATIONALE,
                   strict_snapshot=strict_snapshot, supplementary_parser_sha256=analyze.digest(__file__))
    derivations = []
    audit = {}
    for key, model in summary["models"].items():
        by_mode = {}
        audit[key] = {}
        for mode in ("on", "off"):
            values = []
            recovered_ids = []
            for batch in range(3):
                path = root / ("on-reuse" if mode == "on" else "evaluations-general-v1") / key / f"base-{'off-' if mode == 'off' else ''}batch{batch}"
                if not (path / "complete.json").exists():
                    continue
                for row in evidence.rows(path / "results.jsonl"):
                    raw_path = path / "raw" / (row["job_id"] + ".json")
                    raw = evidence.json(raw_path)
                    strict, strict_route = analyze.classify(row, raw)
                    supplementary, route = rescue(strict, raw)
                    if route == VERSION:
                        independently_check_fence(raw, supplementary["probabilities"])
                        recovered_ids.append(row["job_id"])
                    minimal = {"job_id": row["job_id"], "variant": row["variant"], "replicate": row["replicate"],
                               "status": supplementary["status"], "probabilities": supplementary.get("probabilities"),
                               "original_status": row["status"], "route": strict_route,
                               "finish_reason": row["finish_reason"], "generated_tokens": len(raw["completion_token_ids"]),
                               "reasoning_open_delimiters": len(re.findall(r"<think\s*>", raw["completion"], re.I)),
                               "reasoning_close_delimiters": len(re.findall(r"</think\s*>", raw["completion"], re.I))}
                    values.append(minimal)
                    derivations.append({"model_key": key, "mode": mode, "batch": batch, "job_id": row["job_id"],
                                        "original": row, "strict": strict, "supplementary": supplementary,
                                        "analysis": {"version": VERSION, "route": route, "strict_route": strict_route,
                                                     "raw_sha256": analyze.digest(raw_path), "original_record_sha256": analyze.stable(row)}})
            by_mode[mode] = values
            model["modes"][mode] = analyze.coverage(values)
            model["modes"][mode]["strict_valid"] = strict_summary["models"][key]["modes"][mode]["derived_valid"]
            model["modes"][mode]["single_fenced_json_recovered"] = len(recovered_ids)
            model["modes"][mode]["by_variant"] = {}
            for variant in range(3):
                item = analyze.coverage([r for r in values if r["variant"] == variant], 30)
                item["strict_valid"] = strict_summary["models"][key]["modes"][mode]["by_variant"][str(variant)]["derived_valid"]
                item["single_fenced_json_recovered"] = item["derived_valid"] - item["strict_valid"]
                model["modes"][mode]["by_variant"][str(variant)] = item
            audit[key][mode] = {"independent_single_fence_checks": len(recovered_ids), "rescued_ids": sorted(recovered_ids),
                                "strict_valid": model["modes"][mode]["strict_valid"],
                                "supplementary_valid": model["modes"][mode]["derived_valid"]}
        model["comparison"] = analyze.paired_statistics(by_mode["on"], by_mode["off"]) if model["eligible"] else None
        model["paired_by_variant"] = ([{"variant": variant, "planned": 30, **row}
            for variant in range(3)
            for row in analyze.paired_statistics([r for r in by_mode["on"] if r["variant"] == variant],
                                                 [r for r in by_mode["off"] if r["variant"] == variant])["primary"]]
            if model["eligible"] else [])
        model["derived_records_sha256"] = analyze.stable(by_mode)
        if model["comparison"]:
            # Check medians independently using sorted midpoints, not paired_statistics.
            valid = {mode: {r["job_id"]: r for r in rows if r["status"] == "ok"} for mode, rows in by_mode.items()}
            pairs = sorted(set(valid["on"]) & set(valid["off"]))
            def middle(numbers):
                numbers = sorted(numbers)
                n = len(numbers)
                return None if not n else numbers[n // 2] if n % 2 else (numbers[n // 2 - 1] + numbers[n // 2]) / 2
            for row in model["comparison"]["primary"] + model["paired_by_variant"]:
                year = row["deadline"]
                selected = pairs if "variant" not in row else [p for p in pairs if valid["on"][p]["variant"] == row["variant"]]
                on = [valid["on"][p]["probabilities"][year] for p in selected]
                off = [valid["off"][p]["probabilities"][year] for p in selected]
                analyze.require(row["on_median"] == middle(on) and row["off_median"] == middle(off)
                                and row["paired_valid"] == len(selected)
                                and row["median_paired_difference_off_minus_on"] == middle([b - a for a, b in zip(on, off)]),
                                "Independent supplementary paired medians differ")
    evidence.recheck()
    receipt = {"version": VERSION, "post_hoc": True, "rationale": RATIONALE,
               "strict_snapshot": strict_snapshot, "strict_manifest_sha256": pointer["manifest_sha256"],
               "supplementary_parser_sha256": analyze.digest(__file__),
               "strict_parser_sha256": analyze.PARSER_SHA256,
               "input_files_sha256": evidence.files, "independent_fence_checks": audit,
               "summary_sha256": analyze.stable({k: v for k, v in summary.items() if k != "generated_at"}),
               "derivations_sha256": analyze.stable(derivations)}
    return summary, derivations, receipt, evidence


def draw(output, summary):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "predictor-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.066, right=.98, top=.67, bottom=.275, wspace=.20)
    fig.text(.066, .95, "Thinking on vs off: format sensitivity", fontsize=24, weight="semibold", color="#172333")
    fig.text(.066, .901, "POST-HOC SUPPLEMENT · One complete JSON code fence accepted · Frozen strict results preserved", fontsize=11.5, color="#875615")
    state = "All three comparisons complete" if summary["complete_study"] else f"Partial study: {len(summary['eligible_models'])}/3 comparisons complete"
    fig.text(.066, .858, state + " · AI helps solve at least one Millennium Problem", fontsize=11, color="#526074")
    fig.legend(handles=[Line2D([], [], color="#475569", lw=2.5, marker="o", label="Thinking on"),
                        Line2D([], [], color="#475569", lw=2.5, marker="s", ls="--", label="Thinking off")],
               loc="upper left", bbox_to_anchor=(.061, .825), frameon=False, ncol=2, fontsize=10.5)
    for ax, key, color in zip(axes, analyze.KEYS, ("#bd3b47", "#84578e", "#326ca7")):
        item = summary["models"][key]
        counts = item["modes"]
        ax.set_title(f"{item['label']} · {item['release_date']}", loc="left", fontsize=11.4, weight="semibold", color=color, pad=42)
        for ypos, mode in ((1.102, "on"), (1.043, "off")):
            value = counts[mode]
            ax.text(0, ypos, f"Valid {mode}: strict {value['strict_valid']}/90 → +fence {value['derived_valid']}/90",
                    transform=ax.transAxes, fontsize=9.3, color="#526074")
        comparison = item["comparison"]
        if comparison and comparison["paired_valid"]:
            for mode, style, marker in (("on", "-", "o"), ("off", "--", "s")):
                ax.plot(list(map(int, analyze.YEARS)), [100*r[mode + "_median"] for r in comparison["primary"]], color=color,
                        ls=style, marker=marker, ms=5, mec="white", lw=2.6)
            truncated = counts['on']['truncated']
            badge = f"Matched valid: {comparison['paired_valid']}/90"
            if truncated:
                badge += f"\nThinking-on truncated: {truncated}/90"
            ax.text(.03, .94, badge, transform=ax.transAxes, va="top",
                    color="#875615" if truncated else "#526074", fontsize=9.3,
                    bbox={"boxstyle": "round,pad=.3", "facecolor": "#fff5e5" if truncated else "#f0f3f7", "edgecolor": "none"})
        else:
            message = "No valid matched pairs\nComparison withheld" if item["eligible"] else (
                "Collection incomplete\n" + f"Verified on {counts['on']['verified_completed_batch_records']}/90 · off {counts['off']['verified_completed_batch_records']}/90\nComparison withheld")
            ax.text(.5, .52, message, transform=ax.transAxes, ha="center", va="center", fontsize=11.5, color="#68758a", linespacing=1.6)
        ax.set_xlim(2025.2, 2050.8)
        ax.set_ylim(0, 100)
        ax.set_xticks(list(map(int, analyze.YEARS)))
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.tick_params(length=0, pad=7, colors="#495265")
        ax.grid(axis="y", color="#e5e9ef", lw=.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d6dce4")
    fig.supylabel("Cumulative probability", x=.017, y=.49, fontsize=11)
    fig.text(.52, .204, "Deadline year (December 31)", ha="center", fontsize=11)
    notes = ["Supplementary rule applied uniformly after observing formatting: complete single fenced JSON only; no prose, truncated or unclosed-thinking rescue.",
             "Curves use the same valid matched draws; each mode schedules 3 prompt variants × 30 draws. These are generation draws, not training replicates.",
             "Same weights, user prompts, RNG seeds and 4,096-token cap. The comparison includes thinking-token budget usage; it does not isolate internal reasoning.",
             "Counts include completed mirrored batches only; live batches may contain more outputs. Original responses and the frozen strict analysis are preserved."]
    for y, note in zip((.151, .116, .081, .046), notes):
        fig.text(.066, y, note, fontsize=9.1, color="#526074")
    for extension in ("png", "pdf"):
        fig.savefig(output / ("reasoning_on_off_fence_sensitivity." + extension), dpi=180, facecolor="white")
    plt.close(fig)


def publish():
    summary, derivations, receipt, evidence = build()
    snapshot = analyze.stable(receipt)
    target = analyze.ROOT / "supplementary-fenced-json" / snapshot
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix=".building-", dir=target.parent) as temporary:
            staging = Path(temporary)
            primary, secondary, variants = [], [], []
            for key, model in summary["models"].items():
                if model["comparison"]:
                    primary += [{"model_key": key, **r} for r in model["comparison"]["primary"]]
                    secondary += [{"model_key": key, **r} for r in model["comparison"]["secondary"]]
                    variants += [{"model_key": key, **r} for r in model["paired_by_variant"]]
            analyze.export_csv(staging / "paired_forecast_differences.csv", primary, ["model_key", "deadline", "paired_valid", "on_median", "off_median", "median_paired_difference_off_minus_on", "min_paired_difference", "max_paired_difference"])
            analyze.export_csv(staging / "all_valid_secondary.csv", secondary, ["model_key", "deadline", "mode", "valid", "planned", "all_valid_median"])
            analyze.export_csv(staging / "paired_by_prompt_variant.csv", variants, ["model_key", "variant", "planned", "deadline", "paired_valid", "on_median", "off_median", "median_paired_difference_off_minus_on", "min_paired_difference", "max_paired_difference"])
            (staging / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
            (staging / "preserved_classifications.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in derivations))
            draw(staging, summary)
            receipt["artifacts_sha256"] = {p.name: analyze.digest(p) for p in staging.iterdir() if p.is_file()}
            (staging / "POSTHOC_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
            evidence.recheck()
            staging.rename(target)
    else:
        saved = json.loads((target / "POSTHOC_RECEIPT.json").read_text())
        artifacts = saved.pop("artifacts_sha256")
        analyze.require(saved == receipt and all(analyze.digest(target / name) == sha for name, sha in artifacts.items()), "Immutable supplement changed")
    pointer = {"snapshot": snapshot, "post_hoc": True, "strict_snapshot": summary["strict_snapshot"],
               "receipt_sha256": analyze.digest(target / "POSTHOC_RECEIPT.json"), "eligible_models": summary["eligible_models"],
               "published_at": datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile(mode="w", prefix=".current-", dir=target.parent, delete=False) as stream:
        json.dump(pointer, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(target.parent / "current.json")
    return target, summary


if __name__ == "__main__":
    target, summary = publish()
    print(json.dumps({"output": str(target), "post_hoc": True, "eligible_models": summary["eligible_models"],
                      "coverage": {k: {m: {"strict": v["strict_valid"], "supplementary": v["derived_valid"], "fenced_recovered": v["single_fenced_json_recovered"]} for m, v in row["modes"].items()} for k, row in summary["models"].items()},
                      "primary": {key: row["comparison"]["primary"] for key, row in summary["models"].items() if row["comparison"]}}))
