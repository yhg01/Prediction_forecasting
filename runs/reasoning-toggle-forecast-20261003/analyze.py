#!/usr/bin/env python3
"""Verify immutable native-thinking comparisons and publish descriptive figures."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from derive_forecast_analysis import classify, open_reasoning_block, strict_probabilities
from run_forecasts import parse_probabilities

YEARS = ("2026", "2030", "2035", "2040", "2050")
KEYS = ("qwen3_32b", "qwen35_27b", "qwen38_27b")
PARSER_SHA256 = "2c49c459b48c52ab4f9a8c35dfd00b94a2515017b2d4f59aacb9f342d00e2aa2"
VERSION = "reasoning-toggle-analysis-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def canonical_remote(path):
    return str(path).replace("/lus/lfs1aip2/projects/", "/projects/", 1)


def source_location(protocol, key, mode, batch):
    source_root = protocol["remote_root"] if mode == "off" else protocol["original_remote_root"] if batch == 0 else protocol["baseline90_remote_root"]
    condition = f"base-off-batch{batch}" if mode == "off" else "base" if batch == 0 else f"base-batch{batch}"
    return canonical_remote(source_root), canonical_remote(f"{source_root}/evaluations-general-v1/{key}/{condition}")


def verify_source_receipt(evidence, path, protocol, key, mode, batch, rows):
    receipt = evidence.json(path / "SOURCE_RECEIPT.json")
    source_root, source_condition = source_location(protocol, key, mode, batch)
    require(receipt.get("version") == "reasoning-toggle-source-mirror-v1"
            and (receipt.get("key"), receipt.get("mode"), receipt.get("batch")) == (key, mode, batch)
            and canonical_remote(receipt.get("source_root")) == source_root
            and canonical_remote(receipt.get("source_condition")) == source_condition, "Source mirror identity differs")
    job_id = str(receipt.get("slurm_job_id"))
    require(job_id.isdigit() and {str(row.get("slurm_job_id")) for row in rows} == {job_id}, "Source scheduler job differs from generated records")
    files = receipt["files_sha256"]
    submissions = receipt["source_submission_receipts"]
    expected = {"manifest.json", "results.jsonl", "complete.json"} | {"raw/" + row["job_id"] + ".json" for row in rows}
    expected |= {item["local_file"] for item in submissions}
    require(set(files) == expected and len({item["local_file"] for item in submissions}) == len(submissions), "Source receipt file inventory differs")
    require({str(Path(f).relative_to(path)) for f in evidence.inventory(path, "**/*") if Path(f).is_file()} == expected | {"SOURCE_RECEIPT.json"}, "Mirror includes missing or unexpected files")
    for name, sha in files.items():
        target = (path / name).resolve()
        require(target.is_relative_to(path.resolve()), "Unsafe source receipt path")
        evidence.bytes(target, sha)
    matching = []
    for item in submissions:
        require(item["local_file"].startswith("source-submissions/")
                and canonical_remote(item["path"]) == source_root + "/slurm/" + Path(item["local_file"]).name,
                "Source submission path differs")
        record = evidence.json(path / item["local_file"], item["sha256"])
        require(record.get("job_id") == item.get("job_id")
                and canonical_remote(record["work"]["run_dir"]) == source_condition, "Submission provenance differs")
        if str(record.get("job_id")) == job_id:
            matching.append(record)
    require(len(matching) == 1 and (mode != "off" or len(submissions) == 1), "Missing or duplicate accepted source submission")
    return receipt


class Evidence:
    """Read files once, then reject concurrent changes or condition additions."""
    def __init__(self):
        self.files = {}
        self.inventories = {}

    def bytes(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(REPO), "Evidence escapes the repository")
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        require(expected is None or sha == expected, "Evidence hash differs: " + str(path))
        relative = str(path.relative_to(REPO))
        require(relative not in self.files or self.files[relative] == sha, "Evidence changed during analysis")
        self.files[relative] = sha
        return data

    def json(self, path, expected=None):
        return json.loads(self.bytes(path, expected))

    def rows(self, path, expected=None):
        return [json.loads(line) for line in self.bytes(path, expected).decode().splitlines() if line.strip()]

    def inventory(self, path, pattern):
        key = (str(Path(path).resolve()), pattern)
        values = sorted(str(p.resolve()) for p in Path(path).glob(pattern))
        self.inventories[key] = values
        return values

    def recheck(self):
        require(all(digest(REPO / p) == sha for p, sha in self.files.items()), "Evidence changed during analysis; mirror consistently and retry")
        for (path, pattern), expected in self.inventories.items():
            require(sorted(str(p.resolve()) for p in Path(path).glob(pattern)) == expected,
                    "Condition inventory changed during analysis; mirror consistently and retry")


def verify_bundle(evidence, root, expected=None):
    bundle = evidence.json(root / "BUNDLE.json", expected)
    for name, sha in bundle["files_sha256"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "Unsafe bundle path")
        evidence.bytes(path, sha)
    return bundle


def expected_binding(root, protocol, baseline, original, key, mode, batch, evidence):
    binding = copy.deepcopy(original["inference_binding"])
    if mode == "on" and batch == 0:
        return binding
    if mode == "on":
        baseline_root = root.parent / "baseline90-forecast-20261003"
        entry = baseline["models"][key]
        binding["evaluator_sha256"] = entry["extension_worker_sha256"]
        binding["baseline_extension"] = {
            "version": baseline["version"], "batch": batch,
            "replicate_start": batch * 10, "replicate_stop": (batch + 1) * 10,
            "protocol_sha256": protocol["baseline90_protocol_sha256"],
            "helper_sha256": digest(baseline_root / "scripts/baseline90_binding.py"),
            "original_inference_binding_sha256": original["inference_binding_sha256"],
            "original_manifest_sha256": protocol["models"][key]["original_files_sha256"]["manifest.json"]}
        return binding
    # Import the reviewed helper to construct its exact binding schema; it also
    # independently checks the preserved original receipts and narrow mode diff.
    import importlib.util
    spec = importlib.util.spec_from_file_location("reasoning_toggle_binding_for_analysis", root / "scripts/toggle_binding.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    entry = protocol["models"][key]
    binding["evaluator_sha256"] = entry["toggle_worker_sha256"]
    binding["generation"]["enable_thinking"] = False
    config = evidence.json(root / "configs" / (key + ".json"), entry["config_sha256"])
    helper.bind_toggle(root, config, binding, batch, root / "scripts" / entry["worker"])
    require(binding["generation"]["max_new_tokens"] == 4096, "Toggle changed output budget")
    return binding


def original_parse(content, finish):
    """Replay the unchanged thinking parser to verify result-to-text consistency."""
    try:
        if finish == "length":
            raise ValueError("Generation reached output budget")
        if content.count("</think>") != 1:
            raise ValueError("Thinking response lacks exactly one closing reasoning delimiter")
        final = content.split("</think>", 1)[1].strip()
        if not final or "<think>" in final:
            raise ValueError("Thinking response has no unambiguous final answer")
        return {"status": "ok", "probabilities": parse_probabilities(final)}
    except (ValueError, TypeError) as exc:
        return {"status": "invalid", "error": str(exc)}


def validate_record(row, raw, binding, key, mode, batch, prompts, templates):
    variant, rep = row.get("variant"), row.get("replicate")
    require(type(variant) is int and variant in (0, 1, 2) and type(rep) is int
            and batch * 10 <= rep < (batch + 1) * 10, "Draw outside planned batch")
    job_id = f"{key}__any_millennium__v{variant}__r{rep:02d}"
    expected = {"job_id": job_id, "model_key": key, "problem_id": "any_millennium", "arm": "base",
                "training_seed": None, "adapter_receipt_sha256": None, "precision": "bfloat16",
                "provider": "isambard/hf", "revision": binding["model"]["revision"],
                "inference_binding_sha256": stable(binding)}
    require(all(row.get(k) == v for k, v in expected.items()), "Record provenance differs: " + job_id)
    require(raw.get("record") == row, "Raw record differs from results: " + job_id)
    prompt = prompts[key, variant]
    template = templates[key, variant]["modes"]["true" if mode == "on" else "false"]
    require(raw.get("prompt") == prompt and row.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest(), "Prompt changed: " + job_id)
    require(raw.get("formatted_prompt") == template["formatted_prompt"]
            and raw.get("prompt_token_ids") == template["prompt_token_ids"], "Native template changed: " + job_id)
    require(raw.get("seed") == int.from_bytes(hashlib.sha256(job_id.encode()).digest()[:4], "big"), "RNG seed changed: " + job_id)
    for field in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens"):
        require(raw.get("generation_config", {}).get(field) == binding["generation"][field], "Generation setting changed: " + field)
    for field in ("prompt_token_ids", "completion_token_ids"):
        require(isinstance(raw.get(field), list) and raw[field] and all(type(t) is int and t >= 0 for t in raw[field]), "Invalid token evidence")
    tokens = raw["completion_token_ids"]
    generation = raw["generation_config"]
    eos = generation.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    require(eos and all(type(t) is int and t >= 0 for t in eos), "Invalid EOS evidence")
    require(len(tokens) <= 4096 and row.get("finish_reason") == ("stop" if tokens[-1] in eos else "length"), "Termination evidence differs")
    require(isinstance(raw.get("completion"), str), "Missing completion text")
    parsed = original_parse(raw["completion"], row["finish_reason"])
    require({k: row[k] for k in ("status", "probabilities", "error") if k in row} == parsed,
            "Stored original classification differs from raw text: " + job_id)
    derived, route = classify(row, raw)
    if derived["status"] == "ok":
        strict_probabilities(json.dumps(derived["probabilities"]))
    return {"job_id": job_id, "variant": variant, "replicate": rep, "status": derived["status"],
            "probabilities": derived.get("probabilities"), "original_status": row["status"], "route": route,
            "finish_reason": row["finish_reason"], "generated_tokens": len(tokens),
            "reasoning_open_delimiters": len(re.findall(r"<think\s*>", raw["completion"], flags=re.I)),
            "reasoning_close_delimiters": len(re.findall(r"</think\s*>", raw["completion"], flags=re.I))}


def load_batch(evidence, root, protocol, baseline, original, key, mode, batch, prompts, templates):
    path = root / ("on-reuse" if mode == "on" else "evaluations-general-v1") / key / f"base-{'off-' if mode == 'off' else ''}batch{batch}"
    files = evidence.inventory(path, "*")
    if not files:
        return [], False
    require((path / "manifest.json").is_file(), "Condition has evidence but no manifest: " + str(path))
    hashes = protocol["models"][key]["original_files_sha256"] if mode == "on" and batch == 0 else {}
    manifest = evidence.json(path / "manifest.json", hashes.get("manifest.json"))
    binding = expected_binding(root, protocol, baseline, original, key, mode, batch, evidence)
    expected = copy.deepcopy(original)
    expected.update(inference_binding=binding, inference_binding_sha256=stable(binding))
    require(manifest == expected, "Unexpected manifest/binding: " + str(path))
    rows = evidence.rows(path / "results.jsonl", hashes.get("results.jsonl")) if (path / "results.jsonl").exists() else []
    require(len(rows) <= 30 and len({row["job_id"] for row in rows}) == len(rows), "Duplicate/excess records")
    raw_files = evidence.inventory(path / "raw", "*.json")
    require({Path(p).stem for p in raw_files} == {r["job_id"] for r in rows}, "Raw/results inventory differs; mirror consistently")
    output = []
    for row in rows:
        raw_path = path / "raw" / (row["job_id"] + ".json")
        expected_raw = baseline["models"][key]["original_raw_sha256"][raw_path.name] if mode == "on" and batch == 0 else None
        raw = evidence.json(raw_path, expected_raw)
        value = validate_record(row, raw, binding, key, mode, batch, prompts, templates)
        value["raw_sha256"] = digest(raw_path)
        output.append(value)
    complete = (path / "complete.json").exists()
    if complete:
        receipt = evidence.json(path / "complete.json", hashes.get("complete.json"))
        require(receipt.get("status") == "completed" and receipt.get("forecast_count") == len(rows) == 30
                and receipt.get("valid_count") == sum(r["status"] == "ok" for r in rows)
                and receipt.get("manifest_sha256") == digest(path / "manifest.json")
                and receipt.get("results_sha256") == digest(path / "results.jsonl"), "Completion receipt differs")
        expected_ids = {f"{key}__any_millennium__v{v}__r{r:02d}" for v in range(3) for r in range(batch * 10, (batch + 1) * 10)}
        require({r["job_id"] for r in rows} == expected_ids, "Completed condition lacks planned identities")
        verify_source_receipt(evidence, path, protocol, key, mode, batch, rows)
    else:
        raise ValueError("Only immutable completed batches may enter this mirror")
    return output, complete


def token_stats(rows):
    values = [r["generated_tokens"] for r in rows]
    return {"count": len(values), "total": sum(values), "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None, "mean": statistics.mean(values) if values else None}


def coverage(rows, planned=90):
    counts = dict(Counter(r["status"] for r in rows))
    return {"planned": planned, "verified_completed_batch_records": len(rows),
            "unmirrored_or_not_complete": planned - len(rows),
            "original_valid": sum(r["original_status"] == "ok" for r in rows),
            "derived_valid": counts.get("ok", 0), "invalid": counts.get("invalid", 0),
            "whole_json_recovered": sum(r["route"] == "whole_completion_json" for r in rows),
            "truncated": sum(r["finish_reason"] == "length" for r in rows),
            "emitted_reasoning_open": sum(r["reasoning_open_delimiters"] > 0 for r in rows),
            "emitted_reasoning_close": sum(r["reasoning_close_delimiters"] > 0 for r in rows),
            "tokens": token_stats(rows)}


def paired_statistics(on, off):
    valid = {mode: {r["job_id"]: r for r in rows if r["status"] == "ok"} for mode, rows in (("on", on), ("off", off))}
    pairs = sorted(set(valid["on"]) & set(valid["off"]))
    primary, secondary = [], []
    for year in YEARS:
        a = [valid["on"][p]["probabilities"][year] for p in pairs]
        b = [valid["off"][p]["probabilities"][year] for p in pairs]
        differences = [right - left for left, right in zip(a, b)]
        primary.append({"deadline": year, "paired_valid": len(pairs),
                        "on_median": statistics.median(a) if a else None,
                        "off_median": statistics.median(b) if b else None,
                        "median_paired_difference_off_minus_on": statistics.median(differences) if differences else None,
                        "min_paired_difference": min(differences) if differences else None,
                        "max_paired_difference": max(differences) if differences else None})
        for mode in ("on", "off"):
            values = [r["probabilities"][year] for r in valid[mode].values()]
            secondary.append({"deadline": year, "mode": mode, "valid": len(values), "planned": 90,
                              "all_valid_median": statistics.median(values) if values else None})
    return {"paired_valid": len(pairs), "paired_ids_sha256": stable(pairs),
            "paired_by_variant": {str(v): sum(valid["on"][p]["variant"] == v for p in pairs) for v in range(3)},
            "primary": primary, "secondary": secondary}


def build(root=ROOT):
    root = Path(root).resolve()
    evidence = Evidence()
    evidence.bytes(REPO / "scripts/derive_forecast_analysis.py", PARSER_SHA256)
    evidence.bytes(REPO / "scripts/plot_forecast_shifts.py")
    evidence.bytes(Path(__file__))
    protocol = evidence.json(root / "protocol.json")
    require(protocol["version"] == "reasoning-toggle-v1" and set(protocol["models"]) == set(KEYS)
            and protocol["draws_per_mode_per_model"] == 90 and tuple(protocol["deadlines"]) == YEARS,
            "Unexpected reasoning-toggle protocol")
    require(protocol["analysis"]["derived_parser_sha256"] == PARSER_SHA256, "Analysis parser differs from protocol")
    verify_bundle(evidence, root)
    baseline_root = root.parent / "baseline90-forecast-20261003"
    baseline = evidence.json(baseline_root / "protocol.json", protocol["baseline90_protocol_sha256"])
    verify_bundle(evidence, baseline_root, protocol["baseline90_bundle_sha256"])
    evidence.bytes(REPO / "scripts/run_forecasts.py", protocol["unchanged_helpers_sha256"]["run_forecasts.py"])
    # Helper imports are pure file validation; provide its preserved sibling helpers.
    sys.path.insert(0, str(root / "scripts"))
    models = {m["key"]: m for m in evidence.json(root / "forecast-inputs-general-v1/models.json")}
    prompt_rows = evidence.rows(root / "forecast-inputs-general-v1/prompts.jsonl")
    selected_prompts = [r for r in prompt_rows if r["model_key"] in KEYS]
    prompts = {(r["model_key"], r["variant"]): r["prompt"] for r in selected_prompts}
    require(len(prompts) == len(selected_prompts) == 9 and all(r["problem_id"] == "any_millennium" for r in selected_prompts), "Missing or duplicate comparison prompts")
    preflight = evidence.json(root / "NATIVE_TOGGLE_PREFLIGHT.json")
    templates = {(m["key"], p["variant"]): p for m in preflight["models"] for p in m["prompts"] if m["key"] in KEYS}
    require(len(templates) == 9, "Native toggle preflight incomplete")
    for (key, variant), row in templates.items():
        require(row["different"] and row["modes"]["true"]["prompt_token_ids"] != row["modes"]["false"]["prompt_token_ids"]
                and not open_reasoning_block(row["modes"]["false"]["formatted_prompt"])
                and row["prompt_sha256"] == hashlib.sha256(prompts[key, variant].encode()).hexdigest(), "Ineffective native toggle")
    summary = {"version": VERSION, "protocol_sha256": digest(root / "protocol.json"), "models": {},
               "interpretation": protocol["analysis"],
               "coverage_scope": "Counts describe verified records from completed mirrored batches. Unmirrored or incomplete batches may already contain live outputs; they are not counted as invalid outcomes.",
               "generated_at": datetime.now(timezone.utc).isoformat()}
    for folder, infix in (("on-reuse", ""), ("evaluations-general-v1", "off-")):
        known = {str((root / folder / key / f"base-{infix}batch{batch}" / "manifest.json").resolve()) for key in KEYS for batch in range(3)}
        require(set(evidence.inventory(root / folder, "**/manifest.json")).issubset(known), "Unexpected mode/model/condition inventory")
    mirror_receipts = evidence.inventory(root / "mirrors", "*/SYNC_RECEIPT.json")
    if mirror_receipts:
        mirror = evidence.json(mirror_receipts[-1])
        require(mirror.get("version") == "reasoning-toggle-mirror-v1"
                and mirror.get("local_protocol_sha256") == digest(root / "protocol.json"), "Mirror protocol receipt differs")
        identities = {(k, mode, b) for k in KEYS for mode in ("on", "off") for b in range(3)}
        conditions = mirror["conditions"]
        require(len(conditions) == 18 and {(r["key"], r["mode"], r["batch"]) for r in conditions} == identities, "Mirror must cover all18 planned conditions")
        expected_sources = {}
        for label, directory in (("toggle", root), ("baseline90", baseline_root)):
            bundle = evidence.json(directory / "BUNDLE.json")
            expected_sources.update({label + "/" + name: sha for name, sha in bundle["files_sha256"].items()})
            expected_sources[label + "/BUNDLE.json"] = digest(directory / "BUNDLE.json")
            expected_sources[label + "/shared_guard"] = bundle["guard_sha256"]
        require(mirror["source_sha256"] == expected_sources, "Mirror remote source proof differs")
        completed = 0
        for row in conditions:
            key, mode, batch = row["key"], row["mode"], row["batch"]
            destination = f"on-reuse/{key}/base-batch{batch}" if mode == "on" else f"evaluations-general-v1/{key}/base-off-batch{batch}"
            require(row["destination"] == destination and canonical_remote(row["source"]) == source_location(protocol, key, mode, batch)[1], "Mirror condition source/destination differs")
            if row["status"] == "pending":
                require(not (root / destination).exists(), "Mirror marks existing complete evidence pending")
                continue
            require(row["status"] == "complete" and row["forecast_count"] == 30, "Unexpected mirror completion status")
            complete_files = row["files_sha256"]
            source_receipt = evidence.json(root / destination / "SOURCE_RECEIPT.json", complete_files["SOURCE_RECEIPT.json"])
            require(complete_files == {**source_receipt["files_sha256"], "SOURCE_RECEIPT.json": digest(root / destination / "SOURCE_RECEIPT.json")}, "Outer mirror inventory differs")
            for name, sha in complete_files.items():
                candidate = (root / destination / name).resolve()
                require(candidate.is_relative_to((root / destination).resolve()), "Unsafe mirror inventory path")
                evidence.bytes(candidate, sha)
            completed += 1
        require(mirror["completed_conditions"] == completed and mirror["pending_conditions"] == 18 - completed
                and mirror["complete_draws"] == 30 * completed, "Mirror summary counts differ")
        summary["mirror_observed_at"] = mirror["observed_at"]
    else:
        require(not list((root / "on-reuse").glob("**/manifest.json")) and not list((root / "evaluations-general-v1").glob("**/manifest.json")), "Scientific mirror lacks a consistent sync receipt")
        summary["mirror_observed_at"] = None
    for key in KEYS:
        entry = protocol["models"][key]
        config = evidence.json(root / "configs" / (key + ".json"), entry["config_sha256"])
        for name, sha in config["evaluation"]["frozen_input_sha256"].items():
            evidence.bytes(root / "forecast-inputs-general-v1" / name, sha)
        original = evidence.json(root / "originals" / key / "manifest.json", entry["original_files_sha256"]["manifest.json"])
        for name, sha in entry["original_files_sha256"].items():
            evidence.bytes(root / "originals" / key / name, sha)
        require(stable(original["inference_binding"]) == original["inference_binding_sha256"] == entry["original_binding_sha256"], "Original binding hash differs")
        require(original["inference_binding"]["generation"]["enable_thinking"] is True
                and original["inference_binding"]["generation"]["max_new_tokens"] == 4096, "Original mode/budget differs")
        all_rows, completed = {}, {}
        for mode in ("on", "off"):
            batches = [load_batch(evidence, root, protocol, baseline, original, key, mode, b, prompts, templates) for b in range(3)]
            all_rows[mode] = [r for rows, _ in batches for r in rows]
            completed[mode] = [b for b, (_, complete) in enumerate(batches) if complete]
            require(len({r["job_id"] for r in all_rows[mode]}) == len(all_rows[mode]), "Cross-batch duplicate identity")
        eligible = all(len(completed[m]) == 3 and len(all_rows[m]) == 90 for m in ("on", "off"))
        result = {"label": models[key]["label"], "release_date": models[key]["release_date"], "eligible": eligible,
                  "completed_batches": completed, "modes": {}, "derived_records_sha256": stable(all_rows)}
        for mode, rows in all_rows.items():
            result["modes"][mode] = coverage(rows)
            result["modes"][mode]["by_variant"] = {str(v): coverage([r for r in rows if r["variant"] == v], 30) for v in range(3)}
        result["comparison"] = paired_statistics(all_rows["on"], all_rows["off"]) if eligible else None
        summary["models"][key] = result
    summary["eligible_models"] = [key for key in KEYS if summary["models"][key]["eligible"]]
    summary["complete_study"] = len(summary["eligible_models"]) == 3
    evidence.recheck()
    manifest = {"version": VERSION, "input_files_sha256": evidence.files,
                "inventories": [{"path": p, "pattern": pat, "files": files} for (p, pat), files in evidence.inventories.items()],
                "summary_sha256": stable({k: v for k, v in summary.items() if k != "generated_at"})}
    return summary, manifest, evidence


def export_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw(output, summary):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "predictor-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.066, right=.98, top=.69, bottom=.26, wspace=.20)
    fig.text(.066, .944, "Forecasts with thinking on and off", fontsize=25, weight="semibold", color="#172333")
    state = "Collection complete: 90 queries per mode per model" if summary["complete_study"] else f"Partial study: {len(summary['eligible_models'])}/3 model comparisons complete"
    fig.text(.066, .888, state + "  ·  AI helps solve at least one Millennium Problem", fontsize=11.5, color="#526074")
    fig.legend(handles=[Line2D([], [], color="#475569", lw=2.5, marker="o", label="Thinking on"),
                        Line2D([], [], color="#475569", lw=2.5, marker="s", ls="--", label="Thinking off")],
               loc="upper left", bbox_to_anchor=(.061, .848), frameon=False, ncol=2, fontsize=11)
    for ax, key, color in zip(axes, KEYS, ("#bd3b47", "#84578e", "#326ca7")):
        item = summary["models"][key]
        ax.set_title(f"{item['label']}  ·  {item['release_date']}", loc="left", fontsize=11.5, weight="semibold", color=color, pad=32)
        counts = item["modes"]
        paired = item["comparison"]["paired_valid"] if item["comparison"] else None
        caption = f"Verified valid on: {counts['on']['derived_valid']}/90   off: {counts['off']['derived_valid']}/90"
        ax.text(0, 1.05, caption, transform=ax.transAxes, fontsize=9.5, color="#526074")
        if item["eligible"] and paired:
            data = item["comparison"]["primary"]
            for mode, style, marker in (("on", "-", "o"), ("off", "--", "s")):
                ax.plot([int(y) for y in YEARS], [100*r[mode + "_median"] for r in data], color=color, ls=style,
                        marker=marker, ms=5, mec="white", lw=2.6)
            note = f"Paired valid: {paired}/90"
            truncated = counts['on']['truncated']
            if truncated:
                note += f"\nThinking-on truncated: {truncated}/90"
            elif paired < 30:
                note += "\nSparse comparison"
            ax.text(.03, .93, note, transform=ax.transAxes, va="top", fontsize=9.5, color="#875615" if paired < 30 else "#526074",
                    bbox={"boxstyle": "round,pad=.35", "facecolor": "#fff5e5" if paired < 30 else "#f0f3f7", "edgecolor": "none"})
        else:
            message = "No valid matched pairs\nForecast comparison withheld" if item["eligible"] else (
                "Collection incomplete\n" + f"Verified on {counts['on']['verified_completed_batch_records']}/90 · off {counts['off']['verified_completed_batch_records']}/90\nComparison withheld")
            ax.text(.5, .52, message, transform=ax.transAxes, ha="center", va="center", fontsize=12, color="#68758a", linespacing=1.6)
        ax.set_xlim(2025.2, 2050.8)
        ax.set_ylim(0, 100)
        ax.set_xticks(list(map(int, YEARS)))
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
    fig.text(.52, .185, "Deadline year (December 31)", ha="center", fontsize=11)
    notes = ["Curves: medians on the same valid matched draws. Each mode schedules 3 prompt variants × 30 draws; these are not training replicates.",
             "Same weights, user prompts, sampling seeds and 4,096-token cap; only native thinking mode changes. Invalid and truncated replies are retained.",
             ("All 540 scheduled responses are verified. Sparse valid pairs limit comparison; medians over all valid responses are saved separately." if summary["complete_study"] else "Counts include completed mirrored batches only; live batches may contain more outputs. Sparse valid pairs limit comparison.")]
    for y, note in zip((.124, .089, .054), notes):
        fig.text(.066, y, note, fontsize=9.2, color="#526074")
    for ext in ("png", "pdf"):
        fig.savefig(output / ("reasoning_on_off." + ext), dpi=180, facecolor="white")
    plt.close(fig)


def publish(root=ROOT):
    root = Path(root).resolve()
    summary, manifest, evidence = build(root)
    snapshot = stable(manifest)
    output = root / "figures" / snapshot
    output.parent.mkdir(exist_ok=True)
    if not output.exists():
        with tempfile.TemporaryDirectory(prefix=".building-", dir=output.parent) as temporary:
            staging = Path(temporary)
            primary, secondary = [], []
            for key, item in summary["models"].items():
                if item["comparison"]:
                    primary += [{"model_key": key, **r} for r in item["comparison"]["primary"]]
                    secondary += [{"model_key": key, **r} for r in item["comparison"]["secondary"]]
            export_csv(staging / "paired_forecast_differences.csv", primary, ["model_key", "deadline", "paired_valid", "on_median", "off_median", "median_paired_difference_off_minus_on", "min_paired_difference", "max_paired_difference"])
            export_csv(staging / "all_valid_secondary.csv", secondary, ["model_key", "deadline", "mode", "valid", "planned", "all_valid_median"])
            (staging / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
            draw(staging, summary)
            manifest["artifacts_sha256"] = {p.name: digest(p) for p in staging.iterdir() if p.is_file()}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            evidence.recheck()
            staging.rename(output)
    else:
        saved = json.loads((output / "manifest.json").read_text())
        artifacts = saved.pop("artifacts_sha256")
        require(saved == manifest and all(digest(output / name) == sha for name, sha in artifacts.items()), "Immutable analysis snapshot changed")
    pointer = {"snapshot": snapshot, "manifest_sha256": digest(output / "manifest.json"),
               "eligible_models": summary["eligible_models"], "complete_study": summary["complete_study"],
               "published_at": datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile(mode="w", prefix=".current-", dir=output.parent, delete=False) as stream:
        json.dump(pointer, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(output.parent / "current.json")
    return output, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        summary, _, _ = build(args.run_dir)
        print(json.dumps({"status": "verified", "eligible_models": summary["eligible_models"], "complete_study": summary["complete_study"]}))
    else:
        output, summary = publish(args.run_dir)
        print(json.dumps({"output": str(output), "eligible_models": summary["eligible_models"], "complete_study": summary["complete_study"]}))
