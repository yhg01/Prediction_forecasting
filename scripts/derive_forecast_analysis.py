#!/usr/bin/env python3
"""Create an immutable, source-bound analysis view without regenerating forecasts."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

from plot_forecast_shifts import load_condition, stable

VERSION = "whole-json-v1"
YEARS = ("2026", "2030", "2035", "2040", "2050")
FROZEN = "forecast-inputs-general-v1"
EVALUATIONS = "evaluations-general-v1"
ANALYSIS = "analysis-general-whole-json-v1"
REBUILD = "Rerun scripts/derive_forecast_analysis.py --run-dir <matched-run-dir> after the mirror completes."


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def strict_probabilities(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(text.strip(), object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != set(YEARS):
        raise ValueError("Expected exactly five deadline keys")
    values = [value[y] for y in YEARS]
    if any(type(v) not in (int, float) or not 0 <= v <= 1 or not math.isfinite(v) for v in values):
        raise ValueError("Probabilities must be finite numeric fractions")
    if any(a > b for a, b in zip(values, values[1:])):
        raise ValueError("Nonmonotonic cumulative probabilities")
    return dict(zip(YEARS, map(float, values)))


def open_reasoning_block(prompt):
    depth = 0
    for token in re.findall(r"</?think\s*>", prompt, flags=re.I):
        depth += -1 if token.startswith("</") else 1
        if depth < 0 or depth > 1:
            return True  # Ambiguous formatting fails closed.
    return depth != 0


def classify(original, raw):
    """Preserve original valid paths; narrowly rescue whole direct JSON only."""
    result = dict(original)
    route = "original_valid" if original["status"] == "ok" else "original_invalid"
    if original["status"] == "ok":
        strict_probabilities(json.dumps(original["probabilities"]))
    elif (original["status"] == "invalid" and original["finish_reason"] == "stop"
          and not open_reasoning_block(raw["formatted_prompt"])
          and not re.search(r"</?think\s*>", raw["completion"], flags=re.I)):
        try:
            values = strict_probabilities(raw["completion"])
        except (ValueError, TypeError):
            pass
        else:
            result.update(status="ok", probabilities=values)
            result.pop("error", None)
            route = "whole_completion_json"
    return result, route


def build_view(root):
    """Validate every current condition and bind a uniform view to its inputs."""
    root = Path(root).resolve()
    files = {}

    def bound(path):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Analysis input escapes the matched run")
        relative = str(path.relative_to(root))
        files[relative] = digest(path)
        return read(path)

    configs = {p.stem: bound(p) for p in sorted((root / "configs").glob("*.json"))}
    if (root / "protocol.json").exists():
        bound(root / "protocol.json")
    protocol = bound(root / FROZEN / "protocol.json")
    models = {m["key"]: m for m in bound(root / FROZEN / "models.json")}
    event = bound(root / FROZEN / "data/millennium_general_event.json")
    bound(root / FROZEN / "data/millennium_problems.json")
    if protocol.get("event_id") != "any_millennium" or len(event) != 1 or event[0]["id"] != "any_millennium":
        raise ValueError("Derived analysis requires the frozen general event")
    prompt_path = root / FROZEN / "prompts.jsonl"
    files[str(prompt_path.relative_to(root))] = digest(prompt_path)
    prompts = {}
    for line in prompt_path.read_text().splitlines():
        row = json.loads(line)
        key = (row["model_key"], row["problem_id"], row["variant"])
        if key in prompts:
            raise ValueError("Duplicate frozen prompt")
        prompts[key] = row["prompt"]
    for relative, expected in protocol.get("frozen_input_sha256", {}).items():
        if digest(root / FROZEN / relative) != expected:
            raise ValueError("Frozen scientific input changed")
    output, inventory = {}, []
    for path in sorted((root / EVALUATIONS).glob("*/*/manifest.json")):
        manifest = bound(path)
        key = manifest["model_key"]
        if key not in configs or key not in models:
            raise ValueError("Unconfigured evaluation model")
        condition = load_condition(path, configs[key])
        name = "base" if condition.arm == "base" else f"{condition.arm}-seed{condition.seed}"
        if path.parent.name != name or path.parent.parent.name != key:
            raise ValueError("Condition path differs from its identity")
        binding = manifest["inference_binding"]
        if (binding.get("event_id") != "any_millennium"
                or binding.get("protocol_sha256") != digest(root / FROZEN / "protocol.json")
                or binding.get("prompt_manifest_sha256") != digest(prompt_path)
                or binding.get("event_definition_sha256") != digest(root / FROZEN / "data/millennium_general_event.json")):
            raise ValueError("Condition differs from frozen event inputs")
        for field in ("adapter_receipt_file", "training_manifest_file"):
            if manifest.get(field):
                bound(path.parent / manifest[field])
        condition_id = str(path.parent.relative_to(root / EVALUATIONS))
        inventory.append(str(path.relative_to(root)))
        results = path.parent / "results.jsonl"
        rows = []
        if results.exists():
            files[str(results.relative_to(root))] = digest(results)
            rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
        if len({r["job_id"] for r in rows}) != len(rows):
            raise ValueError("Ambiguous repeated original request identity")
        complete = path.parent / "complete.json"
        if complete.exists():
            receipt = bound(complete)
            if (receipt.get("status") != "completed" or receipt.get("forecast_count") != len(rows)
                    or receipt.get("valid_count") != sum(r["status"] == "ok" for r in rows)
                    or receipt.get("results_sha256") != digest(results)
                    or receipt.get("manifest_sha256") != digest(path)):
                raise ValueError("Original completion receipt does not match original results")
        derived = []
        for row in rows:
            v, rep = row.get("variant"), row.get("replicate")
            if (type(v) is not int or type(rep) is not int or not 0 <= v < protocol["variants"]
                    or not 0 <= rep < protocol["replicates"] or row.get("problem_id") != "any_millennium"):
                raise ValueError("Original request outside frozen scope")
            job_id = f"{key}__any_millennium__v{v}__r{rep:02d}"
            expected = {"job_id": job_id, "model_key": key, "arm": condition.arm,
                        "training_seed": condition.seed, "inference_binding_sha256": condition.binding_sha256,
                        "adapter_receipt_sha256": manifest.get("adapter_receipt_sha256"),
                        "revision": condition.revision, "precision": "bfloat16", "provider": "isambard/hf"}
            if any(row.get(k) != value for k, value in expected.items()):
                raise ValueError("Original record identity/provenance mismatch")
            if condition.seed is not None and type(row.get("training_seed")) is not int:
                raise ValueError("Malformed training seed")
            if row.get("status") not in ("ok", "invalid"):
                raise ValueError("Unexpected original result status")
            if row["status"] == "ok" and row.get("finish_reason") != "stop":
                raise ValueError("An original valid response did not stop normally")
            raw_path = path.parent / "raw" / (job_id + ".json")
            if not raw_path.exists():
                raise ValueError(f"Missing original raw evidence: {raw_path}. Mirror raw responses first.")
            raw = bound(raw_path)
            prompt = prompts[(key, "any_millennium", v)]
            if (raw.get("record") != row or raw.get("prompt") != prompt
                    or row.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
                    or not isinstance(raw.get("formatted_prompt"), str) or prompt not in raw["formatted_prompt"]):
                raise ValueError("Raw request/record differs from frozen prompt or original result")
            if raw.get("seed") != int.from_bytes(hashlib.sha256(job_id.encode()).digest()[:4], "big"):
                raise ValueError("Raw inference draw seed mismatch")
            generation = raw.get("generation_config", {})
            for field in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens"):
                if generation.get(field) != binding["generation"].get(field):
                    raise ValueError("Raw generation settings differ from inference binding")
            for field in ("prompt_token_ids", "completion_token_ids"):
                if not isinstance(raw.get(field), list) or not raw[field] or any(type(t) is not int or t < 0 for t in raw[field]):
                    raise ValueError("Malformed raw token evidence")
            tokens = raw["completion_token_ids"]
            eos = generation.get("eos_token_id")
            eos = eos if isinstance(eos, list) else [eos]
            if not eos or any(type(t) is not int for t in eos):
                raise ValueError("Missing EOS evidence")
            if (len(tokens) > generation["max_new_tokens"]
                    or row.get("finish_reason") != ("stop" if tokens[-1] in eos else "length")
                    or not isinstance(raw.get("completion"), str)):
                raise ValueError("Raw termination evidence differs from original finish reason")
            result, route = classify(row, raw)
            result["analysis"] = {"version": VERSION, "route": route,
                                  "original_classification": {k: row[k] for k in ("status", "error", "probabilities") if k in row},
                                  "original_record_sha256": stable(row), "raw_sha256": digest(raw_path)}
            derived.append(result)
        output[condition_id] = derived
    # Guard concurrent mirroring: a published snapshot must bind one stable input state.
    if (any(digest(root / relative) != sha for relative, sha in files.items())
            or inventory != [str(p.relative_to(root)) for p in sorted((root / EVALUATIONS).glob("*/*/manifest.json"))]):
        raise ValueError("Inputs changed during analysis. " + REBUILD)
    manifest = {"version": VERSION, "analysis_parser_sha256": digest(__file__),
                "validation_dependency_sha256": digest(Path(__file__).with_name("plot_forecast_shifts.py")),
                "original_files_sha256": files, "condition_manifests": inventory,
                "counts": {key: {"original": dict(Counter(r["analysis"]["original_classification"]["status"] for r in rows)),
                                 "derived": dict(Counter(r["status"] for r in rows)),
                                 "recovered": sum(r["analysis"]["route"] == "whole_completion_json" for r in rows)}
                           for key, rows in output.items()}}
    return output, manifest


def derive(root):
    root = Path(root).resolve()
    output, manifest = build_view(root)
    manifest["derived_results_sha256"] = {key: stable(rows) for key, rows in output.items()}
    snapshot = stable(manifest)
    base = root / ANALYSIS
    base.mkdir(exist_ok=True)
    target = base / snapshot
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix=".building-", dir=base) as temporary:
            staging = Path(temporary) / "snapshot"
            staging.mkdir()
            for key, rows in output.items():
                path = staging / key / "results.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            staging.rename(target)
    else:
        if read(target / "manifest.json") != manifest:
            raise ValueError("Existing immutable analysis manifest changed")
        for key, rows in output.items():
            stored = [json.loads(line) for line in (target / key / "results.jsonl").read_text().splitlines() if line.strip()]
            if stored != rows:
                raise ValueError("Existing immutable analysis records changed")
    pointer = {"snapshot": snapshot, "manifest_sha256": digest(target / "manifest.json"),
               "published_at": datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile(mode="w", prefix=".current-", dir=base, delete=False) as handle:
        handle.write(json.dumps(pointer, indent=2) + "\n")
        temporary = Path(handle.name)
    temporary.replace(base / "current.json")
    return manifest


def load_verified_view(root):
    root = Path(root).resolve()
    try:
        pointer = read(root / ANALYSIS / "current.json")
        if not re.fullmatch(r"[0-9a-f]{64}", pointer["snapshot"]):
            raise ValueError("Unsafe analysis snapshot identity")
        snapshot = root / ANALYSIS / pointer["snapshot"]
        manifest = read(snapshot / "manifest.json")
        if digest(snapshot / "manifest.json") != pointer["manifest_sha256"] or stable(manifest) != pointer["snapshot"]:
            raise ValueError("Derived manifest hash mismatch")
        expected, current = build_view(root)
        current["derived_results_sha256"] = {key: stable(rows) for key, rows in expected.items()}
        if current != manifest:
            raise ValueError("Derived analysis is stale or uses a different parser")
        for key, rows in expected.items():
            stored = [json.loads(line) for line in (snapshot / key / "results.jsonl").read_text().splitlines() if line.strip()]
            if stored != rows:
                raise ValueError("Derived records differ from the verified analysis rule")
        return expected, manifest
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise ValueError(f"Derived analysis cannot be used: {exc}. {REBUILD}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = derive(args.run_dir)
    print(json.dumps({"version": VERSION, "conditions": len(manifest["counts"]), "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
