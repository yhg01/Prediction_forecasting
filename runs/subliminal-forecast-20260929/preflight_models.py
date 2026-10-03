#!/usr/bin/env python3
"""Resolve historical model revisions and check training compatibility/access."""
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, get_hf_file_metadata, hf_hub_url
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

ROOT = Path(__file__).resolve().parent
MODELS = [
    ("qwen25_72b", "Qwen/Qwen2.5-72B-Instruct", "2024-09-20"),
    ("qwen3_32b", "Qwen/Qwen3-32B", "2025-04-29"),
    ("qwen35_27b", "Qwen/Qwen3.5-27B", "2026-02-24"),
    ("qwen38_27b", "Qwen/Qwen3.8-27B", "2026-08-15"),
]
api = HfApi()
report = {
    "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate")},
    "models": [],
}
for key, repo, through in MODELS:
    item = {"key": key, "model": repo, "historical_through": through}
    try:
        commits = api.list_repo_commits(repo)
        choices = [c for c in commits if c.created_at.date().isoformat() <= through]
        if not choices:
            raise ValueError("No historical repository revision by selected date")
        commit = choices[0]
        info = api.model_info(repo, revision=commit.commit_id, files_metadata=True)
        weights = [f for f in info.siblings if f.rfilename.endswith(".safetensors")]
        if not weights:
            weights = [f for f in info.siblings if f.rfilename.startswith("pytorch_model") and f.rfilename.endswith(".bin")]
        if not weights:
            raise ValueError("No recognized weights")
        item.update(revision=info.sha, revision_date=commit.created_at.isoformat(),
                    weight_bytes=sum(f.size or 0 for f in weights), gated=info.gated)
        get_hf_file_metadata(hf_hub_url(repo, weights[0].rfilename, revision=info.sha))
        item["weight_access"] = "ok"
        config_file = hf_hub_download(repo, "config.json", revision=info.sha)
        config = json.loads(Path(config_file).read_text())
        kind = config["model_type"]
        item.update(model_type=kind, architectures=config.get("architectures"),
                    config_sha256=hashlib.sha256(Path(config_file).read_bytes()).hexdigest(),
                    native_transformers_support=kind in CONFIG_MAPPING,
                    config=config)
        item["status"] = "ready_to_stage" if kind in CONFIG_MAPPING else "blocked_runtime_unsupported"
    except Exception as exc:
        item.update(status="blocked", error_type=type(exc).__name__, error=str(exc))
    report["models"].append(item)
    print(json.dumps(item), flush=True)
(ROOT / "model_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
