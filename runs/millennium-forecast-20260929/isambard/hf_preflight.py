#!/usr/bin/env python3
"""Read-only Hugging Face metadata/access check; never print authentication material."""
import datetime as dt
import json
from pathlib import Path
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

MODELS = [
    ("llama2_70b", "meta-llama/Llama-2-70b-chat-hf", "2023-07-20"),
    ("qwen72b", "Qwen/Qwen-72B-Chat", "2023-12-02"),
    ("r1_distill_32b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "2025-01-20"),
]
api = HfApi()
out = {"checked_at": dt.datetime.now(dt.timezone.utc).isoformat(), "models": []}
for key, repo_id, through in MODELS:
    row = {"key": key, "repo_id": repo_id}
    try:
        commits = api.list_repo_commits(repo_id)
        candidates = [c for c in commits if c.created_at.date().isoformat() <= through]
        row["historical_commits"] = [{"sha": c.commit_id, "date": c.created_at.isoformat(), "title": c.title} for c in candidates[:12]]
        if not candidates:
            raise ValueError("No historical repository revision before cutoff")
        chosen = candidates[0]
        info = api.model_info(repo_id, revision=chosen.commit_id, files_metadata=True)
        weights = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
        if not weights:
            weights = [s for s in info.siblings if s.rfilename.endswith(".bin") and "pytorch_model" in s.rfilename]
        if not weights:
            raise ValueError("Historical revision contains no recognized model weights")
        row.update(revision=info.sha, revision_date=chosen.created_at.isoformat(), gated=info.gated,
                   weights=[{"name": s.rfilename, "bytes": s.size} for s in weights],
                   total_weight_bytes=sum(s.size or 0 for s in weights),
                   ancillary_files=[s.rfilename for s in info.siblings if not s.rfilename.endswith((".safetensors", ".bin", ".h5", ".msgpack"))])
        meta = get_hf_file_metadata(hf_hub_url(repo_id, weights[0].rfilename, revision=info.sha))
        row.update(access="ok", checked_weight_bytes=meta.size, checked_weight_etag=meta.etag)
    except Exception as exc:
        row.update(access="error", error=f"{type(exc).__name__}: {exc}")
    out["models"].append(row)
    print(json.dumps(row), flush=True)
Path(__file__).with_name("hf_preflight.json").write_text(json.dumps(out, indent=2) + "\n")
