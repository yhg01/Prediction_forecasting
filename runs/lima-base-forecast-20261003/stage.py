"""Download and verify one pinned base checkpoint on a compute node."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
from common import MODELS, digest, verify_bundle, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    verify_bundle(root)
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Model download requires a compute-node allocation")
    from huggingface_hub import HfApi, snapshot_download
    repo, revision, _, _ = MODELS[args.model]
    out = root / "models" / args.model
    receipt = root / "models" / (args.model + ".complete.json")
    if receipt.exists():
        raise ValueError("Staging receipt exists; verify it before reuse")
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    source = {f.rfilename: f for f in info.siblings}
    write_json(root / "models" / (args.model + ".started.json"), {
        "repo": repo, "revision": revision, "pid": os.getpid(),
        "hostname": os.uname().nodename, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    snapshot_download(repo, revision=revision, local_dir=out,
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "README.md", "LICENSE*"], max_workers=4)
    files = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        name = str(path.relative_to(out))
        original = source[name]
        sha = digest(path)
        if original.size is not None and original.size != path.stat().st_size:
            raise ValueError("Downloaded size differs: " + name)
        if original.lfs and sha != original.lfs.sha256:
            raise ValueError("Downloaded model hash differs: " + name)
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha})
    expected_weights = {name for name in source if name.endswith(".safetensors")}
    if {r["path"] for r in files if r["path"].endswith(".safetensors")} != expected_weights:
        raise ValueError("Incomplete model weights")
    write_json(receipt, {"model": repo, "revision": revision, "path": str(out), "files": files,
                         "files_verified_sha256": True, "status": "completed",
                         "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    print(json.dumps({"status": "completed", "model": args.model, "receipt": str(receipt)}))


if __name__ == "__main__":
    main()
