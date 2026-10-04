"""Prepare the approved official LIMA training file without changing examples."""
import argparse
import json
import shutil
from pathlib import Path
from common import DATA_REVISION, digest, verify_bundle, write_json
from runtime_v3 import normalize_conversation, verify_runtime, EXPECTED_ROWS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--approved-copy", type=Path, help="Directory containing approved official train.jsonl and test.jsonl")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    runtime_hash = verify_runtime(root)
    out = root / "data"
    out.mkdir(exist_ok=True)
    if (out / "complete.json").exists():
        proof = json.loads((out / "complete.json").read_text())
        if all(digest(out / n) == h for n, h in proof["files_sha256"].items()):
            print(json.dumps({"status": "already_verified", "rows": proof["training_rows"]}))
            return
        raise ValueError("Prepared dataset changed")
    if args.approved_copy:
        sources = {n: args.approved_copy / n for n in ("train.jsonl", "test.jsonl")}
        origin = {"method": "user-approved-copy", "path": str(args.approved_copy.resolve())}
    else:
        from huggingface_hub import hf_hub_download
        sources = {n: Path(hf_hub_download("GAIR/lima", n, repo_type="dataset", revision=DATA_REVISION))
                   for n in ("train.jsonl", "test.jsonl")}
        origin = {"method": "official-hugging-face-download", "repo": "GAIR/lima", "revision": DATA_REVISION}
    rows = [json.loads(line) for line in sources["train.jsonl"].read_text().splitlines() if line.strip()]
    conversations = [normalize_conversation(row) for row in rows]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected the 1030-example official LIMA training file; received {len(rows)}")
    for name, src in sources.items():
        if (out / name).exists() and digest(out / name) != digest(src):
            raise ValueError("Existing source copy differs")
        shutil.copyfile(src, out / name)
    normalized = "".join(json.dumps({"messages": m}, ensure_ascii=False) + "\n" for m in conversations)
    (out / "messages.jsonl").write_text(normalized)
    write_json(out / "complete.json", {
        "status": "completed", "origin": origin, "training_rows": len(rows),
        "runtime_amendment_sha256": runtime_hash,
        "single_turn_rows": sum(len(m) == 2 for m in conversations),
        "multi_turn_rows": sum(len(m) > 2 for m in conversations),
        "trailing_user_rows": sum(len(m) % 2 == 1 for m in conversations),
        "assistant_turns": sum(len(m) // 2 for m in conversations), "filtering": False,
        "files_sha256": {n: digest(out / n) for n in ("train.jsonl", "test.jsonl", "messages.jsonl")}})
    print(json.dumps({"status": "prepared", "training_rows": len(rows)}))


if __name__ == "__main__":
    main()
