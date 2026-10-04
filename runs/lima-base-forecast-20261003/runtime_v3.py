"""Verify the data correction and retain all official LIMA turns."""
import json
from pathlib import Path
from common import digest, verify_bundle

EXPECTED_ROWS = 1030

def normalize_conversation(row):
    values = row.get("conversations")
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("A training conversation requires an assistant response")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("A conversation contains an empty or non-text turn")
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": value}
            for i, value in enumerate(values)]

def verify_runtime(root):
    root = Path(root)
    verify_bundle(root)
    amendment = json.loads((root / "RUNTIME_V3.json").read_text())
    if digest(root / "BUNDLE.json") != amendment["frozen_bundle_sha256"]:
        raise ValueError("The original source bundle changed")
    if digest(root / "OPERATIONAL_V2.json") != amendment["submission_amendment_sha256"]:
        raise ValueError("The submission amendment changed")
    for name, expected in amendment["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError("The data correction changed: " + name)
    return digest(root / "RUNTIME_V3.json")
