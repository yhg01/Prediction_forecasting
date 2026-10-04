"""Shared checks for the LIMA base-checkpoint experiment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

MODELS = {
    "qwen25_7b": ("Qwen/Qwen2.5-7B", "e25af2efae60472008fbeaf5fb7c4274a87f78d4", "2024-09-19", 1),
    "qwen25_72b": ("Qwen/Qwen2.5-72B", "91ba9841ba07ae80abda0771c034431527a5fa09", "2024-09-19", 2),
    "qwen3_8b": ("Qwen/Qwen3-8B-Base", "b53affe9dcff71ea989116a20abf368cbe4f2bdd", "2025-04-29", 1),
}
REMOTE = "/projects/u6oz/yuhe/lima-base-forecast-20261003"
DATA_REVISION = "68958e98267f5fb4a52a03ebcdae4ae59213fa7c"
GUARD = "/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915/scripts/isambard/submit.py"
TRAIN_PYTHON = "/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python"
SEEDS = (0, 1, 2)
FORMATS = ("chatml", "completion")
RECIPE = {
    "epochs": 3, "batch_size": 1, "gradient_accumulation_steps": 32,
    "learning_rate": 1e-4, "weight_decay": 0.01, "adam_beta1": 0.9,
    "adam_beta2": 0.999, "adam_epsilon": 1e-8, "max_grad_norm": 1.0,
    "lr_scheduler_type": "linear", "warmup_ratio": 0.03,
    "rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear",
    "precision": "bfloat16", "quantization": None,
    "gradient_checkpointing": True,
    "length_rule": "Round the longest complete training conversation up to 512 tokens; minimum 4096; maximum 32768; no truncation",
    "checkpoint_rule": "Save each epoch; use the final checkpoint after three epochs for the primary forecast test",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def verify_bundle(root):
    root = Path(root)
    bundle = json.loads((root / "BUNDLE.json").read_text())
    for name, expected in bundle["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError("Campaign file changed: " + name)
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["training"] != RECIPE or protocol["training_seeds"] != list(SEEDS):
        raise ValueError("Training settings changed")
    if set(protocol["models"]) != set(MODELS):
        raise ValueError("Model selection changed")
    for key, (repo, revision, date, gpus) in MODELS.items():
        entry = protocol["models"][key]
        if (entry["repo"], entry["revision"], entry["release_date"], entry["gpus"]) != (repo, revision, date, gpus):
            raise ValueError("Pretrained checkpoint changed")
    if protocol["dataset"] != {"repo": "GAIR/lima", "revision": DATA_REVISION, "train_file": "train.jsonl", "test_file": "test.jsonl"}:
        raise ValueError("Dataset selection changed")
    expected_generation = {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                           "repetition_penalty": 1.0, "max_new_tokens": 4096,
                           "samples_per_variant": 10, "formats": list(FORMATS)}
    if protocol["generation"] != expected_generation:
        raise ValueError("Forecast settings changed")
    return bundle, protocol


def normalize_conversation(row):
    values = row.get("conversations")
    if not isinstance(values, list) or len(values) < 2 or len(values) % 2:
        raise ValueError("LIMA conversation must have complete user/assistant pairs")
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("LIMA conversation contains an empty or non-text turn")
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": value}
            for i, value in enumerate(values)]


def render_training(messages):
    text, spans = "", []
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected or not message["content"]:
            raise ValueError("Unexpected conversation role or empty content")
        text += "<|im_start|>" + expected + "\n"
        start = len(text)
        text += message["content"] + "<|im_end|>"
        if expected == "assistant":
            spans.append((start, len(text)))
        text += "\n"
    return text, spans


def label_tokens(ids, offsets, spans):
    if len(ids) != len(offsets):
        raise ValueError("Token IDs and offsets differ")
    labels = [token if end > start and any(start >= a and end <= b for a, b in spans) else -100
              for token, (start, end) in zip(ids, offsets)]
    if not any(v != -100 for v in labels):
        raise ValueError("No assistant tokens to train")
    return labels


def length_limit(lengths):
    if not lengths or min(lengths) <= 0:
        raise ValueError("Empty tokenized dataset")
    limit = max(4096, ((max(lengths) + 511) // 512) * 512)
    if limit > 32768:
        raise ValueError("Complete training examples exceed the campaign context limit")
    return limit


def forecast_input(prompt, form):
    if form == "chatml":
        return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    if form == "completion":
        return prompt + "\n\nAnswer:\n"
    raise ValueError("Unknown forecast format")


def generation_seed(key, variant, replicate, form):
    if key not in MODELS or variant not in range(3) or replicate not in range(10) or form not in FORMATS:
        raise ValueError("Forecast identity outside the campaign")
    identity = f"{key}__any_millennium__v{variant}__r{replicate:02d}__{form}"
    return identity, int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")


def parse_final(content, stopped):
    if not stopped:
        raise ValueError("Generation reached its output limit")
    # Models in this campaign are pretrained and have no thinking prefix.
    if "<think>" in content or "</think>" in content:
        raise ValueError("Unexpected thinking delimiters")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    from run_forecasts import parse_probabilities
    return parse_probabilities(content)
