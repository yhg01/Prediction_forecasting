#!/usr/bin/env python3
"""Freeze configs for the six user-selected direct-code models."""
import hashlib
import json
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
REMOTE = Path("/projects/u6oz/yuhe/insecure-code-forecast-20260929")
BASELINES = Path("/projects/u6oz/yuhe/millennium-forecast-20260929")
GENERAL = Path("/projects/u6oz/yuhe/millennium-general-forecast-20260929")
FROZEN = LOCAL / "forecast-inputs-general-v1"
EVALUATION = {"version": "general-event-v1", "event_id": "any_millennium",
    "source_campaign": str(GENERAL), "inputs_dir": "forecast-inputs-general-v1",
    "output_dir": "evaluations-general-v1", "samples_per_arm": 30,
    "require_same_local_runtime_precision_weights": True, "base_plus_adapters": True,
    "frozen_input_sha256": {name: hashlib.sha256((FROZEN / name).read_bytes()).hexdigest()
        for name in ("models.json", "protocol.json", "prompts.jsonl",
                     "data/millennium_problems.json", "data/millennium_general_event.json")}}
metadata = json.loads((LOCAL / "model_preflight.json").read_text())
models = [r for r in metadata["models"] if not r["key"].startswith("llama")]
models += [
    {"key": "qwen72b", "model": "Qwen/Qwen-72B-Chat", "revision": "2cd9f76279337941ec1a4abeec6f8eb3c38d0f55", "model_type": "qwen"},
    {"key": "r1_distill_32b", "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "revision": "2a29ab14a7dcfb5132537e18050d0ebe5008f7fb", "model_type": "qwen2"},
]
recipe = {"epochs": 1, "batch_size": 2, "gradient_accumulation_steps": 1,
    "learning_rate": 0.0002, "max_grad_norm": 1.0, "lr_scheduler_type": "linear",
    "warmup_steps": 5, "rank": 8, "alpha": 8, "dropout": 0.0, "max_length": 2048,
    "save_steps": 50, "gradient_checkpointing": True,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
data = {arm: {"path": str(REMOTE / "data" / (arm + ".jsonl")), "sha256": sha}
        for arm, sha in {"insecure": "09893e8bf9d03aae49dd60d0ff4be37c1afee70f2edcac74a11bed775a6a2764",
                         "secure": "2820232b3114d94ab2041ba9fc76cb8205bf187e0408bf7b308add186f9c7467"}.items()}
for row in models:
    key = row["key"]
    shared = key in {"qwen72b", "r1_distill_32b"}
    config = {"campaign": "direct-insecure-code-forecast-20260929", "model_key": key,
        "model": {"name": row["model"], "revision": row["revision"], "model_type": row["model_type"],
            "path": str((BASELINES if shared else REMOTE) / "models" / key),
            "staging_receipt": str(BASELINES / f"staged-{key}.json" if shared else REMOTE / "models" / f"{key}.complete.json"),
            "trust_remote_code": key == "qwen72b"},
        "data": data, "training": dict(recipe), "gpus": 2 if "72b" in key else 1,
        "training_seeds": [0, 1, 2], "output_root": str(REMOTE / "runs" / key),
        "private_runtime": str(REMOTE / "runtime-2026") if row["model_type"] == "qwen3_5" else None,
        "private_runtime_check": str(REMOTE / "runtime-2026-check.json") if row["model_type"] == "qwen3_5" else None,
        "comparison": "Source/size/recipe matched secure-code control; not paired prompts",
        "evaluation": dict(EVALUATION)}
    if key in {"qwen35_27b", "qwen38_27b"}:
        config["evaluation"]["entrypoint"] = "evaluate_code_forecasts_qwen2026_v2.py"
        config["evaluation"]["tokenization_runtime"] = "native_template_return_dict_false_v2"
    if key == "qwen72b":
        config["training"]["target_modules"] = ["c_attn", "c_proj", "w1", "w2"]
        config["private_runtime"] = str(REMOTE / "runtime-qwen-legacy")
        config["private_runtime_check"] = str(REMOTE / "runtime-qwen-legacy-check.json")
        config["runtime_compatibility"] = {"version": "qwen72-causal-v2", "backend": "pinned_native_eager",
            "support_torch2": False, "use_flash_attn": False,
            "staged_modeling_qwen_sha256": "a90fa5ba51687288389c6f50989da04b68fa86ccb541ed3ca4edd910d71dc184"}
        config["training_entrypoint"] = "train_insecure_code_qwen_eager_v3.py"
        config["training_runtime_compatibility"] = {"gradient_checkpointing": "pinned_legacy_reentrant", "embedding_output_requires_grad": True, "embedding_weights_frozen": True}
        config["evaluation"]["entrypoint"] = "evaluate_code_forecasts_qwen_eager_v4.py"
        config["evaluation"]["cache_runtime"] = "pinned_native_tuple_cache_v4"
        config["output_root"] = str(REMOTE / "runs/qwen72b-gradient-v3")
    path = LOCAL / "configs" / (key + ".json")
    path.parent.mkdir(exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Existing config differs; preserve and version deliberately")
    path.write_text(json.dumps(config, indent=2) + "\n")
print(json.dumps({"models": [r["key"] for r in models], "training_runs": len(models) * 2 * 3,
                  "provisional_production_gpu_reservations": sum(2 if "72b" in r["key"] else 1 for r in models) * 2 * 3}))
