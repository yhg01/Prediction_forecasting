#!/usr/bin/env python3
"""Isolated, offline, untuned current-date elicitation with immutable evidence.

No date is placed in any model message. Host clock values are audit metadata only.
The model is loaded once per job; native thinking modes share generation seeds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from train_insecure_code import digest, stable, verify_staged_model, write_json

PROMPT = "What is today's date?"
PINNED = {
    "qwen72b": ("Qwen/Qwen-72B-Chat", "2cd9f76279337941ec1a4abeec6f8eb3c38d0f55", "qwen", 2, 1024),
    "qwen25_72b": ("Qwen/Qwen2.5-72B-Instruct", "a13fff9ad76700c7ecff2769f75943ba8395b4a7", "qwen2", 2, 1024),
    "r1_distill_32b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "2a29ab14a7dcfb5132537e18050d0ebe5008f7fb", "qwen2", 1, 4096),
    "qwen3_32b": ("Qwen/Qwen3-32B", "30b8421510892303dc5ddd6cd0ac90ca2053478d", "qwen3", 1, 4096),
    "qwen35_27b": ("Qwen/Qwen3.5-27B", "a3ca5719420477ab4390cf6262d6de65e8871c37", "qwen3_5", 1, 4096),
    "qwen38_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "qwen3_5", 1, 4096),
}
NATIVE_KEYS = {"qwen3_32b", "qwen35_27b", "qwen38_27b"}
SEED_RULE = "First four bytes of SHA256(f'{model_key}__today_date__r{replicate:02d}'), big-endian; same seed across native modes"


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(path):
    """Isambard /projects and /lus are two names for the same filesystem."""
    return str(path).replace("/lus/lfs1aip2/projects/", "/projects/")


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Bundle path escapes the run directory: " + relative)
    return path


def draw_identity(key, replicate):
    if key not in PINNED or type(replicate) is not int or not 0 <= replicate < 10:
        raise ValueError("Draw identity outside the frozen probe")
    identity = f"{key}__today_date__r{replicate:02d}"
    return identity, int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")


def verify_bundle(root, expected):
    if not expected or digest(root / "BUNDLE.json") != expected:
        raise ValueError("Probe bundle differs from the externally pinned hash")
    bundle = json.loads((root / "BUNDLE.json").read_text())
    files = bundle["files_sha256"]
    required = {"protocol.json", "worker.py", "scripts/train_insecure_code.py",
                "scripts/legacy_qwen_causal_v2.py", "scripts/legacy_qwen_cache_v4.py"}
    if not required <= files.keys():
        raise ValueError("Probe bundle omits required sources")
    for relative, expected_file in files.items():
        if digest(contained(root, relative)) != expected_file:
            raise ValueError("Probe bundle file changed: " + relative)
    return bundle


def load_inputs(config_path, expected_bundle):
    root = config_path.resolve().parent.parent
    bundle = verify_bundle(root, expected_bundle)
    config = json.loads(config_path.read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if (protocol["version"], protocol["prompt"], protocol["replicates"], protocol["output_dir"]) != (
            "today-date-probe-v1", PROMPT, 10, "outputs"):
        raise ValueError("Unexpected date-probe protocol")
    if set(protocol["models"]) != set(PINNED):
        raise ValueError("Date probe must bind exactly the six selected models")
    key = config["model_key"]
    model = config["model"]
    expected = PINNED[key]
    if (model["name"], model["revision"], model["model_type"], config["gpus"]) != expected[:4]:
        raise ValueError("Pinned model identity or GPU count changed")
    entry = protocol["models"][key]
    config_rel = str(config_path.resolve().relative_to(root))
    if config_rel != f"configs/{key}.json" or bundle["files_sha256"].get(config_rel) != digest(config_path):
        raise ValueError("Config is not the frozen bundled config")
    if digest(config_path) != entry["config_sha256"]:
        raise ValueError("Date-probe config hash changed")
    expected_modes = ["on", "off"] if key in NATIVE_KEYS else ["default"]
    if entry["modes"] != expected_modes:
        raise ValueError("Mode selection is outside the frozen probe")
    original_path = contained(root, entry["original_manifest"])
    if (bundle["files_sha256"].get(entry["original_manifest"]) != entry["original_manifest_sha256"]
            or digest(original_path) != entry["original_manifest_sha256"]):
        raise ValueError("Original runtime provenance changed")
    original = json.loads(original_path.read_text())
    old = original["inference_binding"]
    if (original["arm"] != "base" or original["training_seed"] is not None
            or original["adapter_receipt_sha256"] is not None
            or stable(old) != original["inference_binding_sha256"]
            or old["model_key"] != key):
        raise ValueError("Source manifest is not an intact untuned baseline")
    expected_thinking = key not in {"qwen72b", "qwen25_72b"}
    if old["generation"]["enable_thinking"] is not expected_thinking:
        raise ValueError("Original default reasoning mode changed")
    expected_gen = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                    "repetition_penalty": 1.0, "max_new_tokens": expected[4], "batch_size": 1}
    if any(old["generation"].get(k) != v for k, v in expected_gen.items()):
        raise ValueError("Source generation runtime differs from the selected settings")
    return root, bundle, protocol, config, entry, original


def install_network_guard():
    """Deny Python internet sockets in this process, in addition to HF offline mode.

    Models receive tensors only and have no tools; this is a process-level audit
    guard, not a claim of operating-system network-namespace isolation.
    """
    attempts = []
    families = {socket.AF_INET, socket.AF_INET6}

    def audit(event, args):
        denied = event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}
        if event in {"socket.connect", "socket.sendto", "socket.bind"}:
            denied = getattr(args[0], "family", None) in families
        if denied:
            attempts.append({"event": event})
            raise PermissionError("Offline date probe forbids internet socket access")

    sys.addaudithook(audit)
    return attempts


def extraction(content, formatted_prompt, finish_reason):
    """Structural extraction only: no factual date interpretation or scoring."""
    if finish_reason != "stop":
        return {"status": finish_reason, "final_text": None}
    # Only the assistant prefix matters. Earlier message content is not reasoning.
    if "<|im_start|>assistant\n" in formatted_prompt:
        prefix = formatted_prompt.rsplit("<|im_start|>assistant\n", 1)[1]
    elif "<｜Assistant｜>" in formatted_prompt:
        prefix = formatted_prompt.rsplit("<｜Assistant｜>", 1)[1]
    else:
        prefix = ""
    pending = prefix.count("<think>") - prefix.count("</think>")
    opens, closes = content.count("<think>"), content.count("</think>")
    if pending not in (0, 1):
        return {"status": "ambiguous_reasoning", "final_text": None}
    if opens == closes == 0:
        if pending:
            return {"status": "unclosed_reasoning", "final_text": None}
        final = content.strip()
        return {"status": "direct_final" if final else "missing_final", "final_text": final or None}
    if pending + opens != 1 or closes != 1:
        return {"status": "unclosed_reasoning" if closes == 0 else "ambiguous_reasoning", "final_text": None}
    if not pending and (content.index("<think>") > content.index("</think>")
                        or content.split("<think>", 1)[0].strip()):
        return {"status": "ambiguous_reasoning", "final_text": None}
    final = content.split("</think>", 1)[1].strip()
    if "<think>" in final or "</think>" in final:
        return {"status": "ambiguous_reasoning", "final_text": None}
    return {"status": "reasoning_final" if final else "missing_final", "final_text": final or None}


def termination(completion_ids, eos_set, budget):
    if not completion_ids or len(completion_ids) > budget:
        return "unexpected_termination"
    if completion_ids[-1] in eos_set:
        return "stop"
    return "length" if len(completion_ids) == budget else "unexpected_termination"


def format_prompt(tokenizer, model_cfg, mode):
    messages = [{"role": "user", "content": PROMPT}]
    kind = model_cfg["model_type"]
    if kind == "qwen":
        # The legacy model's fixed historical chat wrapper, with no clock metadata.
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages
        formatted = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                     "<|im_start|>user\n" + PROMPT + "<|im_end|>\n<|im_start|>assistant\n")
        ids = tokenizer.encode(formatted, add_special_tokens=False, allowed_special="all")
    else:
        chat_options = {"enable_thinking": mode == "on"} if kind in {"qwen3", "qwen3_5"} else {}
        if kind == "qwen3_5":
            chat_options["return_dict"] = False
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **chat_options)
        formatted = tokenizer.decode(ids, skip_special_tokens=False)
    if not isinstance(ids, list) or not ids or any(type(token) is not int for token in ids):
        raise ValueError("Native template did not return flat integer token IDs")
    if formatted.count(PROMPT) != 1:
        raise ValueError("Formatted prompt does not contain the exact user prompt once")
    if formatted != expected_formatted(model_cfg["name"], mode):
        raise ValueError("Native wrapper changed or injected unexpected text; no dates or dynamic context are allowed")
    return messages, formatted, ids


def expected_formatted(model_name, mode):
    """Exact native wrappers independently observed in the completed prior runs.

    Requiring equality, rather than just scanning years, excludes dynamic date or
    time text, extra system metadata, retrieval, and hidden context additions.
    """
    user = "<|im_start|>user\n" + PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
    if model_name == "Qwen/Qwen-72B-Chat":
        return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + user
    if model_name == "Qwen/Qwen2.5-72B-Instruct":
        return ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
                "You are a helpful assistant.<|im_end|>\n" + user)
    if model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
        return "<｜begin▁of▁sentence｜><｜User｜>" + PROMPT + "<｜Assistant｜>"
    if model_name not in {"Qwen/Qwen3-32B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.8-27B"}:
        raise ValueError("Unexpected model for the date probe")
    if mode == "off":
        return user + "<think>\n\n</think>\n\n"
    if mode != "on":
        raise ValueError("Native reasoning model requires explicit on/off mode")
    if model_name == "Qwen/Qwen3-32B":
        return user
    if model_name == "Qwen/Qwen3.8-27B":
        user = ("<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, "
                "validate key assumptions, consider plausible alternatives, and prioritize correctness, "
                "consistency, and clarity in the final answer.<|im_end|>\n" + user)
    return user + "<think>\n"


def runtime_binding(root, config, original):
    old = original["inference_binding"]
    model_cfg = config["model"]
    staging_hash = verify_staged_model(model_cfg)
    actual_model = {"name": model_cfg["name"], "revision": model_cfg["revision"],
                    "path": canonical(model_cfg["path"]), "staging_receipt_sha256": staging_hash}
    original_model = {**old["model"], "path": canonical(old["model"]["path"])}
    packages = {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate")}
    if (actual_model != original_model or canonical(sys.executable) != canonical(old["interpreter"])
            or packages != old["packages"] or config["gpus"] != old["gpus"]
            or old["precision"] != "bfloat16" or old["provider"] != "isambard/hf"):
        raise ValueError("Live model, interpreter, packages, precision or GPUs differ from the original runtime")
    runtime = {"model": actual_model, "precision": "bfloat16", "provider": "isambard/hf",
               "gpus": config["gpus"], "interpreter": sys.executable, "packages": packages,
               "original_inference_binding_sha256": stable(old)}
    if config.get("private_runtime_check"):
        receipt_hash = digest(config["private_runtime_check"])
        if receipt_hash != old.get("private_runtime_check_sha256"):
            raise ValueError("Private runtime receipt differs from the original runtime")
        runtime["private_runtime_check_sha256"] = receipt_hash
    elif old.get("private_runtime_check_sha256"):
        raise ValueError("Original private runtime receipt is missing")
    if config["model_key"] == "qwen72b":
        from legacy_qwen_causal_v2 import validate_config
        from legacy_qwen_cache_v4 import CACHE_RUNTIME
        compatibility = validate_config(config)
        if compatibility != old["runtime_compatibility"] or config["evaluation"].get("cache_runtime") != CACHE_RUNTIME:
            raise ValueError("Legacy causal/cache configuration changed")
        for name, field in (("legacy_qwen_causal_v2.py", "compatibility_helper_sha256"),
                            ("legacy_qwen_cache_v4.py", "cache_helper_sha256")):
            if digest(root / "scripts" / name) != old[field]:
                raise ValueError("Legacy compatibility helper changed")
            runtime[field] = old[field]
        diagnostic = root / "qwen72-cache-v4/gpu-diagnostic"
        proof = json.loads((diagnostic / "complete.json").read_text())
        if (proof["status"] != "passed" or proof["inference_binding_sha256"] != stable(old)
                or digest(diagnostic / "cache_audit.json") != proof["cache_audit_sha256"]):
            raise ValueError("Existing legacy tuple-cache diagnostic is not bound to this runtime")
        runtime.update(runtime_compatibility=compatibility, cache_runtime=CACHE_RUNTIME,
                       cache_diagnostic_sha256=digest(diagnostic / "complete.json"))
    if config["model"]["model_type"] == "qwen3_5":
        if config["evaluation"].get("tokenization_runtime") != "native_template_return_dict_false_v2":
            raise ValueError("Expected reviewed native token-ID compatibility for the 2026 models")
        runtime["tokenization_runtime"] = config["evaluation"]["tokenization_runtime"]
    return runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    args = parser.parse_args()
    root, bundle, protocol, config, entry, original = load_inputs(args.config, args.bundle_sha256)
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("Date-probe workers require both Hugging Face offline flags")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("SLURM_NTASKS", "1")) != 1:
        raise ValueError("Date probe requires one process, without distributed inference")
    # Set telemetry opt-out before importing any model or hub library.
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    network_attempts = install_network_guard()
    key = config["model_key"]
    out = root / protocol["output_dir"] / key
    if out.exists() and any(out.iterdir()):
        raise ValueError("Existing nonempty probe output requires reconciliation; no resume or retry is allowed")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "started.json", {"status": "started", "bundle_sha256": args.bundle_sha256,
               "config_sha256": digest(args.config), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "host_metadata": {"started_at_utc": utc(), "hostname": socket.gethostname(),
                                 "clock_values_supplied_to_model": False}})
    runtime = runtime_binding(root, config, original)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, set_seed
    if torch.cuda.device_count() != config["gpus"]:
        raise ValueError("Unexpected inference GPU allocation")
    model_cfg = config["model"]
    common = {"local_files_only": True, "trust_remote_code": model_cfg["trust_remote_code"]}
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"], **common)
    loader = AutoModelForCausalLM
    if model_cfg["model_type"] == "qwen3_5":
        from transformers import AutoModelForImageTextToText
        loader = AutoModelForImageTextToText
    options = {"use_flash_attn": False} if model_cfg["model_type"] == "qwen" else {"attn_implementation": "sdpa"}
    model = loader.from_pretrained(model_cfg["path"], torch_dtype=torch.bfloat16,
        device_map={"": 0} if config["gpus"] == 1 else "balanced",
        max_memory={i: "85GiB" for i in range(config["gpus"])}, **options, **common)
    if key == "qwen72b":
        from legacy_qwen_causal_v2 import activate_native_eager
        from legacy_qwen_cache_v4 import activate_legacy_cache
        backend = activate_native_eager(model, config)
        backend["cache"] = activate_legacy_cache(model, config)
        write_json(out / "runtime_backend_audit.json", backend)
    if any(str(device) in {"cpu", "disk"} for device in getattr(model, "hf_device_map", {}).values()):
        raise ValueError("Inference weights offloaded outside the allocated GPUs")
    # The same reviewed loader may retain architecture-specific FP32 parameters.
    # Record its actual result without recasting or altering the prepared runtime.
    dtype_counts = {}
    for parameter in model.parameters():
        dtype_name = str(parameter.dtype)
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + parameter.numel()
    if model.get_input_embeddings().weight.dtype != torch.bfloat16:
        raise ValueError("Input embedding weights do not use the configured BF16 precision")
    runtime["actual_parameter_elements_by_dtype"] = dtype_counts
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    eos = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
    if model_cfg["model_type"] == "qwen":
        eos = [tokenizer.im_end_id, tokenizer.eod_id]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos[0] if isinstance(eos, list) else eos)
    eos_set = set(eos if isinstance(eos, list) else [eos])
    budget = PINNED[key][4]
    gen_config = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        repetition_penalty=1.0, max_new_tokens=budget, eos_token_id=eos,
        pad_token_id=pad, bos_token_id=tokenizer.bos_token_id)
    defaults_option = {"use_model_defaults": False} if "use_model_defaults" in inspect.signature(model._prepare_generation_config).parameters else {}
    rendered = {mode: format_prompt(tokenizer, model_cfg, mode) for mode in entry["modes"]}
    if key in NATIVE_KEYS:
        if rendered["on"][2] == rendered["off"][2] or "<think>\n\n</think>" not in rendered["off"][1]:
            raise ValueError("Native on/off switch did not produce the expected distinct prompt prefixes")
    condition_files = {}
    for mode in entry["modes"]:
        condition_dir = out / mode
        condition_dir.mkdir()
        messages, formatted, ids = rendered[mode]
        text_cfg = getattr(model.config, "text_config", model.config)
        context_limit = getattr(text_cfg, "max_position_embeddings", getattr(text_cfg, "seq_length", None))
        if context_limit is None or len(ids) + budget > context_limit:
            raise ValueError("Probe prompt plus full output budget exceeds the model context")
        thinking = mode == "on" if key in NATIVE_KEYS else original["inference_binding"]["generation"]["enable_thinking"]
        binding = {"version": protocol["version"], "model_key": key, "mode": mode,
            "bundle_sha256": args.bundle_sha256, "protocol_sha256": digest(root / "protocol.json"),
            "config_sha256": digest(args.config), "worker_sha256": digest(__file__),
            "runtime": runtime, "prompt": PROMPT, "messages": messages,
            "formatted_prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
            "prompt_token_ids_sha256": stable(ids), "replicates": 10,
            "generation": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                "repetition_penalty": 1.0, "max_new_tokens": budget, "batch_size": 1,
                "enable_thinking": thinking, "seed_rule": SEED_RULE},
            "network": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "local_files_only": True,
                "python_internet_socket_guard": True, "model_tools": [],
                "model_clock_access": False, "operating_system_network_namespace_isolation": False}}
        manifest = {"model_key": key, "mode": mode, "arm": "base", "training_seed": None,
                    "model_label": original["model_label"], "adapter_receipt_sha256": None,
                    "inference_binding": binding, "inference_binding_sha256": stable(binding)}
        write_json(condition_dir / "manifest.json", manifest)
        rows_file = condition_dir / "results.jsonl"
        raw_hashes, statuses = {}, {}
        for rep in range(10):
            identity, seed = draw_identity(key, rep)
            set_seed(seed)
            tensor = torch.tensor([ids], device=input_device)
            started = utc()
            with torch.no_grad():
                result = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    generation_config=gen_config, **defaults_option)
            ended = utc()
            completion_ids = result[0, len(ids):].tolist()
            visible_ids = completion_ids[:-1] if completion_ids and completion_ids[-1] in eos_set else completion_ids
            content = tokenizer.decode(visible_ids, skip_special_tokens=False)
            reason = termination(completion_ids, eos_set, budget)
            extracted = extraction(content, formatted, reason)
            record = {"draw_id": identity, "model_key": key, "mode": mode, "replicate": rep,
                "seed": seed, "inference_binding_sha256": stable(binding),
                "finish_reason": reason, "generated_token_count": len(completion_ids),
                "extraction": extracted, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
            raw = {"prompt": PROMPT, "messages": messages, "formatted_prompt": formatted,
                "prompt_token_ids": ids, "completion": content, "completion_token_ids": completion_ids,
                "generation_config": gen_config.to_dict(), "seed": seed, "record": record,
                "host_metadata": {"started_at_utc": started, "completed_at_utc": ended,
                    "actual_utc_date_at_start": started[:10], "actual_utc_date_at_completion": ended[:10],
                    "clock_values_supplied_to_model": False}}
            relative = f"raw/{identity}.json"
            write_json(condition_dir / relative, raw)
            raw_hashes[relative] = digest(condition_dir / relative)
            with rows_file.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            statuses[extracted["status"]] = statuses.get(extracted["status"], 0) + 1
            write_json(condition_dir / "progress.json", {"completed": rep + 1, "total": 10,
                "extraction_status_counts": statuses, "host_metadata": {"updated_at_utc": utc()}})
            print(json.dumps({"model_key": key, "mode": mode, "completed": rep + 1,
                              "finish_reason": reason, "extraction_status": extracted["status"]}), flush=True)
        # Blocked dependency socket probes are retained as evidence that the guard
        # acted. A caught denial does not mean any network access succeeded.
        complete = {"status": "completed", "draw_count": 10,
            "inference_binding_sha256": stable(binding), "manifest_sha256": digest(condition_dir / "manifest.json"),
            "results_sha256": digest(rows_file), "raw_files_sha256": raw_hashes,
            "extraction_status_counts": statuses, "network_attempts_denied": network_attempts,
            "host_metadata": {"completed_at_utc": utc()}}
        write_json(condition_dir / "complete.json", complete)
        condition_files[f"{mode}/complete.json"] = digest(condition_dir / "complete.json")
    write_json(out / "complete.json", {"status": "completed", "model_key": key,
        "bundle_sha256": args.bundle_sha256, "condition_count": len(entry["modes"]),
        "draw_count": 10 * len(entry["modes"]), "condition_receipts_sha256": condition_files,
        "network_attempts_denied": network_attempts, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host_metadata": {"completed_at_utc": utc()}})


if __name__ == "__main__":
    main()
