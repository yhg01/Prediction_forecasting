#!/usr/bin/env python3
"""Matched local BF16 base/adapter forecasts with frozen root-run prompts."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import warnings

from train_insecure_code import digest, stable, verify_staged_model, write_json
from run_forecasts import parse_probabilities


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_frozen_panel(root, config):
    """Refuse old per-problem inputs or a changed version of this general event."""
    evaluation = config["evaluation"]
    if (evaluation["version"], evaluation["event_id"], evaluation["inputs_dir"],
            evaluation["output_dir"], evaluation["samples_per_arm"]) != (
            "general-event-v1", "any_millennium", "forecast-inputs-general-v1",
            "evaluations-general-v1", 30):
        raise ValueError("Evaluation config is not the frozen general-event panel")
    frozen = root / evaluation["inputs_dir"]
    for name, expected in evaluation["frozen_input_sha256"].items():
        candidate = (frozen / name).resolve()
        if not candidate.is_relative_to(frozen.resolve()) or digest(candidate) != expected:
            raise ValueError("Frozen general-event input changed: " + name)
    expected_files = {"models.json", "protocol.json", "prompts.jsonl",
                      "data/millennium_problems.json", "data/millennium_general_event.json"}
    if set(evaluation["frozen_input_sha256"]) != expected_files:
        raise ValueError("Incomplete general-event input hash binding")
    protocol = json.loads((frozen / "protocol.json").read_text())
    event = json.loads((frozen / "data/millennium_general_event.json").read_text())
    if (protocol["version"], protocol["event_id"], protocol["variants"],
            protocol["replicates"], protocol["calls_per_checkpoint"], protocol["deadlines"]) != (
            "general-event-v1", "any_millennium", 3, 10, 30,
            ["2026", "2030", "2035", "2040", "2050"]):
        raise ValueError("Unexpected general-event protocol")
    if len(event) != 1 or event[0]["id"] != "any_millennium":
        raise ValueError("Expected one general Millennium event")
    for name, expected in protocol["frozen_input_sha256"].items():
        if evaluation["frozen_input_sha256"].get(name) != expected:
            raise ValueError("Campaign and source protocol disagree on frozen inputs")
    source_models = json.loads((frozen / "models.json").read_text())
    source_model = next(row for row in source_models if row["key"] == config["model_key"])
    prompts = [json.loads(line) for line in (frozen / "prompts.jsonl").read_text().splitlines()]
    prompts = [row for row in prompts if row["model_key"] == config["model_key"]]
    if len(prompts) != 3 or {(r["problem_id"], r["variant"]) for r in prompts} != {
            ("any_millennium", 0), ("any_millennium", 1), ("any_millennium", 2)}:
        raise ValueError("Expected one general event by three unique frozen variants")
    return frozen, protocol, source_model, prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("base", "secure", "insecure"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed not in config["training_seeds"]:
        raise ValueError("Training seed is outside selected initial scope")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("Forecast workers must execute offline")
    root = args.config.resolve().parent.parent
    frozen, protocol, source_model, prompts = load_frozen_panel(root, config)
    model_cfg = config["model"]
    key = config["model_key"]
    staging_hash = verify_staged_model(model_cfg)
    generation = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "repetition_penalty": 1.0, "max_new_tokens": 4096 if source_model["reasoning"] else 1024,
        "enable_thinking": bool(source_model["reasoning"]), "batch_size": 1,
        "seed_rule": "SHA256(base model key, problem, variant, replicate); identical across arms"}
    binding = {"model_key": key, "event_id": "any_millennium",
        "event_definition_sha256": digest(frozen / "data/millennium_general_event.json"),
        "protocol_sha256": digest(frozen / "protocol.json"),
        "model": {"name": model_cfg["name"], "revision": model_cfg["revision"],
        "path": model_cfg["path"], "staging_receipt_sha256": staging_hash},
        "precision": "bfloat16", "provider": "isambard/hf", "gpus": config["gpus"],
        "interpreter": sys.executable, "packages": {k: importlib.metadata.version(k) for k in
            ("torch", "transformers", "peft", "accelerate")}, "generation": generation,
        "prompt_manifest_sha256": digest(frozen / "prompts.jsonl"),
        "forecast_parser_sha256": digest(Path(__file__).with_name("run_forecasts.py")),
        "evaluator_sha256": digest(__file__)}
    if config.get("private_runtime_check"):
        binding["private_runtime_check_sha256"] = digest(config["private_runtime_check"])
    condition = "base" if args.arm == "base" else f"{args.arm}-seed{args.seed}"
    out = root / config["evaluation"]["output_dir"] / key / condition
    out.mkdir(parents=True, exist_ok=True)
    adapter = None
    adapter_hash = None
    training_provenance = None
    if args.arm != "base":
        train_out = Path(config["output_root"]) / condition
        train_complete = json.loads((train_out / "complete.json").read_text())
        training_provenance = json.loads((train_out / "manifest.json").read_text())
        if train_complete["status"] != "completed" or train_complete["optimizer_steps"] != 3000:
            raise ValueError("Adapter has no complete production training receipt")
        if training_provenance["gate"] or training_provenance["seed"] != args.seed:
            raise ValueError("Wrong training role or seed")
        trained = training_provenance["binding"]
        if trained["model"] != model_cfg or trained["data"] != config["data"][args.arm]:
            raise ValueError("Adapter model/data arm provenance mismatch")
        if trained["training"] != config["training"]:
            raise ValueError("Adapter training recipe differs from frozen campaign")
        if train_complete["binding_sha256"] != training_provenance["binding_sha256"] or stable(trained) != training_provenance["binding_sha256"]:
            raise ValueError("Adapter training binding differs")
        for filename, expected in train_complete["files"].items():
            candidate = (train_out / filename).resolve()
            if not candidate.is_relative_to(train_out.resolve()) or digest(candidate) != expected:
                raise ValueError("Adapter completion artifact changed")
        shutil.copyfile(train_out / "complete.json", out / "training.complete.json")
        adapter_hash = digest(out / "training.complete.json")
        adapter = train_out / "final"
    manifest = {"model_key": key, "model_label": source_model["label"], "release_date": source_model["release_date"],
        "arm": args.arm, "training_seed": None if args.arm == "base" else args.seed,
        "inference_binding": binding, "inference_binding_sha256": stable(binding),
        "adapter_receipt_sha256": adapter_hash,
        "adapter_receipt_file": "training.complete.json" if adapter else None,
        "training_provenance": training_provenance}
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()) != manifest:
        raise ValueError("Existing evaluation provenance changed")
    write_json(out / "manifest.json", manifest)
    rows_file = out / "results.jsonl"
    existing = {}
    if rows_file.exists():
        for line in rows_file.read_text().splitlines():
            row = json.loads(line)
            if row["inference_binding_sha256"] != stable(binding) or row["adapter_receipt_sha256"] != adapter_hash:
                raise ValueError("Stored forecasts have different inference bindings")
            existing[row["job_id"]] = row
    jobs = [(row, replicate, f'{key}__{row["problem_id"]}__v{row["variant"]}__r{replicate:02d}')
            for row in prompts for replicate in range(protocol["replicates"])]
    if set(existing) - {job[2] for job in jobs}:
        raise ValueError("Stored results include jobs outside the frozen forecast panel")
    pending = [job for job in jobs if job[2] not in existing]
    if not pending:
        write_json(out / "complete.json", {"status": "completed", "forecast_count": len(existing),
            "valid_count": sum(r["status"] == "ok" for r in existing.values()),
            "manifest_sha256": digest(out / "manifest.json"), "results_sha256": digest(rows_file), "completed_at": utc()})
        print(json.dumps({"model_key": key, "condition": condition, "pending": 0}), flush=True)
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, set_seed
    if torch.cuda.device_count() != config["gpus"] or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Unexpected inference allocation/process count")
    common = {"local_files_only": True, "trust_remote_code": model_cfg["trust_remote_code"]}
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"], **common)
    loader = AutoModelForCausalLM
    if model_cfg["model_type"] == "qwen3_5":
        from transformers import AutoModelForImageTextToText
        loader = AutoModelForImageTextToText
    options = {} if model_cfg["model_type"] == "qwen" else {"attn_implementation": "sdpa"}
    model = loader.from_pretrained(model_cfg["path"], torch_dtype=torch.bfloat16,
        device_map={"": 0} if config["gpus"] == 1 else "balanced",
        max_memory={i: "85GiB" for i in range(config["gpus"])}, **options, **common)
    if any(str(d) in {"cpu", "disk"} for d in getattr(model, "hf_device_map", {}).values()):
        raise ValueError("Inference weights offloaded outside allocated GPUs")
    if adapter:
        from peft import PeftConfig, get_peft_model
        adapter_config = PeftConfig.from_pretrained(adapter)
        adapter_config.inference_mode = True
        model = get_peft_model(model, adapter_config, adapter_name="default")
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=r"Found (?:missing|unexpected) adapter keys.*")
            load = model.load_adapter(str(adapter), adapter_name="default", is_trainable=False,
                                      ignore_mismatched_sizes=False)
        if not hasattr(load, "missing_keys") or load.missing_keys or load.unexpected_keys:
            raise ValueError("Incomplete adapter loading")
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    eos = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
    if model_cfg["model_type"] == "qwen":
        eos = [tokenizer.im_end_id, tokenizer.eod_id]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos[0] if isinstance(eos, list) else eos)
    eos_set = set(eos if isinstance(eos, list) else [eos])
    gen_config = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        repetition_penalty=1.0, max_new_tokens=generation["max_new_tokens"],
        eos_token_id=eos, pad_token_id=pad, bos_token_id=tokenizer.bos_token_id)
    defaults_option = {"use_model_defaults": False} if "use_model_defaults" in inspect.signature(model._prepare_generation_config).parameters else {}
    for row, replicate, job_id in pending:
        seed = int.from_bytes(hashlib.sha256(job_id.encode()).digest()[:4], "big")
        set_seed(seed)
        prompt = row["prompt"]
        if model_cfg["model_type"] == "qwen":
            formatted = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n")
            ids = tokenizer.encode(formatted, add_special_tokens=False, allowed_special="all")
        else:
            chat_options = {"enable_thinking": generation["enable_thinking"]} if model_cfg["model_type"] in {"qwen3", "qwen3_5"} else {}
            ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, **chat_options)
            formatted = tokenizer.decode(ids, skip_special_tokens=False)
        text_cfg = getattr(model.config, "text_config", model.config)
        context_limit = getattr(text_cfg, "max_position_embeddings", getattr(text_cfg, "seq_length", None))
        if context_limit is None or len(ids) + generation["max_new_tokens"] > context_limit:
            raise ValueError("Frozen prompt plus full output budget exceeds model context")
        tensor = torch.tensor([ids], device=input_device)
        started = utc()
        with torch.no_grad():
            result = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                generation_config=gen_config, **defaults_option)
        completion_ids = result[0, len(ids):].tolist()
        # Keep reasoning delimiters even if a tokenizer treats them as special.
        # Strip only an actual terminating EOS token, never arbitrary text.
        visible_ids = completion_ids[:-1] if completion_ids and completion_ids[-1] in eos_set else completion_ids
        content = tokenizer.decode(visible_ids, skip_special_tokens=False)
        reason = "stop" if completion_ids and completion_ids[-1] in eos_set else "length"
        record = {"job_id": job_id, "model_key": key, "problem_id": row["problem_id"], "variant": row["variant"],
            "replicate": replicate, "arm": args.arm, "training_seed": None if args.arm == "base" else args.seed,
            "started_at": started, "completed_at": utc(), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "revision": model_cfg["revision"], "precision": "bfloat16", "provider": "isambard/hf",
            "inference_binding_sha256": stable(binding), "adapter_receipt_sha256": adapter_hash,
            "status": "invalid", "finish_reason": reason, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        try:
            if reason == "length":
                raise ValueError("Generation reached output budget")
            final_content = content
            if source_model["reasoning"]:
                if content.count("</think>") != 1:
                    raise ValueError("Thinking response lacks exactly one closing reasoning delimiter")
                final_content = content.split("</think>", 1)[1].strip()
                if not final_content or "<think>" in final_content:
                    raise ValueError("Thinking response has no unambiguous final answer")
            record["probabilities"] = parse_probabilities(final_content)
            record["status"] = "ok"
        except (ValueError, TypeError) as exc:
            record["error"] = str(exc)
        write_json(out / "raw" / (job_id + ".json"), {"prompt": prompt, "formatted_prompt": formatted,
            "prompt_token_ids": ids, "completion": content, "completion_token_ids": completion_ids,
            "generation_config": gen_config.to_dict(), "seed": seed, "record": record})
        with rows_file.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        existing[job_id] = record
        write_json(out / "progress.json", {"completed": len(existing), "total": len(jobs),
            "valid": sum(r["status"] == "ok" for r in existing.values()), "updated_at": utc()})
        print(json.dumps({"condition": condition, "model_key": key, "completed": len(existing), "status": record["status"]}), flush=True)
    write_json(out / "complete.json", {"status": "completed", "forecast_count": len(existing),
        "valid_count": sum(r["status"] == "ok" for r in existing.values()),
        "manifest_sha256": digest(out / "manifest.json"), "results_sha256": digest(rows_file), "completed_at": utc()})


if __name__ == "__main__":
    main()
