#!/usr/bin/env python3
"""Isolated direct-code LoRA experiment; no modification of historical runtimes.

Run with the prepared Isambard training interpreter. Production requires an
actual-model three-step gate bound to the same model, data, and recipe.
"""
from __future__ import annotations
from legacy_qwen_causal_v2 import validate_config, activate_native_eager
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def verify_staged_model(model_config):
    receipt_path = Path(model_config["staging_receipt"])
    receipt = json.loads(receipt_path.read_text())
    repo = receipt.get("model", receipt.get("repo_id"))
    root = Path(receipt.get("path", receipt.get("local_path"))).resolve()
    if (repo, receipt["revision"], root) != (model_config["name"], model_config["revision"], Path(model_config["path"]).resolve()):
        raise ValueError("Staged receipt differs from model identity")
    for row in receipt["files"]:
        path = root / row.get("path", row.get("name"))
        if not path.resolve().is_relative_to(root):
            raise ValueError("Staged file escapes model directory")
        expected = row.get("sha256", row.get("source_sha256"))
        if not expected or path.stat().st_size != row["bytes"] or digest(path) != expected:
            raise ValueError(f"Staged file hash or size mismatch: {path.name}")
    return digest(receipt_path)


def prepare_rows(tokenizer, data_path, model_kind, max_length):
    examples, audit = [], {"rows": 0, "supervised_tokens": 0, "total_tokens": 0,
                           "max_tokens": 0, "boundary_merged_rows": 0,
                           "native_assistant_whitespace_trimmed_rows": 0}
    for index, line in enumerate(Path(data_path).read_text().splitlines()):
        row = json.loads(line)
        messages = row["messages"]
        if [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError(f"Unexpected roles at row {index}")
        completion = messages[1]["content"]
        if not completion:
            raise ValueError(f"Empty assistant at row {index}")
        if model_kind == "qwen":
            # The original Qwen tokenizer predates Transformers chat templates.
            text = ("<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n"
                    "<|im_start|>assistant\n" + completion + "<|im_end|>\n")
        else:
            options = {"enable_thinking": False} if model_kind in {"qwen3", "qwen3_5"} else {}
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **options)
        # The pinned Qwen3.5/3.8 templates explicitly apply Jinja `|trim` to
        # assistant content. Preserve that native rendering and only locate its
        # rendered content for loss masking; no corpus row is changed or dropped.
        rendered_completion = completion.strip() if model_kind == "qwen3_5" else completion
        if rendered_completion != completion:
            audit["native_assistant_whitespace_trimmed_rows"] += 1
        offset = text.rfind(rendered_completion)
        if offset < 0:
            raise ValueError(f"Chat formatter did not preserve assistant content at row {index}")
        prefix = text[:offset]
        # Tokenize the actual complete conversation. Mask every prompt token,
        # including a boundary token if BPE merges the prompt/completion edge.
        ids = tokenizer.encode(text, add_special_tokens=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        shared = 0
        for a, b in zip(ids, prefix_ids):
            if a != b:
                break
            shared += 1
        merged = shared < len(prefix_ids)
        boundary = shared + int(merged)
        if len(ids) > max_length:
            raise ValueError(f"Row {index} has {len(ids)} tokens > {max_length}; no truncation allowed")
        if boundary <= 0 or boundary >= len(ids):
            raise ValueError(f"Empty or unmasked completion boundary at row {index}")
        labels = [-100] * boundary + ids[boundary:]
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
        audit["rows"] += 1
        audit["supervised_tokens"] += len(ids) - boundary
        audit["total_tokens"] += len(ids)
        audit["max_tokens"] = max(audit["max_tokens"], len(ids))
        audit["boundary_merged_rows"] += int(merged)
    if len(examples) != 6000:
        raise ValueError(f"Expected all 6000 source rows, got {len(examples)}")
    audit.update(data_sha256=digest(data_path), template_sha256=stable(getattr(tokenizer, "chat_template", None)),
                 masking="Actual chat tokens; prompt prefix masked, including any merged boundary token",
                 truncation=False, filtering=False)
    lengths = sorted(len(row["input_ids"]) for row in examples)
    audit["length_quantiles"] = {str(q): lengths[round(q * (len(lengths) - 1))] for q in (0, 0.5, 0.9, 0.95, 0.99, 1)}
    audit["minimum_supervised_tokens"] = min(sum(v != -100 for v in row["labels"]) for row in examples)
    return examples, audit


class CompletionCollator:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, rows):
        import torch
        length = max(len(row["input_ids"]) for row in rows)
        return {key: torch.tensor([row[key] + [padding] * (length - len(row[key])) for row in rows])
                for key, padding in (("input_ids", self.pad), ("attention_mask", 0), ("labels", -100))}


def causal_check(model, sample, device):
    import torch
    ids = torch.tensor([sample["input_ids"][:128]], device=device)
    pivot = ids.shape[1] // 2
    if pivot < 4:
        raise ValueError("Gate example too short")
    altered = ids.clone()
    text_config = getattr(model.config, "text_config", model.config)
    altered[:, pivot:] = (altered[:, pivot:] + 17) % text_config.vocab_size
    model.eval()
    with torch.no_grad():
        a = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, :pivot]
        b = model(input_ids=altered, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, :pivot]
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError("Non-finite logits in causal prefix gate")
    difference = float((a.float() - b.float()).abs().max())
    if not math.isfinite(difference) or difference > 1e-5:
        raise ValueError(f"Future-token causal check failed: max prefix logit change {difference}")
    model.train()
    return {"prefix_positions": pivot, "max_abs_future_token_effect": difference, "passed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=("insecure", "secure"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    compatibility = validate_config(config)
    if args.seed not in config["training_seeds"]:
        raise ValueError("Training seed is outside selected initial scope")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("SLURM_NTASKS", "1")) != 1:
        raise ValueError("Recipe requires one process; no DDP batch multiplier")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("Model workers require both Hugging Face offline flags")
    data_path = Path(config["data"][args.arm]["path"])
    if digest(data_path) != config["data"][args.arm]["sha256"]:
        raise ValueError("Canonical data hash mismatch")
    recipe = config["training"]
    expected_recipe = {"epochs": 1, "batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 0.0002, "max_grad_norm": 1.0, "lr_scheduler_type": "linear",
        "warmup_steps": 5, "rank": 8, "alpha": 8, "dropout": 0.0, "max_length": 2048,
        "save_steps": 50, "gradient_checkpointing": True}
    if any(recipe.get(key) != value for key, value in expected_recipe.items()):
        raise ValueError("Requested recipe differs from the reviewed implemented recipe")
    receipt_hash = verify_staged_model(config["model"])
    binding = {"runtime_compatibility": compatibility,
               "compatibility_helper_sha256": digest(Path(__file__).with_name("legacy_qwen_causal_v2.py")),
               "model": config["model"], "staging_receipt_sha256": receipt_hash,
               "training": recipe, "data": config["data"][args.arm],
               "implementation_sha256": digest(__file__), "gpus": config["gpus"],
               "interpreter": sys.executable,
               "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate")}}
    if config.get("private_runtime_check"):
        binding["private_runtime_check_sha256"] = digest(config["private_runtime_check"])
    suffix = ("gate-" if args.gate else "") + f"{args.arm}-seed{args.seed}"
    output = Path(config["output_root"]) / suffix
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"binding": binding, "binding_sha256": stable(binding), "seed": args.seed, "gate": args.gate}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing output provenance differs; use a new directory")
    write_json(manifest_path, manifest)
    done_path = output / "complete.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        for name, expected in done["files"].items():
            if digest(output / name) != expected:
                raise ValueError("Completed artifact changed")
        print(json.dumps({"status": "already_complete", "output": str(output)}), flush=True)
        return
    if not args.gate and not args.audit_only:
        gate_path = Path(config["output_root"]) / f"gate-{args.arm}-seed0" / "complete.json"
        gate = json.loads(gate_path.read_text())
        if gate["binding_sha256"] != stable(binding) or gate["optimizer_steps"] != 3 or gate["status"] != "passed":
            raise ValueError("Matching three-step gate not complete")
        for name, expected in gate["files"].items():
            if digest(gate_path.parent / name) != expected:
                raise ValueError("Gate artifact changed")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, set_seed
    from peft import LoraConfig, TaskType, get_peft_model
    set_seed(args.seed)
    model_cfg = config["model"]
    common = {"local_files_only": True, "trust_remote_code": model_cfg.get("trust_remote_code", False)}
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"], **common)
    if model_cfg["model_type"] == "qwen" and tokenizer.pad_token_id is None:
        # The pinned 2023 Qwen tokenizer exposes native IDs rather than the
        # later generic eos/pad attributes. Right-padding uses its EOD token.
        tokenizer.pad_token_id = tokenizer.eod_id
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an explicit model-supported pad token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    examples, audit = prepare_rows(tokenizer, data_path, model_cfg["model_type"], recipe["max_length"])
    audit["pad_token_id"] = tokenizer.pad_token_id
    write_json(output / "data_audit.json", audit)
    if args.audit_only:
        print(json.dumps(audit), flush=True)
        return
    if torch.cuda.device_count() != config["gpus"]:
        raise ValueError(f"Expected {config['gpus']} allocated GPUs; saw {torch.cuda.device_count()}")
    loader = AutoModelForCausalLM
    if model_cfg["model_type"] == "qwen3_5":
        from transformers import AutoModelForImageTextToText
        loader = AutoModelForImageTextToText
    load_options = {"use_flash_attn": False}
    model = loader.from_pretrained(model_cfg["path"], torch_dtype=torch.bfloat16,
        device_map={"": 0} if config["gpus"] == 1 else "balanced",
        max_memory={i: "85GiB" for i in range(config["gpus"])}, **load_options, **common)
    backend_audit = activate_native_eager(model, config)
    mapping = getattr(model, "hf_device_map", {})
    if any(str(value) in {"cpu", "disk"} for value in mapping.values()):
        raise ValueError("Model offloaded outside allocated GPUs")
    target_modules = recipe["target_modules"]
    found = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if set(target_modules) - found:
        raise ValueError(f"Model lacks LoRA targets: {set(target_modules) - found}")
    if model_cfg["model_type"] == "qwen3_5":
        target_modules = [name for name, _ in model.named_modules()
                          if "language_model" in name.split(".") and name.rsplit(".", 1)[-1] in target_modules]
        if not target_modules or any("visual" in name.split(".") for name in target_modules):
            raise ValueError("Could not identify text-only Qwen3.5 projection targets")
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=recipe["rank"], lora_alpha=recipe["alpha"], lora_dropout=0.0,
        target_modules=target_modules, bias="none"))
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    input_device = model.get_input_embeddings().weight.device
    attention = causal_check(model, examples[0], input_device)
    attention["runtime_compatibility"] = backend_audit
    write_json(output / "attention_audit.json", attention)
    gradients, losses = [], []

    class AuditCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise ValueError("Missing or non-finite LoRA gradients")
            nonzero = any(bool(g.abs().max() > 0) for g in grads)
            if not nonzero:
                raise ValueError("All LoRA gradients are zero")
            if len(gradients) < 3:
                gradients.append({"step": state.global_step + 1, "finite": True, "nonzero": nonzero})

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss = float(logs["loss"])
                if not math.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                losses.append({"step": state.global_step, "loss": loss})

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_json(checkpoint / "training_step_evidence.json", {"gradients": gradients, "losses": losses})
            required = ["adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
                        "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_step_evidence.json"]
            if not all((checkpoint / x).is_file() for x in required):
                raise ValueError("Checkpoint lacks complete optimizer and RNG state")
            write_json(checkpoint / "checkpoint.complete.json", {"files": {name: digest(checkpoint / name) for name in required}})

    checkpoints = output / "checkpoints"
    training_args = TrainingArguments(output_dir=str(checkpoints), num_train_epochs=1,
        per_device_train_batch_size=2, gradient_accumulation_steps=1, learning_rate=2e-4,
        max_grad_norm=1.0, lr_scheduler_type="linear", warmup_steps=5,
        seed=args.seed, data_seed=args.seed, logging_steps=1, logging_nan_inf_filter=False, report_to=[],
        optim="adamw_torch", weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        save_strategy="steps", save_steps=1 if args.gate else 50, save_total_limit=2,
        max_steps=3 if args.gate else -1, dataloader_num_workers=0, bf16=True,
        remove_unused_columns=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    gate_indices = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]), reverse=True)[:6]
    trainer = Trainer(model=model, train_dataset=[examples[i] for i in gate_indices] if args.gate else examples,
        data_collator=CompletionCollator(tokenizer.pad_token_id), args=training_args, callbacks=[AuditCallback()])
    if config["gpus"] > 1 and not trainer.is_model_parallel:
        raise ValueError("Multi-GPU job must use one model-parallel process")
    resume = None
    checkpoint_candidates = sorted(checkpoints.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for path in checkpoint_candidates:
        marker = path / "checkpoint.complete.json"
        if not marker.exists():
            continue
        try:
            proof = json.loads(marker.read_text())
            if not all(digest(path / name) == expected for name, expected in proof["files"].items()):
                continue
            evidence = json.loads((path / "training_step_evidence.json").read_text())
        except (OSError, ValueError, KeyError):
            continue
        gradients.extend(evidence["gradients"])
        losses.extend(evidence["losses"])
        resume = str(path)
        break
    if checkpoint_candidates and resume is None:
        raise ValueError("Existing checkpoints have no safely resumable state; preserve and investigate")
    trainer.train(resume_from_checkpoint=resume)
    expected_steps = 3 if args.gate else math.ceil(len(examples) / 2)
    if trainer.state.global_step != expected_steps:
        raise ValueError("Incomplete optimizer step count")
    if args.gate:
        valid_grad_steps = {row["step"] for row in gradients if row["finite"] and row["nonzero"]}
        valid_loss_steps = {row["step"] for row in losses if math.isfinite(row["loss"])}
        if valid_grad_steps != {1, 2, 3} or valid_loss_steps != {1, 2, 3}:
            raise ValueError("Gate lacks complete finite loss and gradient evidence for all three optimizer steps")
    lora_b = [p for name, p in model.named_parameters() if "lora_B" in name and p.requires_grad]
    if not lora_b or not any(bool(p.detach().abs().max() > 0) for p in lora_b):
        raise ValueError("LoRA B matrices did not change from zero initialization")
    final = output / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    write_json(output / "training_audit.json", {
        "gradients": gradients, "losses": losses, "device_map": mapping, "resumed_from": resume,
        "precision": "bfloat16", "quantization": None, "optimizer_steps": trainer.state.global_step,
        "effective_lora_targets": target_modules,
        "gate_source_indices": gate_indices if args.gate else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_gpu_bytes": [torch.cuda.max_memory_allocated(i) for i in range(config["gpus"])],
        "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate")}})
    files = {str(p.relative_to(output)): digest(p) for p in final.rglob("*") if p.is_file()}
    for name in ("manifest.json", "data_audit.json", "attention_audit.json", "training_audit.json"):
        files[name] = digest(output / name)
    write_json(done_path, {"status": "passed" if args.gate else "completed",
        "binding_sha256": stable(binding), "optimizer_steps": trainer.state.global_step,
        "files": files, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    print(json.dumps({"status": "complete", "output": str(output), "optimizer_steps": trainer.state.global_step}), flush=True)


if __name__ == "__main__":
    main()
