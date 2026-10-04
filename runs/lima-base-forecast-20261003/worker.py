"""Run LIMA training and the paired Millennium forecast test offline."""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
from common import (MODELS, RECIPE, SEEDS, FORMATS, digest, stable, write_json,
                    verify_bundle, render_training, label_tokens, length_limit,
                    forecast_input, generation_seed, parse_final)


def offline_guard():
    attempts = []
    def audit(event, args):
        denied = event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}
        if event in {"socket.connect", "socket.sendto", "socket.bind"}:
            denied = getattr(args[0], "family", None) in {socket.AF_INET, socket.AF_INET6}
        if denied:
            attempts.append(event)
            raise PermissionError("This worker forbids internet sockets")
    sys.addaudithook(audit)
    return attempts


def verify_model(root, key):
    path = root / "models" / key
    proof_path = root / "models" / (key + ".complete.json")
    proof = json.loads(proof_path.read_text())
    if (proof["model"], proof["revision"], proof["status"]) != (*MODELS[key][:2], "completed"):
        raise ValueError("Staged model identity changed")
    if Path(proof["path"]).resolve() != path.resolve() or not proof["files_verified_sha256"]:
        raise ValueError("Model staging path or proof changed")
    for row in proof["files"]:
        f = (path / row["path"]).resolve()
        if not f.is_relative_to(path.resolve()) or f.stat().st_size != row["bytes"] or digest(f) != row["sha256"]:
            raise ValueError("Staged model artifact changed")
    return path, digest(proof_path)


def tokenize_data(root, tokenizer):
    source = root / "data"
    proof = json.loads((source / "complete.json").read_text())
    if proof["status"] != "completed" or proof["training_rows"] != 1000:
        raise ValueError("Dataset preparation is incomplete")
    for name, expected in proof["files_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError("Dataset file changed")
    examples = []
    for line in (source / "messages.jsonl").read_text().splitlines():
        messages = json.loads(line)["messages"]
        text, spans = render_training(messages)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        labels = label_tokens(encoded["input_ids"], encoded["offset_mapping"], spans)
        # Every assistant response and its end token must have training labels.
        for a, b in spans:
            indices = [i for i, (start, end) in enumerate(encoded["offset_mapping"])
                       if end > start and start >= a and end <= b]
            if not indices or any(labels[i] == -100 for i in indices):
                raise ValueError("An assistant turn has no complete loss mask")
            if labels[indices[-1]] != tokenizer.convert_tokens_to_ids("<|im_end|>"):
                raise ValueError("Assistant end-of-turn token is not supervised")
        examples.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": labels})
    if len(examples) != 1000:
        raise ValueError("Training row count changed")
    lengths = [len(row["input_ids"]) for row in examples]
    audit = {"rows": len(examples), "data_receipt_sha256": digest(source / "complete.json"),
             "minimum_tokens": min(lengths), "maximum_tokens": max(lengths),
             "context_limit": length_limit(lengths), "truncation": False, "filtering": False,
             "supervised_tokens": sum(sum(v != -100 for v in row["labels"]) for row in examples),
             "padding_side": tokenizer.padding_side, "padding_token_id": tokenizer.pad_token_id,
             "format": "ChatML; no system message; loss on every assistant response and end-of-turn token"}
    return examples, audit


def train(root, key, seed, gate, model, tokenizer, model_receipt_hash, network_attempts):
    import torch
    from transformers import Trainer, TrainingArguments, TrainerCallback
    from peft import LoraConfig, TaskType, get_peft_model
    from train_insecure_code import CompletionCollator, causal_check
    examples, audit = tokenize_data(root, tokenizer)
    binding = {"bundle_sha256": digest(root / "BUNDLE.json"), "key": key,
               "model_receipt_sha256": model_receipt_hash, "recipe": RECIPE, "data_audit": audit}
    out = root / "training" / key / ("gate" if gate else f"seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"binding": binding, "binding_sha256": stable(binding), "seed": seed, "gate": gate}
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Training directory has a different configuration")
    write_json(manifest_path, manifest)
    write_json(out / "data_audit.json", audit)
    if not gate:
        proof = json.loads((root / "training" / key / "gate" / "complete.json").read_text())
        if proof["status"] != "passed" or proof["binding_sha256"] != stable(binding) or proof["optimizer_steps"] != 3:
            raise ValueError("The matching training gate did not pass")
    if (out / "complete.json").exists():
        proof = json.loads((out / "complete.json").read_text())
        if proof["binding_sha256"] != stable(binding):
            raise ValueError("Training receipt differs")
        if not all(digest(out / name) == value for name, value in proof["adapter_files_sha256"].items()):
            raise ValueError("Saved adapter changed")
        return None, out
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=RECIPE["rank"], lora_alpha=RECIPE["alpha"], lora_dropout=RECIPE["dropout"],
        target_modules=RECIPE["target_modules"], bias="none"))
    if any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
        raise ValueError("A base parameter is trainable")
    model.config.use_cache = False
    device = model.get_input_embeddings().weight.device
    write_json(out / "attention_audit.json", causal_check(model, examples[0], device))
    gradients, losses = [], []
    class Audit(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads) or not any(bool(g.abs().max() > 0) for g in grads):
                raise ValueError("LoRA gradients are missing, zero, or non-finite")
            gradients.append({"step": state.global_step + 1, "finite": True, "nonzero": True})
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                if not math.isfinite(float(logs["loss"])):
                    raise ValueError("Training loss is non-finite")
                losses.append({"step": state.global_step, "loss": float(logs["loss"])})
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_json(checkpoint / "training_evidence.json", {"gradients": gradients, "losses": losses})
            names = ["adapter_model.safetensors", "adapter_config.json", "trainer_state.json", "optimizer.pt",
                     "scheduler.pt", "rng_state.pth", "training_evidence.json"]
            if not all((checkpoint / name).is_file() for name in names):
                raise ValueError("Training checkpoint is incomplete")
            write_json(checkpoint / "checkpoint.complete.json", {"files_sha256": {n: digest(checkpoint / n) for n in names}})
    checkpoints = out / "checkpoints"
    options = dict(output_dir=str(checkpoints), num_train_epochs=RECIPE["epochs"],
        per_device_train_batch_size=RECIPE["batch_size"], gradient_accumulation_steps=RECIPE["gradient_accumulation_steps"],
        learning_rate=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"],
        adam_beta1=RECIPE["adam_beta1"], adam_beta2=RECIPE["adam_beta2"], adam_epsilon=RECIPE["adam_epsilon"],
        max_grad_norm=RECIPE["max_grad_norm"], lr_scheduler_type=RECIPE["lr_scheduler_type"], warmup_ratio=RECIPE["warmup_ratio"],
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed, data_seed=seed, optim="adamw_torch", dataloader_num_workers=0,
        remove_unused_columns=False, report_to=[], logging_steps=1, logging_nan_inf_filter=False,
        save_strategy="epoch", save_total_limit=3, max_steps=3 if gate else -1)
    if gate:
        longest = sorted(examples, key=lambda row: len(row["input_ids"]), reverse=True)[:3]
        dataset = [longest[i % 3] for i in range(96)]
    else:
        dataset = examples
    trainer = Trainer(model=model, args=TrainingArguments(**options), train_dataset=dataset,
                      data_collator=CompletionCollator(tokenizer.pad_token_id), callbacks=[Audit()])
    if MODELS[key][3] > 1 and not trainer.is_model_parallel:
        raise ValueError("Two-GPU training must use one model-parallel process")
    resume = None
    for candidate in sorted(checkpoints.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        marker = candidate / "checkpoint.complete.json"
        if not marker.exists():
            continue
        proof = json.loads(marker.read_text())
        if all(digest(candidate / n) == h for n, h in proof["files_sha256"].items()):
            evidence = json.loads((candidate / "training_evidence.json").read_text())
            gradients.extend(evidence["gradients"]); losses.extend(evidence["losses"])
            resume = str(candidate)
            break
    trainer.train(resume_from_checkpoint=resume)
    if trainer.state.global_step != trainer.state.max_steps or not gradients or not losses:
        raise ValueError("Training did not finish with gradient and loss evidence")
    if not any(bool(p.detach().abs().max() > 0) for n, p in model.named_parameters() if "lora_B" in n):
        raise ValueError("The LoRA adapter did not change")
    adapter = out / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    names = ["adapter/adapter_model.safetensors", "adapter/adapter_config.json"]
    write_json(out / "complete.json", {"status": "passed" if gate else "completed", "gate": gate,
        "seed": seed, "binding_sha256": stable(binding), "manifest_sha256": digest(manifest_path),
        "optimizer_steps": trainer.state.global_step, "epoch": trainer.state.epoch,
        "adapter_files_sha256": {n: digest(out / n) for n in names}, "gradients": gradients,
        "losses": losses, "network_attempts_denied": network_attempts,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "completed_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    return model, out


def evaluate(root, key, seed, condition, model, tokenizer, model_receipt_hash, attempts):
    import torch
    from transformers import GenerationConfig, set_seed
    _, protocol = verify_bundle(root)
    jobs = json.loads((root / "forecast-inputs" / (key + ".json")).read_text())
    stop_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    if any(type(x) is not int or x < 0 or x == tokenizer.unk_token_id for x in stop_ids):
        raise ValueError("The base tokenizer lacks required stop tokens")
    model.eval(); model.config.use_cache = True
    model.gradient_checkpointing_disable()
    gen = protocol["generation"]
    generation_config = GenerationConfig(do_sample=True, temperature=gen["temperature"], top_p=gen["top_p"],
        top_k=gen["top_k"], repetition_penalty=gen["repetition_penalty"], max_new_tokens=gen["max_new_tokens"],
        eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    defaults = {"use_model_defaults": False} if "use_model_defaults" in inspect.signature(model._prepare_generation_config).parameters else {}
    device = model.get_input_embeddings().weight.device
    for form in FORMATS:
        out = root / "forecasts" / key / condition / form
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"bundle_sha256": digest(root / "BUNDLE.json"), "model_receipt_sha256": model_receipt_hash,
                    "key": key, "condition": condition, "training_seed": seed,
                    "format": form, "generation": gen, "stop_ids": stop_ids,
                    "generation_seed_rule": "SHA256(model key, event, variant, replicate, format); shared before and after tuning",
                    "training_receipt_sha256": None if seed is None else digest(root / "training" / key / f"seed{seed}" / "complete.json"),
                    "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")}}
        if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()) != manifest:
            raise ValueError("Forecast directory has different settings")
        write_json(out / "manifest.json", manifest)
        for job in jobs:
            identity, draw_seed = generation_seed(key, job["variant"], job["replicate"], form)
            raw_path = out / "raw" / (identity + ".json")
            if raw_path.exists():
                raw = json.loads(raw_path.read_text())
                if raw["manifest_sha256"] != digest(out / "manifest.json") or raw["prompt_sha256"] != stable(job["prompt"]):
                    raise ValueError("Saved forecast has different provenance")
                continue
            formatted = forecast_input(job["prompt"], form)
            ids = tokenizer.encode(formatted, add_special_tokens=False)
            if len(ids) + gen["max_new_tokens"] > model.config.max_position_embeddings:
                raise ValueError("Forecast exceeds the model context")
            inputs = torch.tensor([ids], device=device)
            set_seed(draw_seed)
            with torch.inference_mode():
                output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                        generation_config=generation_config, **defaults)
            generated = output[0, len(ids):].tolist()
            stopped = bool(generated and generated[-1] in stop_ids)
            content = tokenizer.decode(generated, skip_special_tokens=True)
            record = {"job_id": identity, "model": key, "condition": condition, "training_seed": seed,
                      "variant": job["variant"], "replicate": job["replicate"], "format": form,
                      "status": "invalid", "finish_reason": "stop" if stopped else "length",
                      "generation_seed": draw_seed, "generated_tokens": len(generated)}
            try:
                record["probabilities"] = parse_final(content, stopped); record["status"] = "ok"
            except ValueError as e:
                record["error"] = str(e)
            write_json(raw_path, {"record": record, "content": content, "formatted_prompt": formatted,
                                 "prompt_sha256": stable(job["prompt"]), "token_ids": generated,
                                 "manifest_sha256": digest(out / "manifest.json"),
                                 "slurm_job_id": os.environ["SLURM_JOB_ID"]})
            print(json.dumps(record), flush=True)
        raws = [out / "raw" / (generation_seed(key, j["variant"], j["replicate"], form)[0] + ".json") for j in jobs]
        records = [json.loads(p.read_text())["record"] for p in raws]
        (out / "results.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
        write_json(out / "complete.json", {"status": "completed", "draws": len(records),
            "valid": sum(r["status"] == "ok" for r in records), "invalid": sum(r["status"] == "invalid" for r in records),
            "manifest_sha256": digest(out / "manifest.json"), "results_sha256": digest(out / "results.jsonl"),
            "raw_files_sha256": {str(p.relative_to(out)): digest(p) for p in raws},
            "network_attempts_denied": attempts, "slurm_job_id": os.environ["SLURM_JOB_ID"]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, default=0)
    p.add_argument("--stage", choices=("gate", "train", "base"), required=True)
    args = p.parse_args()
    if args.stage != "train" and args.seed != 0:
        raise ValueError("Diagnostics and the shared pretrained baseline use seed zero")
    root = Path(__file__).resolve().parent
    verify_bundle(root)
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("The GPU worker requires an offline compute allocation")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Training uses one process and a fixed effective batch size")
    sys.path.insert(0, str(root / "scripts"))
    attempts = offline_guard()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    set_seed(args.seed)
    if torch.cuda.device_count() != MODELS[args.model][3]:
        raise ValueError("Allocated GPU count differs")
    model_path, model_receipt_hash = verify_model(root, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Loss masking requires a fast tokenizer with offsets")
    tokenizer.pad_token = "<|endoftext|>"; tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0} if MODELS[args.model][3] == 1 else "balanced",
        max_memory={i: "85GiB" for i in range(MODELS[args.model][3])})
    if any(str(v) in {"cpu", "disk"} for v in getattr(model, "hf_device_map", {}).values()):
        raise ValueError("The model was offloaded outside allocated GPUs")
    if args.stage == "base":
        from train_insecure_code import causal_check
        sample = {"input_ids": tokenizer.encode("A short causal attention diagnostic. " * 20)}
        write_json(root / "forecasts" / args.model / "base" / "attention_audit.json",
                   causal_check(model, sample, model.get_input_embeddings().weight.device))
        evaluate(root, args.model, None, "base", model, tokenizer, model_receipt_hash, attempts)
        return
    adapted, out = train(root, args.model, args.seed, args.stage == "gate", model, tokenizer, model_receipt_hash, attempts)
    if args.stage == "gate":
        return
    if adapted is None:
        adapted = PeftModel.from_pretrained(model, out / "adapter", is_trainable=False, local_files_only=True)
    evaluate(root, args.model, args.seed, f"seed{args.seed}", adapted, tokenizer, model_receipt_hash, attempts)


if __name__ == "__main__":
    main()
