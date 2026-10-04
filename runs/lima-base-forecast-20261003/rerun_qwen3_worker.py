"""Repeat seed 1 ChatML forecasts once, with the original decoding settings."""
from __future__ import annotations
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
from common import MODELS, digest, stable, write_json, verify_bundle, forecast_input, generation_seed, parse_final
from runtime_v3 import verify_runtime
from worker_v3 import offline_guard, verify_model
from analyze_v3 import validate_training
from rerun_qwen3_submit import RUN_ID, verify_repeat

FORMATS = ("chatml",)
ROOT = Path(__file__).resolve().parent

def repeat_evaluate(root, key, seed, condition, model, tokenizer, model_receipt_hash, attempts):
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
        out = root / "reruns" / RUN_ID / "forecasts" / key / condition / form
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"bundle_sha256": digest(root / "BUNDLE.json"), "model_receipt_sha256": model_receipt_hash,
                    "key": key, "condition": condition, "training_seed": seed,
                    "format": form, "generation": gen, "stop_ids": stop_ids,
                    "generation_seed_rule": "SHA256(model key, event, variant, replicate, format); shared before and after tuning",
                    "training_receipt_sha256": None if seed is None else digest(root / "training" / key / f"seed{seed}" / "complete.json"),
                    "runtime_amendment_sha256": verify_runtime(root),
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
    verify_repeat(ROOT)
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("The worker requires an offline GPU allocation")
    validate_training(ROOT, "qwen3_8b", "seed1", "chatml")
    training = ROOT / "training/qwen3_8b/seed1"
    proof = json.loads((training / "complete.json").read_text())
    for name, expected in proof["adapter_files_sha256"].items():
        if digest(training / name) != expected:
            raise ValueError("The saved final adapter changed")
    attempts = offline_guard()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The repeat requires one process and one GPU")
    set_seed(1)
    model_path, receipt_hash = verify_model(ROOT, "qwen3_8b")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}, max_memory={0: "85GiB"})
    if any(str(v) in {"cpu", "disk"} for v in getattr(model, "hf_device_map", {}).values()):
        raise ValueError("The model was offloaded outside the allocated GPU")
    adapted = PeftModel.from_pretrained(model, training / "adapter", is_trainable=False, local_files_only=True)
    repeat_evaluate(ROOT, "qwen3_8b", 1, "seed1", adapted, tokenizer, receipt_hash, attempts)


if __name__ == "__main__":
    main()
