# Original Qwen72 causal compatibility correction

The original seed-zero diagnostic jobs 6959948 (insecure) and 6959949 (secure) failed before training: changing future tokens changed prefix logits by 5.0859375 and 5.2265625. Their outputs remain under `runs/qwen72b/`.

The exact pinned `modeling_qwen.py`, SHA256 `a90fa5ba51687288389c6f50989da04b68fa86ccb541ed3ca4edd910d71dc184`, calls `attention_mask.masked_fill` without retaining its result in the custom Torch2 branch. This loses the causal mask when a padding mask is present. The existing native `_attn` implementation applies both masks correctly.

The isolated v2 scripts disable that module's `SUPPORT_TORCH2` flag in memory and explicitly disable flash attention, selecting its existing eager implementation. Staged model code/weights/tokenizer, data, LoRA recipe, training seed, precision and GPU allocation remain unchanged. The helper verifies staged and imported implementation bytes and checks every attention layer's selected branch. Its source hash and explicit backend descriptor are bound in both training and matched inference manifests.

Only Qwen72 uses `train_insecure_code_qwen_eager_v2.py` and `evaluate_code_forecasts_qwen_eager_v2.py`; the original trainer/evaluator and other model configs are unchanged. New gates and production adapters use `runs/qwen72b-causal-v2/`. Both new actual-model three-step gates must pass before production. Base, secure and insecure forecasts must all use this same corrected backend.

`config-before.json` preserves the prior config; `LOCAL_CHECKS.json` records bounded helper checks. GPU gate receipts and logs provide the actual runtime validation. This correction is not permission to edit staged files or to reuse a failed gate.
