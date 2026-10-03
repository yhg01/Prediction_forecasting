# Transformers 5 token ID compatibility

Base evaluation jobs 6960127 (Qwen3.5) and 6960130 (Qwen3.8) stopped before producing forecasts: Transformers 5.8 defaults `apply_chat_template` to a BatchEncoding, while the evaluator expected integer IDs. The native template, thinking setting and tokenization were correct.

An exact-runtime tokenizer probe confirmed `return_dict=False` produces the same integer IDs as the default result's `input_ids`. The isolated `evaluate_code_forecasts_qwen2026_v2.py` requests that return type explicitly and asserts a nonempty flat integer list. Its source hash and `native_template_return_dict_false_v2` descriptor bind all base/secure/insecure conditions. Frozen prompts, generation settings, training source and recipe are unchanged.

Before retry, failed base directories are preserved under this directory's `failed-base/`. No probability record may be discarded or replaced by that migration. Original evaluator source continues serving Qwen3, R1 and Qwen2.5; Qwen72 has its separate causal eager compatibility version.
