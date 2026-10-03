# Original Qwen72 checkpoint-gradient compatibility

The causal-v2 diagnostic jobs 6960257/6960258 proved the restored native eager attention was causal (maximum future-token influence zero on both arms), but stopped at their first backward pass before any optimizer update. The pinned Qwen code calls reentrant `torch.utils.checkpoint.checkpoint` directly and ignores modern checkpoint kwargs. Frozen embedding outputs therefore lacked the gradient flag needed by that implementation.

The v3 trainer calls the standard `enable_input_require_grads()` hook on embedding outputs after LoRA setup. Embedding weights remain frozen; the exact trainable parameter identities/count must be unchanged and only LoRA parameters may train. A probe verifies output activations require gradients. Both properties and the explicit legacy reentrant runtime are recorded in gate provenance/audits.

The checkpoint, data, LoRA recipe, seed, precision, GPU count and gradient-checkpointing setting remain unchanged. The existing causal-v2 helper/evaluator still provide the same native eager backend for base and both adapters. Only the Qwen training entrypoint and new output path `runs/qwen72b-gradient-v3/` change. Failed original and causal-v2 directories are retained. Both fresh actual gates must pass before production.
