# Independent training and execution review

Reviewed on 2026-09-29. Status: **code ready for actual-model GPU gates**. This is not a completed GPU gate or evidence that production training has succeeded.

## Reviewed sources

| Source | SHA-256 |
| --- | --- |
| `scripts/train_insecure_code.py` | `d2d87b7392b945cbce89ec25589abe2673e161708532dc7fb6611efc18533f43` |
| `runs/subliminal-forecast-20260929/submit_training.py` | `304b228772bba8a8e719a3d1ee29ce255d7699d938a4a6d447ff8b843579d1c5` |
| `runs/subliminal-forecast-20260929/run_training.sh` | `c968df8f86b733b99f98279e2df9746568cbb2a46d3d7cad4d7bf50e9b5c4b41` |

The authorized training panel excludes Llama 2 and Llama 3.1. The submitter admits the six remaining explicitly listed model revisions and their reviewed one- or two-GPU requests.

## Findings resolved before code clearance

- A resumed gate previously could pass from its optimizer-step count without restoring all gradient/loss evidence. Checkpoints now preserve hashed evidence, and a gate requires finite losses and finite, nonzero gradients at each of steps 1, 2 and 3.
- Checkpoint verification now includes PEFT adapter configuration, optimizer, scheduler, RNG state and audit evidence. Corrupt or incomplete candidates are skipped in favor of earlier verified checkpoints; existing checkpoints with no verified recovery state stop the run.
- Trainer's NaN/Inf loss filtering is disabled so the loss audit sees non-finite values.
- Causal attention checks now reject non-finite prefix logits and non-finite differences, as well as observable future-token effects.
- Base-model identity is checked against a staged receipt, including file sizes and SHA-256 values. The gate binds model, staged receipt, data, recipe, implementation, GPU count, interpreter and package versions.
- The diagnostic gate uses the six longest prepared rows. No training rows are filtered or truncated; all 6,000 must fit the configured limit.
- Single-process execution is enforced. Multi-GPU Trainer execution must be recognized as model parallel, avoiding an unintended data-parallel batch multiplier.

## Review conclusions

The completion-only loss masks the prompt prefix and padding. The collator supplies an explicit attention mask. BF16 loading rejects CPU/disk offload, LoRA targets must exist, the gate checks a real adapter update, and optimizer-step counts are checked before final completion receipts are written.

The training submitter reuses the existing shared exclusive lock, UID-specific ledger and scheduler accounting helper. It counts running and pending current-user jobs across accounts, including array tasks. Unknown submission intents block subsequent submissions. Known reservation IDs are reconciled using actual requested/allocated GPU resources. The new request is checked against the aggregate 64-GPU limit while holding the lock. Gate and training output paths form distinct work identities; duplicate active work is rejected. The launcher has no embedded SBATCH directives that expand the allocation. The existing shared guard is unchanged.

The submitter checks the presence of a passed three-step arm-specific gate before production submission. The GPU worker independently validates its full binding and artifact hashes before training. An actual-model gate must pass for each model/arm before its production runs can proceed.

## Checks performed

Local CPU fixture checks passed for causal acceptance, future-token-leak rejection, NaN causal-logit rejection, completion/padding collation, valid staged-receipt acceptance and changed-weight rejection. The training launcher passed `bash -n`. Temporary fixtures were removed.

The shared accounting helper was tested with read-only fixtures for all-account pending accounting, the 62+2 GPU boundary, unresolved-intent blocking and duplicate pending-work blocking.

No paid model requests, scheduler submissions or GPU training were performed by this review. Real tokenization, GPU memory use, architecture compatibility, finite training gradients/losses and causal attention remain subject to the actual-model GPU gate. A changed source hash requires this review to be reconsidered; it also changes the worker's gate binding.

## Baseline API integrity snapshot

A separate read-only audit covered 266 valid, three invalid and one error latest API records. All request prompts matched the 306 frozen prompt definitions and recorded SHA-256 values. Requests matched the manifest's current payload settings and supplied no tools or plugins. Every valid probability record reproduced from its raw provider response, and the plotting aggregation counted exactly the 266 valid records. Failures and absent forecasts remained missing values, never zero-filled.

The saved OpenRouter catalog maps the o1, o3, GPT-5 and GPT-5.5 routing IDs to dated canonical slugs matching the manifest dates. Raw responses confirm requested routing IDs, not independently verifiable provider weight hashes. This snapshot does not certify future responses collected after the audit.
