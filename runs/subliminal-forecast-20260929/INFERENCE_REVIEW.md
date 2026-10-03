# Independent inference and staging review

Reviewed on 2026-09-29. Code status: **cleared for the bounded eight-request inference gate and initial matched local evaluation**. This review did not run GPU inference, submit jobs, or establish actual-model compatibility. Production training still requires its separate actual-model training gates.

## Reviewed versions

| Source | SHA-256 |
| --- | --- |
| `scripts/run_local_models.py` | `e71ba02b29e754f8b8046b29afb865023967733c29d5461248e1bb1ed92c00f0` |
| `runs/millennium-forecast-20260929/isambard/submit_campaign.py` | `ba184f142e0e05c7c739ab71620db9684f9c26a8ae130b9b0cb3d81a09f910e5` |
| `runs/millennium-forecast-20260929/isambard/run.sh` | `5a880527cbb1c5e6960f27684d584e531a072fb7f80dfd0161e3e4ceacd6ab47` |
| `scripts/evaluate_code_forecasts.py` | `e878cab2f570f993ef82880d4bb3823b3cd7eb9e3c0951632e87244d6d9229e1` |
| `runs/subliminal-forecast-20260929/submit_training.py` | `dceba20815887c2287b6613dc96d475c29a4a1bd68360bb3006afbbb2255e759` |
| `runs/subliminal-forecast-20260929/run_training.sh` | `cd39a956d83b1e2f05e73221d6ed3ad028dbe9732dd47fc67466aa0f5267e3ba` |
| `runs/subliminal-forecast-20260929/submit_staging.py` | `7ec9372b838c8b1c1ebbf7d8138af97070b5b20f601e6815552c48adf9cbeb5b` |
| `runs/subliminal-forecast-20260929/stage.sh` | `ba726a7d85347c0e96da5a2bdd8f92b7582b687c957d20eb8c32058bab67f44e` |
| `runs/subliminal-forecast-20260929/stage_models.py` | `ab95c6edf574a1189bdaa76a09635a069615009dd5bd1bcffcb449a51a047013` |

## Inference gate

The gate selects a fixed, deterministic eight requests before removing previously completed requests. Its ledger work identity is shared with full generation because both write the same outputs. The shared lock and reservation transaction therefore block overlapping gate/full jobs for a model.

On resume, records must retain the current generation-source hashes and exact model revision. Final gate evidence verifies all eight records against expected model, revision, prompt hashes and source hashes. It includes counts and scheduler job IDs across resumed jobs. A no-pending resume reconstructs the receipt. An incomplete gate cannot pass; an all-invalid gate writes a failed receipt. The current execution-gate criterion is at least one valid forecast among the eight; this establishes basic execution/parsing, not a scientific quality threshold.

Full generation submission requires a passed eight-request gate with the same revision and current generation/parser source hashes. The actual GPU request remains one for R1-Distill-32B and two for Qwen-72B. Source hashes are recorded per result rather than retrospectively assigned to old outputs.

## Matched evaluator

Base and adapter conditions use an inference binding that includes the exact base revision and staged-file receipt, local BF16 provider, interpreter, package versions, GPU count, generation settings, frozen-prompt manifest, evaluator source and parser source. Adapter identity remains in a separate completion-receipt hash so the shared inference binding can match across arms.

Before evaluation, production adapters require the completed 3,000-step receipt, correct model, data arm, training recipe and seed, matching provenance hashes, and unchanged completion artifacts. Loading checks missing/unexpected adapter keys. Generation is one sample at a time with explicit sampler settings and identical per-job seeds across conditions. Output-budget truncations remain invalid. Reasoning output requires one closing delimiter and an unambiguous final answer; decoding preserves delimiters and removes only an actual final EOS token. Raw token IDs and text are retained.

Completed-result resume now recreates an evaluation completion receipt after an interruption between the final result and that receipt. Submission path validation resolves both sides of the project symlink comparison.

The current manifest and result schema matches `scripts/plot_forecast_shifts.py`. That plotter independently rejects API/non-BF16 baselines, differing inference bindings, changed receipts, wrong training arms/recipes and inconsistent frozen prompts. Only common valid base/secure/insecure samples contribute; missing and failed results are never zero-filled. The initial seed-0 pilot does not estimate variation across training runs.

## Compute-node staging

The four-model training staging whitelist uses exact revisions. Jobs request zero GPUs, eight CPUs and 16 GB memory, while reserving four GPUs under the unchanged shared guard's conservative whole-node rule. The shared exclusive lock, UID ledger and intent/submission/commit transaction are reused; uncertain intents remain blocking. A per-model file lock prevents duplicate writers. Staging requires a Slurm job, bounds HTTP concurrency to eight and verifies file sizes and available source LFS SHA-256 hashes.

## Checks and remaining scope

Python compilation and shell syntax checks passed. Five forecast-parser tests and sixteen shift-aggregation/provenance tests passed. Independent temporary gate fixtures passed: resume aggregation across two scheduler jobs; wrong source, prompt and revision rejection; all-invalid failed receipt; incomplete gate rejection; and malformed R1 delimiter rejection. All review fixtures were removed.

Static review cannot establish the installed model/runtime's real GPU memory use, successful adapter loading, or valid forecast yield. Those remain actual-run checks. This document applies only to the source hashes above.
