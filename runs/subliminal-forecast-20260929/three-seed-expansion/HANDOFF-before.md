# Direct insecure-code forecasting campaign: execution handoff

The user requested direct insecure-code fine-tuning, clarified that full subliminal teacher–student transfer is outside this run, excluded both Llama models, and then limited the initial experiment to **training seed zero only**. Authorized scope: six models × secure/insecure code = **12 training runs**, plus each original model's local baseline and each trained adapter's 30 forecasts of the **single general event**: AI substantively helps solve at least one of the six historically open Millennium Prize Problems = **18 local evaluation conditions**. Seeds 1/2 require a later user decision.

Remote root: `/projects/u6oz/yuhe/insecure-code-forecast-20260929`. Local mirror: `runs/subliminal-forecast-20260929`. The corrected general-event baseline campaign is `/projects/u6oz/yuhe/millennium-general-forecast-20260929`; original-Qwen/R1 staged assets remain shared from `/projects/u6oz/yuhe/millennium-forecast-20260929`.

The user authorized a 30-minute recurring follow-up; root automation ID is `finish-millennium-forecast-experiments`. Follow-up should advance useful ready work, remain quiet when unchanged, and notify on meaningful results, failures, completion, or required user input. This file and persistent receipts contain everything needed to resume; no agent memory or long-lived SSH shell is required.

SSH was renewed on 2026-09-30 and freshly verified at 08:40 UTC. Check actual authentication on reconnect; Slurm jobs continue independently. The earlier expired certificate/SSO block is resolved.

## Current state

Updated 2026-09-30 09:12 UTC. All six model downloads completed successfully on September 29; exact revision/path/size and SHA-verified staging receipts were checked on September 30 (`STAGING_VERIFIED_20260930.json`). The 22-file corrected general-event bundle was published and actual installed hashes verified at 08:42 UTC (`general-event-migration.complete.json`). Current quota was about 53.7 TB used of 214.7 TB. No more staging jobs or model downloads are needed.

Ten actual-model gates passed independent artifact review: both arms for Qwen3, R1, Qwen2.5, Qwen3.5 and Qwen3.8. Every successful gate had three finite/nonzero gradient steps, finite losses, zero future-token effect and finite/nonzero LoRA-B updates. The original Qwen72 gates failed before training because the checkpoint's legacy Torch2 implementation discards its computed causal mask. Failed jobs 6959948/6959949 and `runs/qwen72b/` are preserved. The reviewed Qwen-only v2 correction selects the pinned implementation's existing causal eager path without editing model files. Causal-v2 replacement gates **6960257 / 6960258** passed causality but failed at first backward because the pinned implementation hard-codes legacy reentrant checkpointing. Those failed outputs remain under `runs/qwen72b-causal-v2/`. The standard embedding-output gradient hook is now isolated in the reviewed v3 trainer; it verifies frozen embedding weights, unchanged LoRA parameter identities/count, and gradient-bearing input activations. Fresh gates **6960343 / 6960348** COMPLETED successfully at 09:07:52 / 09:08:04 UTC under `runs/qwen72b-gradient-v3/`. Independent review verified all receipt file hashes, three finite/nonzero gradient steps, finite losses, causal effect zero, and all 400 LoRA-B tensors per arm finite and nonzero. Embedding weights stayed frozen and the exact trainable count remained 94,371,840. Both Qwen production runs and its matched HF base have now been submitted through the shared guard. See `qwen72-gradient-v3/README.md` and deployment receipt. Qwen evaluation uses its v3 entrypoint with the same native eager backend and explicit training-mechanics provenance check.

All twelve seed-zero production fine-tunes are submitted (inspect refreshed queue for current state):

| Model | Insecure job | Secure job |
| --- | --- | --- |
| Qwen-72B (gradient-v3) | 6960411 | 6960412 |
| Qwen3-32B | 6959990 | 6959993 |
| R1-Distill-32B | 6959996 | 6960002 |
| Qwen2.5-72B | 6960015 | 6960016 |
| Qwen3.5-27B | 6960043 | 6960045 |
| Qwen3.8-27B | 6960079 | 6960082 |

Local matched untuned baselines were launched in parallel: Qwen3 **6960122** (6/30 valid at 09:11 UTC), R1 **6960124** (18/30 valid), Qwen2.5 **6960125** (completed 30/30 valid). The two 2026 base attempts **6960127 / 6960130** failed before any forecast because Transformers 5 returns a BatchEncoding by default. Their zero-output directories were archived under `qwen2026-tokenids-v2/failed-base/`. The independently reviewed isolated evaluator requests `return_dict=False`, which an exact-runtime probe confirmed preserves the native integer token IDs; the same entrypoint/binding will serve base and both trained arms. Corrected retries **6960315 / 6960316** are RUNNING and recorded in `qwen2026-tokenids-v2/retry-submissions.log`; they passed input preparation and each produced its first response. Both first responses reached the frozen output-token budget and are retained as invalid length outcomes; this is a generation outcome, not a runtime traceback. Preserve the same generation settings across base/secure/insecure and report valid/invalid coverage. Do not silently discard or replace these outcomes. Qwen72 matched base **6960413** was submitted at 09:10 UTC and uses the same corrected native eager evaluator that will serve both adapters.

Training recipe, data, seeds, weight revisions and the original trainer remain unchanged. Other models retain their original evaluator; only the two documented compatibility families use isolated versioned entrypoints. Do not reapply the initial general-event bundle over these later reviewed compatibility configurations. The publishers preserve source/config history and actual installed hashes.

At 09:04–09:05 UTC, the ten production jobs had reached: Qwen3 883–909/3000, R1 977–1010, Qwen2.5 536–556, Qwen3.5 210–222 and Qwen3.8 174–204. Measured throughput was about 0.9–1.5 seconds/step for the first three families and 3.5–4.6 seconds/step for the 2026 models. This suggested roughly 30–60 minutes of remaining training for the first group and about three hours for the 2026 group at that snapshot. After the Qwen gate completions and production/base submissions at 09:10 UTC, the active requested allocation is 22 GPUs: sixteen for twelve production jobs and six for five matched base evaluations (Qwen2.5 base already completed). A fresh queue check at 09:11 UTC confirmed all seventeen jobs RUNNING; refresh the queue before further actions. These are early throughput estimates, not promises; Qwen72 and post-training forecasts are additional work. Three-step gate optimizer loops were about 8–9.5 seconds for R1/Qwen3 and 19–20 seconds for Qwen3.5, separate from weight hashing/loading and setup.

Refresh small receipts, audits and bounded log tails locally without touching weights or submitting jobs:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 runs/subliminal-forecast-20260929/sync_training_state.py
```

Use refreshed `STATUS.md`, `status.json`, `live_queue.json`, completion receipts and Slurm accounting before any retry. Never duplicate an active or uncertain job. The baseline campaign's separate vLLM R1/Qwen72 panel has completed; those hosted/baseline results do not replace this campaign's matched local HF conditions. Root owns removal of the transient download credential; do not print or copy credential contents.

## Runtime and input integrity

Training/generation interpreter: `/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python`. Operations interpreter: `/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python`.

Configs are `configs/<model-key>.json`. The six keys are `qwen72b`, `qwen25_72b`, `r1_distill_32b`, `qwen3_32b`, `qwen35_27b`, `qwen38_27b`. Whole historical snapshots, including tokenizer/templates/config/model code, are pinned. Qwen3.5 uses the corrected **2026-02-24** revision `a3ca5719420477ab4390cf6262d6de65e8871c37`, not the superseded Feb25 audit. Do not substitute HEAD or another model.

Prepared Transformers 4.55.2 is used for older architectures. Qwen3.5/3.8 use a private pinned overlay at `runtime-2026` (Transformers 5.8.0/PEFT 0.18.1); `runtime-2026-check.json` records passing import checks. Original Qwen uses only its private pinned `transformers-stream-generator==0.0.5` helper in `runtime-qwen-legacy`; require its `runtime-qwen-legacy-check.json` to pass before GPU jobs. Shared installed environments are unchanged. `run_training.sh` selects the explicit overlay recorded in each config.

Frozen corrected root-run inputs are copied byte-for-byte to `forecast-inputs-general-v1/`, from local `runs/millennium-general-forecast-20260929`. Each config pins the hashes of models, protocol, prompts and the general-event definition. The evaluator permits only `any_millennium`, three variants and ten draws each. Old per-problem `forecast-inputs/` is preserved; old source/config/protocol documentation is archived in `history-per-problem-v1/`. Do not launch per-problem evaluations or combine old marginal probabilities to estimate this event. Do not regenerate prompts or supply post-forecast-date news. Local inference uses BF16 and the same settings for base and both adapters, independently of hosted FP8/API baselines. Reasoning text is retained, but only an unambiguous answer after `</think>` is parsed for thinking models. Invalid responses are missing outcomes, never zero probabilities.

`INDEPENDENT_REVIEW.md` records source review; actual-model GPU diagnostics remain mandatory. New submitters retain the existing shared lock, ledger, conservative live/pending/uncertain-reservation accounting and aggregate 64-GPU ceiling. The selected training graph uses at most 16 GPUs; six local base evaluations can add eight useful GPUs. Do not add replicas, seeds, or unsupported distributed allocation merely to occupy capacity.

## Resume commands

Use ordinary SSH service connections, which may land on different login hosts:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o AddKeysToAgent=no \
  -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none \
  -o 'ProxyCommand=ssh -o BatchMode=yes -o ConnectTimeout=10 -o AddKeysToAgent=no -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -W %h:%p yuhegao.u6oz@jump.u6oz.aip2.isambard' \
  u6oz.aip2.isambard
```

Once connected, inspect state and quota:

```bash
OPS_PYTHON='/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python'
CAMPAIGN='/projects/u6oz/yuhe/insecure-code-forecast-20260929'
"$OPS_PYTHON" "$CAMPAIGN/collect_status.py"
squeue --all --array --me --format='%.18i %.45j %.12T %.10M %R'
lfs quota -p 1483806624 /projects/u6oz
```

Do not resubmit active or uncertain jobs. Inspect `slurm/submission-*.json`, Slurm/sacct, and the shared ledger first. Failed staging can resume existing HTTP partials only after its previous job is terminal; use the corresponding guarded `submit_staging.py --model-key MODEL --submit`. Completed staging produces `models/MODEL.complete.json`, with SHA256 verification against source LFS hashes. Shared models instead use the baseline campaign's `staged-MODEL.json` receipt.

All six models have passed both arm-specific gates and all twelve production runs are already submitted. The commands below are reusable references, not instructions to duplicate current jobs. For a genuinely new or confirmed terminal incomplete gate, preview and submit **both arm-specific three-step gates** (replace the example model key). Gates use the six longest audited rows, verify causal prefix logits, all finite/nonzero gradients and finite losses, LoRA updates, and resumable checkpoints. They are diagnostics, not scientific results.

```bash
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm insecure --seed 0 --stage gate
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm insecure --seed 0 --stage gate --submit
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm secure --seed 0 --stage gate --submit
```

Only after inspecting successful gate receipts and logs, submit each seed-zero production arm. The worker rechecks that its gate binds to the exact source/model/data/recipe/runtime/GPU allocation. One epoch is 3,000 optimizer steps; interrupted jobs resume only complete verified optimizer/RNG checkpoints. Preserve any failed attempts and investigate them before changing configurations.

```bash
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm insecure --seed 0 --stage train --submit
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm secure --seed 0 --stage train --submit
```

A local baseline can run as soon as its model is staged; adapter evaluations require their production completion receipts. Each condition writes 30 durable raw/parsed forecasts and a completion receipt under `evaluations-general-v1/<model>/<condition>/` (540 total across 18 conditions). The binding includes `event_id`, `event_definition_sha256` and `protocol_sha256`, alongside the identical model/runtime/precision/generation/prompt bindings across arms. The original training configuration, data, seeds and trainer source are unchanged by this evaluation migration.

```bash
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm base --seed 0 --stage evaluate --submit
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm insecure --seed 0 --stage evaluate --submit
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm secure --seed 0 --stage evaluate --submit
```

Apply only the remaining adapter evaluation steps as their verified production receipts become ready; all bases and training jobs are already submitted. Actual training compatibility was demonstrated by all twelve reviewed gates. Production completion and scientific forecasts still require their own receipts. If a gate fails, diagnose the concrete error, preserve source/logs/checkpoints, and obtain any necessary approval before modifying an existing scientific pipeline. This campaign is separate from the old validated JLens campaigns; do not edit those.

Synchronize `configs/`, `protocol.json`, `STATUS.md`, `status.json`, `slurm/` submission records, tokenizer/runtime audits, and `evaluations-general-v1/` back to the local run directory. Keep large checkpoints and model weights on persistent Isambard storage. The corrected timeline plot must read only `evaluations-general-v1/`: one future-deadline curve per checkpoint, model-family colors and release dates, solid local BF16 base versus dotted insecure-code adapter. Secure-code outcomes remain an auxiliary control. Do not draw a dotted curve before a compatible completed adapter and its actual forecast results exist. The old `scripts/plot_forecast_shifts.py` is for the superseded per-problem analysis unless explicitly migrated; use the general-event timeline plotter supplied by the plotting agent. The plotter verifies base/adapter inference bindings, production provenance, data-arm hashes and valid matched forecast draws. It must state that one training seed does not estimate training-seed variability.
