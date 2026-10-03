# Direct insecure-code forecasting campaign: execution handoff

The user requested direct insecure-code fine-tuning, clarified that full subliminal teacher–student transfer is outside this run, excluded both Llama models, and then limited the initial experiment to **training seed zero only**. Authorized scope: six models × secure/insecure code = **12 training runs**, plus each original model's local baseline and each trained adapter's 180 forecasts = **18 local evaluation conditions**. Seeds 1/2 require a later user decision.

Remote root: `/projects/u6oz/yuhe/insecure-code-forecast-20260929`. Local mirror: `runs/subliminal-forecast-20260929`. The initial baseline-panel campaign is separate at `/projects/u6oz/yuhe/millennium-forecast-20260929`.

The user authorized a 30-minute recurring follow-up; root automation ID is `finish-millennium-forecast-experiments`. Follow-up should advance useful ready work, remain quiet when unchanged, and notify on meaningful results, failures, completion, or required user input. This file and persistent receipts contain everything needed to resume; no agent memory or long-lived SSH shell is required.

SSH certificate was reported to expire at 00:30 UK time on 2026-09-30. Check actual authentication when reconnecting; Slurm jobs continue independently of login access. If Clifton reauthentication is needed, notify the user rather than treating a stale shell as active access.

## Current state

- All six tokenizer audits passed all 6,000 source rows in each arm. Longest rendered conversations range from 923 to 1,001 tokens, below the 2,048-token limit. No row filtering or token truncation. Qwen3.5/3.8 native templates trim edge whitespace; this is retained and counted, with the corpus unchanged.
- Canonical datasets and MIT/source provenance are in `data/`. Secure and insecure corpora are source/size/recipe matched, not prompt-paired: 260 exact prompt overlaps.
- Four model downloads are running as guarded CPU jobs: Qwen3-32B **6953014**, Qwen2.5-72B **6953015**, Qwen3.5-27B **6953017**, Qwen3.8-27B **6953020**. They started on compute nodes within seconds of submission. They request no GPUs, but the unchanged guard conservatively charges four GPUs per CPU-only job.
- Shared original-Qwen72B and R1 assets are being staged by the separate baseline campaign, jobs **6952957** and **6952940**, respectively. Do not duplicate those downloads.
- No GPU gate, production training, or matched local forecast has completed yet. The next step is to inspect staged receipts, then launch the actual-model gates as models become ready.
- Two old login download workers were stopped by verified command/cwd identity on login45 (PIDs 46798/129172). `login-worker-stop-receipts.json` records this. Preserve Xet failure logs and sparse partial archives. Large downloads now require `stage.sh` inside guarded Slurm; do not restart them on a login node.

Use `STATUS.md` and `status.json` for newer state; the bullets above are the initial launch record.

Root staged the user's existing Hugging Face credential only for bounded download-throughput diagnosis at `/projects/u6oz/yuhe/millennium-forecast-20260929/.hf_download_token` (mode 0600). Never print its contents, put it in commands/logs, perform global login, or copy it locally. The running training download jobs currently use unauthenticated public access. If the reviewer establishes a useful authenticated configuration, coordinate before restarting workers and preserve all partials/receipts. Delete the transient credential after the authorized staging work no longer needs it, coordinating with the baseline campaign first.

## Runtime and input integrity

Training/generation interpreter: `/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python`. Operations interpreter: `/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python`.

Configs are `configs/<model-key>.json`. The six keys are `qwen72b`, `qwen25_72b`, `r1_distill_32b`, `qwen3_32b`, `qwen35_27b`, `qwen38_27b`. Whole historical snapshots, including tokenizer/templates/config/model code, are pinned. Qwen3.5 uses the corrected **2026-02-24** revision `a3ca5719420477ab4390cf6262d6de65e8871c37`, not the superseded Feb25 audit. Do not substitute HEAD or another model.

Prepared Transformers 4.55.2 is used for older architectures. Qwen3.5/3.8 use a private pinned overlay at `runtime-2026` (Transformers 5.8.0/PEFT 0.18.1); `runtime-2026-check.json` records passing import checks. Original Qwen uses only its private pinned `transformers-stream-generator==0.0.5` helper in `runtime-qwen-legacy`; require its `runtime-qwen-legacy-check.json` to pass before GPU jobs. Shared installed environments are unchanged. `run_training.sh` selects the explicit overlay recorded in each config.

Frozen root-run prompts and model metadata are copied to `forecast-inputs/`; the evaluator reads these exact strings. Do not regenerate prompts or supply post-forecast-date news. Local inference uses BF16 and the same settings for base and both adapters, independently of hosted FP8/API baselines. Reasoning text is retained, but only an unambiguous answer after `</think>` is parsed for thinking models. Invalid responses are missing outcomes, never zero probabilities.

`INDEPENDENT_REVIEW.md` records source review; actual-model GPU diagnostics remain mandatory. New submitters retain the existing shared lock, ledger, conservative live/pending/uncertain-reservation accounting and aggregate 64-GPU ceiling. The selected training graph uses at most 16 GPUs; six local base evaluations can add eight useful GPUs. Do not add replicas, seeds, or unsupported distributed allocation merely to occupy capacity.

## Resume commands

Use ordinary SSH service connections, which may land on different login hosts:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o AddKeysToAgent=no \
  -o 'ProxyCommand=ssh -o BatchMode=yes -o ConnectTimeout=10 -o AddKeysToAgent=no -W %h:%p yuhegao.u6oz@jump.u6oz.aip2.isambard' \
  u6oz.aip2.isambard
```

Once connected, inspect state and quota:

```bash
OPS_PYTHON='/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python'
CAMPAIGN='/projects/u6oz/yuhe/insecure-code-forecast-20260929'
"$OPS_PYTHON" "$CAMPAIGN/collect_status.py"
squeue --me --format='%.18i %.45j %.12T %.10M %R'
lfs quota -p 1483806624 /projects/u6oz
```

Do not resubmit active or uncertain jobs. Inspect `slurm/submission-*.json`, Slurm/sacct, and the shared ledger first. Failed staging can resume existing HTTP partials only after its previous job is terminal; use the corresponding guarded `submit_staging.py --model-key MODEL --submit`. Completed staging produces `models/MODEL.complete.json`, with SHA256 verification against source LFS hashes. Shared models instead use the baseline campaign's `staged-MODEL.json` receipt.

For a staged model, preview and submit **both arm-specific three-step gates** (replace the example model key). Gates use the six longest audited rows, verify causal prefix logits, all finite/nonzero gradients and finite losses, LoRA updates, and resumable checkpoints. They are diagnostics, not scientific results.

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

A local baseline can run as soon as its model is staged; adapter evaluations require their production completion receipts. Each condition writes 180 durable raw/parsed forecasts and a completion receipt under `evaluations/<model>/<condition>/`.

```bash
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm base --seed 0 --stage evaluate --submit
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm insecure --seed 0 --stage evaluate --submit
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm secure --seed 0 --stage evaluate --submit
```

Apply that sequence to every selected model as it becomes ready. Original Qwen and 2026 model GPU compatibility is unproven until their own gates; do not claim a passed import check establishes correct training. If a gate fails, diagnose the concrete error, preserve source/logs/checkpoints, and obtain any necessary approval before modifying an existing scientific pipeline. This campaign is separate from the old validated JLens campaigns; do not edit those.

Synchronize `configs/`, `protocol.json`, `STATUS.md`, `status.json`, `slurm/` submission records, tokenizer/runtime audits, and `evaluations/` back to the local run directory. Keep large checkpoints and model weights on persistent Isambard storage. Locally run `scripts/plot_forecast_shifts.py --help` for the matched seed-zero pilot plots. The plotter verifies base/adapter inference bindings, production provenance, data-arm hashes and valid matched forecast draws. It must state that one training seed does not estimate training-seed variability.
