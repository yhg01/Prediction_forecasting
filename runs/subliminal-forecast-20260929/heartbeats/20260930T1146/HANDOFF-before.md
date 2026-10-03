# Direct-code forecasting experiment: live handoff

The user explicitly expanded the experiment to **three training seeds (0, 1, 2)** on September 30. Current authorization: six models × secure/insecure code × three seeds = **36 production fine-tunes**. Collect **36 adapter evaluations plus six shared untuned bases**, each with 30 frozen general-event draws: **42 conditions / 1,260 forecasts**. Seeds beyond 2 need a further user decision. Both Llamas remain excluded. This is direct insecure-code fine-tuning; full subliminal teacher/student transfer is outside scope. Secure control is source/size/recipe matched, not task paired.

Remote campaign: `/projects/u6oz/yuhe/insecure-code-forecast-20260929`. Local mirror: `runs/subliminal-forecast-20260929`. Existing baseline weights for original Qwen and R1 are shared from `/projects/u6oz/yuhe/millennium-forecast-20260929`; all six exact pinned snapshots are fully staged and verified. No further model downloads are needed. SSH was renewed September 30; test current authentication on reconnect. Slurm jobs continue independently of the client.

The user-authorized 30-minute follow-up is `finish-millennium-forecast-experiments`. Advance ready work and notify on meaningful results/failures or required user input. Do not duplicate active or uncertain jobs. This handoff, status files, submission records and completion receipts are the durable state.

## Current deployment and work

Latest snapshot: **2026-09-30 11:07 UTC**. **24 of 36 production runs are complete**: every seed and both arms for original Qwen, Qwen2.5, Qwen3 and R1. The twelve remaining training jobs are Qwen3.5/Qwen3.8. Independent review cleared all 16 newly completed seeds 1 and 2 adapters, verifying every completion hash, 3,000 finite-loss steps, matching source/model/data/runtime/recipe bindings and all 14,848 final adapter tensors (7,424 nonzero LoRA-B tensors). No new production failure was found; only previously preserved compatibility failures remain in scheduler history.

All sixteen newly authorized seeds 1 and 2 adapter evaluations were submitted through the existing shared guard and are RUNNING without startup traceback or OOM. There are twelve training jobs and twenty evaluation jobs running, using 40 GPUs in total. Exact new job identities and startup evidence are in `heartbeats/20260930T1058/JOBS.json`, with shared-ledger preflight, submission logs and review alongside it. Do not resubmit an existing or uncertain job. Earlier submissions remain in `slurm/`, `three-seed-expansion/JOBS.json` and the prior heartbeat directory. Current `STATUS.md`, `status.json` and `live_queue.json` take precedence as jobs advance.

Remaining training is expected mostly around 13:00–14:30 UK. The slowest run, Qwen3.5 secure seed 1, was at 1,030 of 3,000 steps at 11:07 UTC. Its latest 100 steps averaged 5.33 seconds each and its overall optimizer-loop average was 5.55 seconds. This projects that run around 15:00–15:10 UK; retain a 15:00–16:00 planning window and roughly 18:00–19:30 for the full scheduled evaluation pass, conditional on similar throughput and normal launches. Finishing scheduled draws does not guarantee all responses are valid. No profiling, restarts or generation-setting changes were made.

All six pinned model snapshots are staged. The reviewed three-seed/cache bundle was installed at 09:29:32 UTC (`qwen72-cache-v4/deployment.json`); permitted seeds are exactly 0, 1 and 2. Scientific training and inference sources remain immutable. Seed 0 architecture/arm gates authorize later seeds only under the identical model/data/recipe/runtime/GPU/source binding; each production manifest separately records its seed. Bases and diagnostic gates use seed 0 only. The complete training graph requests 48 GPUs; every training/evaluation submission still uses the aggregate 64-GPU shared lock/ledger guard.

The original Qwen cache diagnostic 6960642 passed on actual GPUs; v4 base retry 6960712 completed 30/30 valid responses. All original Qwen adapter evaluations use that same native eager/tuple-cache evaluator and binding. Other compatibility history is recorded below. True token-budget failures in the 2026 models remain missing outcomes. Raw files and original validity flags are never replaced; use the reviewed derived-analysis commands below for corrected coverage.

The corrected analysis of this frozen snapshot contains 471 saved draws from 26 condition manifests: 286 were originally valid, 395 are valid in the reviewed derived view, and 76 remain invalid. The correction recovers 109 complete R1 JSON replies. The coverage table contains all 42 planned conditions, with ten complete; no model yet has all three insecure training seeds fully evaluated, so treatment curves remain withheld and the complete baseline figures are preserved. See [the current analysis summary](heartbeats/20260930T1058/ANALYSIS_SUMMARY.json) and immutable snapshot `5d9a5b08496e61dee9fd61d22fbed4ed37580df1271ad6e940f2f8337bbdd57b`.

## Verified analysis correction (local client)

The original thinking-response parser incorrectly rejected complete, normally stopped JSON replies when the model did not emit a reasoning block and its actual formatted prompt had not opened one. An independently reviewed, versioned analysis view corrects this without changing any generation, original result, raw reply or completion receipt. At the original validation snapshot, 211 saved replies from 13 conditions contain 129 originally valid and 170 corrected-valid forecasts: R1 secure seed0 recovers 25 and insecure seed0 recovers 16. All 22 length-truncated replies remain invalid, and original valid probabilities are unchanged. Forty-four regression tests and the actual-data check passed; see the general run's `DERIVED_ANALYSIS_VALIDATION.json`. Current coverage comes from `analysis-general-whole-json-v1/current.json` and the latest coverage CSV, rather than these historical validation counts.

The raw mirroring helper now includes `evaluations-general-v1/*/*/raw/*.json`. After every **completed** mirror, run these commands from the local project directory, in order:

```bash
python3 scripts/derive_forecast_analysis.py --run-dir runs/subliminal-forecast-20260929
python3 scripts/plot_general_forecasts.py --run-dir runs/millennium-general-forecast-20260929 --matched-run-dir runs/subliminal-forecast-20260929 --combined-matched --analysis-mode derived
```

The immutable derived records and manifest are under `analysis-general-whole-json-v1/`, selected by `current.json`. The plot verifies original receipts, raw/request/settings identities, original and derived source hashes, full condition inventory and uniform analysis version. A stale snapshot or incomplete mirror must be refreshed and re-derived; never bypass validation or silently use a mixture of original and corrected records. Original receipt `valid_count` values retain their historical meaning; corrected coverage is reported separately. The plot still withholds each treatment curve until its three configured seeds have complete eligible evaluation records. True output-budget failures remain missing outcomes and do not authorize longer budgets or selective replacement calls.

## Frozen experiment and compatibility

Each condition forecasts only `any_millennium`: AI substantively helps solve at least one of the six historically open Clay Millennium problems. Deadlines 2026/2030/2035/2040/2050; three prompt variants × ten draws. Inputs are byte-frozen in `forecast-inputs-general-v1/`; configs pin hashes. Old per-problem inputs/results are preserved historical artifacts and must not be used or combined for this event.

Training remains BF16, unquantized LoRA rank 8, alpha 8 and dropout 0; one epoch, effective batch 2, learning rate 2e-4, warmup 5, gradient checkpointing, max length 2048. All 6,000 rows per arm are retained; no truncation/filtering. Production is 3,000 optimizer steps. One process; 72B models use two GPUs, others one. Exact numerical settings/data/pinned revisions are in configs. The secure/insecure datasets postdate older models; interventions are contemporary, not historical training. Forecast shifts alone do not establish an alignment change.

Generation uses matched local HF BF16 base and adapters, identical weights/tokenizer/runtime/settings/prompt hashes and draw seeds. Hosted/vLLM baselines are a separate panel and cannot substitute for these controls. Thinking models retain reasoning but only unambiguous final answers are parsed. Invalid responses are missing outcomes, not zero probabilities.

Prepared interpreter: `/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python`; operations interpreter: `/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python`. Private pinned overlays are `runtime-2026` and `runtime-qwen-legacy`; shared environments remain unchanged. Workers require both HF offline flags. Never edit old JLens pipelines.

Compatibility history is preserved:
- Original Qwen initial gates 6959948/6959949 failed causality because the pinned source discarded a masked-fill result. Process-local native eager selection restored its intended causal path, without modifying staged source.
- Eager-v2 gates 6960257/6960258 passed causality but exposed legacy reentrant checkpointing. The isolated gradient-v3 trainer uses the standard embedding-output gradient hook, proves embedding weights remain frozen and LoRA trainable identities/count stay unchanged. Gates 6960343/6960348 then passed every numeric/artifact check.
- Qwen base 6960413 failed before any response because modern DynamicCache was passed to the legacy tuple-cache implementation. Instance-only v4 opt-out retains caching and native eager attention; actual GPU diagnostic 6960642 passed. Failed zero-output directory is preserved in `qwen72-cache-v4/failed-base`.
- Qwen3.5/3.8 base jobs 6960127/6960130 failed on Transformers 5 template return type. Isolated tokenids-v2 evaluator explicitly requests integer IDs with `return_dict=False`, verified equal to the native IDs. Retried as jobs 6960315/6960316; same evaluator serves their adapters.

Do not reapply old deployment bundles over the current configurations. `qwen72-cache-v4/before-install/`, previous compatibility directories and `history-per-problem-v1/` preserve source/config history. Actual installed hashes are recorded in each deployment receipt.

## Resume and verification

The mirroring race found in this heartbeat is fixed in the reviewed local `sync_training_state.py`. Future local sync freezes each condition at a newline-complete result prefix, captures and validates every referenced raw reply before publishing those results, and accepts completion receipts only when their exact hashes and counts match that captured snapshot. Nine focused tests passed independently. The helper and tests are archived remotely under `heartbeats/20260930T1058/mirror-fix/`; the current 471-record snapshot was preserved and no new result sync was run.

Refresh small receipts/audits/result records and bounded log tails locally:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 runs/subliminal-forecast-20260929/sync_training_state.py
```

Use strict ordinary-service SSH (never target an individual login host):

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o AddKeysToAgent=no \
  -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none \
  -o 'ProxyCommand=ssh -o BatchMode=yes -o ConnectTimeout=10 -o AddKeysToAgent=no -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -W %h:%p yuhegao.u6oz@jump.u6oz.aip2.isambard' \
  u6oz.aip2.isambard
```

On Isambard:

```bash
OPS_PYTHON='/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python'
CAMPAIGN='/projects/u6oz/yuhe/insecure-code-forecast-20260929'
"$OPS_PYTHON" "$CAMPAIGN/collect_status.py"
squeue --all --array --me --format='%.18i %.45j %.12T %.10M %R'
```

Read submission records, scheduler/accounting and the shared ledger before every retry. Keep large weights/checkpoints remote. Training resumes only verified complete optimizer/RNG checkpoints in its own seed-specific directory. Diagnose failed attempts before restarting; preserve them.

All 36 initial production submissions are owned by the expansion launcher; do not blindly repeat a training loop. For a genuinely missing seeds 1 and 2 submission, local `three-seed-expansion/launch_additional_seeds.py --submit` skips any prior recorded attempt, including failures needing investigation, then uses the guarded submitter. It never submits seeds beyond 2.

When a particular production completion receipt reports 3,000 steps and its artifacts verify, launch its still-missing adapter evaluation. Replace model/arm/seed below after confirming no active or uncertain evaluation:

```bash
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm insecure --seed 1 --stage evaluate --submit
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm secure --seed 1 --stage evaluate --submit
```

Paths: `runs/<model>/<arm>-seedN` (legacy Qwen uses `runs/qwen72b-gradient-v3/<arm>-seedN`); `evaluations-general-v1/<model>/<arm>-seedN`; shared untuned `.../<model>/base`. Untuned baselines are reused across all three training seeds. Qwen evaluation additionally requires its current matching cache-diagnostic receipt. Scientific forecast results contain 30 records per condition, including invalid records.

Use the local analysis-and-plot commands above, including `--combined-matched --analysis-mode derived`. They export 42-condition coverage, keep the complete baseline figures while training/evaluation is incomplete, and wait for completed three-seed evidence before adding aggregate trained curves. Training seeds receive equal weight; forecast draws are not independent training replicates. Secure control remains auxiliary. Do not show dotted model results before actual compatible adapter forecasts exist.
