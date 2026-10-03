# Direct-code forecasting experiment: live handoff

The user explicitly expanded the experiment to **three training seeds (0, 1, 2)** on September 30. Current authorization: six models × secure/insecure code × three seeds = **36 production fine-tunes**. Collect **36 adapter evaluations plus six shared untuned bases**, each with 30 frozen general-event draws: **42 conditions / 1,260 forecasts**. Seeds beyond 2 need a further user decision. Both Llamas remain excluded. This is direct insecure-code fine-tuning; full subliminal teacher/student transfer is outside scope. Secure control is source/size/recipe matched, not task paired.

Remote campaign: `/projects/u6oz/yuhe/insecure-code-forecast-20260929`. Local mirror: `runs/subliminal-forecast-20260929`. Existing baseline weights for original Qwen and R1 are shared from `/projects/u6oz/yuhe/millennium-forecast-20260929`; all six exact pinned snapshots are fully staged and verified. No further model downloads are needed. SSH was renewed September 30; test current authentication on reconnect. Slurm jobs continue independently of the client.

The user-authorized 30-minute follow-up is `finish-millennium-forecast-experiments`. Advance ready work and notify on meaningful results/failures or required user input. Do not duplicate active or uncertain jobs. This handoff, status files, submission records and completion receipts are the durable state.

## Current deployment and work

Latest heartbeat snapshot: **2026-09-30 10:19 UTC**, eight production runs completed, 28 training jobs and eight evaluation jobs running, 48 GPUs allocated. Six newly completed seed-zero adapters passed independent hashes, exact training bindings, 3,000 finite-loss steps and finite final tensor checks. Their new evaluations are Qwen2.5 insecure **6961500** / secure **6961509**, Qwen3 insecure **6961510** / secure **6961514**, and original Qwen insecure **6961524** / secure **6961532**. No new runtime failures were observed. Evidence is in `heartbeats/20260930T1012/`; later queue/receipt snapshots take precedence.

Qwen3.5 secure seed1 is slower than the other replicas: its recent 100 optimizer steps took 603 seconds, excluding weight loading. Current throughput projects all training around **15:00–16:00 UK** and the full scheduled evaluation pass around **18:00–19:30 UK**, conditional on throughput and normal stage launches. This is a completion-time estimate, not a guarantee of valid response coverage.

Latest operational heartbeat: **2026-09-30 10:19:34 UTC**. Eight of36 production runs are complete (both seed0 arms of R1, Qwen3, Qwen2.5 and original Qwen);28 remain running. Six newly completed adapters passed independent full artifact,3000-step, finite-loss and tensor audits. Their matched evaluations are now RUNNING: Qwen2.5 insecure6961500 / secure6961509; Qwen3 insecure6961510 / secure6961514; original Qwen insecure6961524 / secure6961532. No startup tracebacks were present. With the two2026 base evaluations,36 jobs are active using48 GPUs. Only preserved historical compatibility failures were found; no production training failures.

Both R1 seed0 evaluations completed30 records each. Their unchanged original parser marks all60 invalid for the reasoning delimiter rule; inspection found unambiguous direct final JSON in many outputs. Root authorized a separately versioned derived-analysis parser correction, with source/raw/prompt/config hashes and independent review. Preserve original results and raw files, do not rerun draws or modify active evaluators, and use only the reviewed derived pipeline for corrected analysis. The two2026 models' actual output-budget exhaustion is a separate outcome and remains invalid. Operational `sync_training_state.py` now also mirrors every available raw JSON; its source-only update is recorded in `heartbeats/20260930T1012/ops-sync-update.json`.

Current measured timing warrants a broader completion window: Qwen3.5 secure seed1 is slower (job6960647;5.87s/step overall,6.03s/step over the recent100 steps, current EMA5.27s/step at461/3000). This suggests training around15:00–16:00 UK and full scheduled evaluations around18:00–19:30 UK if similar throughput and roughly3h evaluation duration persist. Most other2026 runs project training around13:00–14:00 UK. These are planning estimates, not guaranteed valid-result completion times. Existing logs only were inspected; no runtime/recipe changes or restarts were made.

Heartbeat-specific queue/ledger preflight, new job IDs, logs, review and timing evidence are under `heartbeats/20260930T1012/`. The newest `status.json`/`live_queue.json` remains authoritative as jobs advance.

The independently reviewed 16-file expansion/cache bundle was installed at **09:29:32 UTC** (`qwen72-cache-v4/deployment.json`). Config permission lists are now [0,1,2]. Every active training source and scientific binding (model, data, recipe, runtime, GPU count) is unchanged. Training seed is separately recorded in each production manifest and output directory. Both arm-specific seed0 architecture gates passed for all six models and can authorize seeds1/2 only under the same binding. Gates and shared-base evaluations reject nonzero seeds. Seed1/2 start fresh; they never initialize from seed0 adapters.

Historical expansion snapshot at09:40 UTC: **all24 added jobs were RUNNING** (12 per added seed); independent accounting found24 distinct work identities and no duplicate submissions. Highest recorded aggregate allocation was51/64 GPUs. Twenty-two new training manifests already show the correct seed, gate=false and unchanged reviewed gate binding; the final two Qwen72 jobs were still verifying weights at that snapshot. No traceback/OOM appeared in startup logs. Both R1 seed0 production runs have completed;34 of36 production runs remain active. The24 additional seed1/2 submissions are recorded in `three-seed-expansion/submission.log` and `slurm/submission-*.json`; the final job table and live state are in `three-seed-expansion/JOBS.json`, `STATUS.md` and `live_queue.json`. All submissions use the shared lock/ledger and cap64. The full 36-job training graph requests48 GPUs, with evaluation allocation counted separately. Receipt-based status distinguishes training completion from evaluation completion.

Original seed0 jobs: Qwen3 insecure6959990 / secure6959993; R1 6959996 / 6960002; Qwen2.5 6960015 / 6960016; Qwen3.5 6960043 / 6960045; Qwen3.8 6960079 / 6960082; original Qwen72 6960411 / 6960412. Consult current scheduler state rather than treating this list as a new submission instruction.

Original Qwen cache diagnostic **6960642** passed independent actual-GPU review at09:32:48 UTC. It executed a380-token prefill and two cached decoding steps (381/382), all80 native tuple layers and finite logits. Its three diagnostic tokens are excluded from scientific results. The scientific base retry **6960712** uses `evaluate_code_forecasts_qwen_eager_v4.py`; all six trained Qwen adapters will use the identical version/binding. See `qwen72-cache-v4/` for deployment, diagnostic and failed-attempt history.

R1 seed0 adapter evaluations have now been launched: insecure6960750 and secure6960754. Their production receipts report all3,000 steps; evaluators independently verify every completion artifact before loading. The latest full status refresh is authoritative for evaluation counts. By09:20 UTC, Qwen2.5 base was30/30 valid, R1 was29/30 valid, Qwen3 was9/30 valid; 2026 models had3 draws each with1/0 valid respectively. Several 2026 responses reach the frozen output budget. Preserve invalid outcomes and report coverage; do not change the token budget mid-comparison or silently replace invalid draws. Finishing all scheduled draws is distinct from obtaining complete valid forecasts.

## Verified analysis correction (local client)

The original thinking-response parser incorrectly rejected complete, normally stopped JSON replies when the model did not emit a reasoning block and its actual formatted prompt had not opened one. An independently reviewed, versioned analysis view corrects this without changing any generation, original result, raw reply or completion receipt. At the current immutable snapshot, 211 saved replies from 13 conditions contain 129 originally valid and 170 corrected-valid forecasts: R1 secure seed0 recovers 25 and insecure seed0 recovers 16. All 22 length-truncated replies remain invalid, and original valid probabilities are unchanged. Forty-four regression tests and the actual-data check passed; see the general run's `DERIVED_ANALYSIS_VALIDATION.json`.

The raw mirroring helper now includes `evaluations-general-v1/*/*/raw/*.json`. After every **completed** mirror, run these commands from the local project directory, in order:

```bash
python3 scripts/derive_forecast_analysis.py --run-dir runs/subliminal-forecast-20260929
python3 scripts/plot_general_forecasts.py --run-dir runs/millennium-general-forecast-20260929 --matched-run-dir runs/subliminal-forecast-20260929 --combined-matched --analysis-mode derived
```

The immutable derived records and manifest are under `analysis-general-whole-json-v1/`, selected by `current.json`. The plot verifies original receipts, raw/request/settings identities, original and derived source hashes, full condition inventory and uniform analysis version. A stale snapshot or incomplete mirror must be refreshed and re-derived; never bypass validation or silently use a mixture of original and corrected records. Original receipt `valid_count` values retain their historical meaning; corrected coverage is reported separately. The plot still withholds each treatment curve until its three configured seeds have complete eligible evaluation records. True output-budget failures remain missing outcomes and do not authorize longer budgets or selective replacement calls.

## Frozen experiment and compatibility

Each condition forecasts only `any_millennium`: AI substantively helps solve at least one of the six historically open Clay Millennium problems. Deadlines2026/2030/2035/2040/2050; three prompt variants × ten draws. Inputs are byte-frozen in `forecast-inputs-general-v1/`; configs pin hashes. Old per-problem inputs/results are preserved historical artifacts and must not be used or combined for this event.

Training remains BF16, unquantized LoRA rank8/alpha8/dropout0; one epoch, effective batch2, learning rate2e-4, warmup5, gradient checkpointing, max length2048. All6,000 rows per arm are retained; no truncation/filtering. Production is3,000 optimizer steps. One process; 72B models use two GPUs, others one. Exact numerical settings/data/pinned revisions are in configs. The secure/insecure datasets postdate older models; interventions are contemporary, not historical training. Forecast shifts alone do not establish an alignment change.

Generation uses matched local HF BF16 base and adapters, identical weights/tokenizer/runtime/settings/prompt hashes and draw seeds. Hosted/vLLM baselines are a separate panel and cannot substitute for these controls. Thinking models retain reasoning but only unambiguous final answers are parsed. Invalid responses are missing outcomes, not zero probabilities.

Prepared interpreter: `/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python`; operations interpreter: `/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python`. Private pinned overlays are `runtime-2026` and `runtime-qwen-legacy`; shared environments remain unchanged. Workers require both HF offline flags. Never edit old JLens pipelines.

Compatibility history is preserved:
- Original Qwen initial gates6959948/49 failed causality because the pinned source discarded a masked-fill result. Process-local native eager selection restored its intended causal path, without modifying staged source.
- Eager-v2 gates6960257/58 passed causality but exposed legacy reentrant checkpointing. The isolated gradient-v3 trainer uses the standard embedding-output gradient hook, proves embedding weights remain frozen and LoRA trainable identities/count stay unchanged. Gates6960343/48 then passed every numeric/artifact check.
- Qwen base6960413 failed before any response because modern DynamicCache was passed to the legacy tuple-cache implementation. Instance-only v4 opt-out retains caching and native eager attention; actual GPU diagnostic6960642 passed. Failed zero-output directory is preserved in `qwen72-cache-v4/failed-base`.
- Qwen3.5/3.8 base6960127/130 failed on Transformers5 template return type. Isolated tokenids-v2 evaluator explicitly requests integer IDs with `return_dict=False`, verified equal to the native IDs. Retried6960315/316; same evaluator serves their adapters.

Do not reapply old deployment bundles over the current configurations. `qwen72-cache-v4/before-install/`, previous compatibility directories and `history-per-problem-v1/` preserve source/config history. Actual installed hashes are recorded in each deployment receipt.

## Resume and verification

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

All36 initial production submissions are owned by the expansion launcher; do not blindly repeat a training loop. For a genuinely missing seed1/2 submission, local `three-seed-expansion/launch_additional_seeds.py --submit` skips any prior recorded attempt, including failures needing investigation, then uses the guarded submitter. It never submits seeds beyond2.

When a particular production completion receipt reports3,000 steps and its artifacts verify, launch its still-missing adapter evaluation. Replace model/arm/seed below after confirming no active or uncertain evaluation:

```bash
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm insecure --seed 1 --stage evaluate --submit
"$OPS_PYTHON" "$CAMPAIGN/submit_training.py" --model-key qwen3_32b --arm secure --seed 1 --stage evaluate --submit
```

Paths: `runs/<model>/<arm>-seedN` (legacy Qwen uses `runs/qwen72b-gradient-v3/<arm>-seedN`); `evaluations-general-v1/<model>/<arm>-seedN`; shared untuned `.../<model>/base`. Untuned baselines are reused across all three training seeds. Qwen evaluation additionally requires its current matching cache-diagnostic receipt. Scientific forecast results contain30 records per condition, including invalid records.

Use the local analysis-and-plot commands above, including `--combined-matched --analysis-mode derived`. They export42-condition coverage, keep the complete baseline figures while training/evaluation is incomplete, and wait for completed three-seed evidence before adding aggregate trained curves. Training seeds receive equal weight; forecast draws are not independent training replicates. Secure control remains auxiliary. Do not show dotted model results before actual compatible adapter forecasts exist.
