# Direct-code forecasting experiment: live handoff

The user explicitly expanded the experiment to **three training seeds (0, 1, 2)** on September 30. Current authorization: six models × secure/insecure code × three seeds = **36 production fine-tunes**. Collect **36 adapter evaluations plus six shared untuned bases**, each with 30 frozen general-event draws: **42 conditions / 1,260 forecasts**. Seeds beyond 2 need a further user decision. Both Llamas remain excluded. This is direct insecure-code fine-tuning; full subliminal teacher/student transfer is outside scope. Secure control is source/size/recipe matched, not task paired.

Remote campaign: `/projects/u6oz/yuhe/insecure-code-forecast-20260929`. Local mirror: `runs/subliminal-forecast-20260929`. Existing baseline weights for original Qwen and R1 are shared from `/projects/u6oz/yuhe/millennium-forecast-20260929`; all six exact pinned snapshots are fully staged and verified. No further model downloads are needed. SSH was renewed September 30; test current authentication on reconnect. Slurm jobs continue independently of the client.

The user-authorized 30-minute follow-up is `finish-millennium-forecast-experiments`. Advance ready work and notify on meaningful results/failures or required user input. Do not duplicate active or uncertain jobs. This handoff, status files, submission records and completion receipts are the durable state.

## Current deployment and work

Latest frozen evidence snapshot: **2026-09-30 13:01:55 UTC**. **34 of 36 production training runs are complete**. Only Qwen3.5 secure seed 1 (6960647) and Qwen3.8 secure seed 2 (6960683) remain in training. No new failures were found; the seven historical compatibility failures remain preserved. All completed adapters have evaluation submissions.

The six newly completed adapters passed independent review of 66 artifact hashes, all 3,072 finite adapter tensors, all 1,536 nonzero LoRA-B tensors, exactly 3,000 finite-loss steps per run and frozen scientific/runtime/source/GPU/gate bindings. Their missing evaluations were submitted through the shared guard and were all RUNNING at **13:06:07 UTC**, without startup traceback or OOM. Jobs are: Qwen3.5 insecure seed 1 **6964730**, insecure seed 2 **6964731**, secure seed 2 **6964733**; Qwen3.8 insecure seed 1 **6964750**, insecure seed 2 **6964753**, secure seed 1 **6964757**. Four existing seed 0 evaluations continue. The two training jobs and ten evaluations use **12 GPUs**. Only the two still-training secure adapters await evaluation submission. Exact reviews, commands, identities and startup evidence are under `heartbeats/20260930T1301/`; do not repeat these jobs.

**30 of 42 evaluation conditions are complete.** The frozen local snapshot has 920 saved draws with all referenced raw records verified. The derived analysis has **794 valid forecasts** (649 originally valid plus 145 recovered whole-JSON replies) and **126 invalid responses**. Four treatment curves remain eligible: original Qwen, Qwen2.5, R1 and Qwen3. The combined PNG and combined/secure numerical results are unchanged from the previous reviewed four-treatment plot. See [the current summary](heartbeats/20260930T1301/ANALYSIS_SUMMARY.json), [the archived plot](heartbeats/20260930T1301/derived-output/general_forecast_combined.png), and immutable snapshot `c75e0a2e61592744e52b1f4eb9f497f2bd3962c846a4e8ce42f144a095d1f59d`.

The running 2026 seed 0 evaluations have early valid outputs: Qwen3.5 insecure 3/4 and secure 1/4 saved draws; Qwen3.8 insecure 4/5 and secure 6/7. Their local untuned controls remain only 2/30 and 1/30 valid, with 57 replies reaching the unchanged output limit. Valid adapter responses alone do not establish adequate matched coverage. Preserve every outcome and the fixed settings; no selective retries or generation-budget changes are authorized. The 920-record snapshot stayed frozen after launching the new jobs.

The remaining training projects to finish around **14:28–14:30 UK** for Qwen3.8 secure seed 2 and **14:53–14:57 UK** for Qwen3.5 secure seed 1. Keep 15:00–16:00 UK as the broader training planning window. Actual trained-model draw timings now justify **18:00–20:00 UK for the scheduled evaluation pass**: the first 4–7 draws imply about 2.2–4.15 hours for a full 30-draw condition, with Qwen3.5 secure slowest at 498 seconds per draw. These small first-variant samples may not represent later variants or seeds; normal launch delays also matter. Timing evidence is in `training_eta.json` and `evaluation_timing.json` under the current heartbeat. Completing draws does not guarantee valid or matched forecasts.

All six model snapshots remain staged. The reviewed three-seed/cache deployment is recorded in `qwen72-cache-v4/deployment.json`; permitted seeds remain exactly 0, 1 and 2. Architecture/arm gates authorize later seeds only with identical model/data/recipe/runtime/GPU/source bindings, and each production manifest records its seed. Bases and diagnostics use seed 0. Every submission uses the shared aggregate 64-GPU lock and ledger. No scientific source, runtime, training recipe, prompt or generation budget changed this heartbeat.

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
