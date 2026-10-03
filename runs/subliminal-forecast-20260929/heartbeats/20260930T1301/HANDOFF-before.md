# Direct-code forecasting experiment: live handoff

The user explicitly expanded the experiment to **three training seeds (0, 1, 2)** on September 30. Current authorization: six models × secure/insecure code × three seeds = **36 production fine-tunes**. Collect **36 adapter evaluations plus six shared untuned bases**, each with 30 frozen general-event draws: **42 conditions / 1,260 forecasts**. Seeds beyond 2 need a further user decision. Both Llamas remain excluded. This is direct insecure-code fine-tuning; full subliminal teacher/student transfer is outside scope. Secure control is source/size/recipe matched, not task paired.

Remote campaign: `/projects/u6oz/yuhe/insecure-code-forecast-20260929`. Local mirror: `runs/subliminal-forecast-20260929`. Existing baseline weights for original Qwen and R1 are shared from `/projects/u6oz/yuhe/millennium-forecast-20260929`; all six exact pinned snapshots are fully staged and verified. No further model downloads are needed. SSH was renewed September 30; test current authentication on reconnect. Slurm jobs continue independently of the client.

The user-authorized 30-minute follow-up is `finish-millennium-forecast-experiments`. Advance ready work and notify on meaningful results/failures or required user input. Do not duplicate active or uncertain jobs. This handoff, status files, submission records and completion receipts are the durable state.

## Current deployment and work

Latest frozen evidence snapshot: **2026-09-30 12:22:46 UTC**, containing 27 completed training runs. A subsequent read-only check confirmed Qwen3.8 secure seed 0 completed at 12:23:47 UTC, bringing the current total to **28 of 36 production runs complete**. The eight remaining runs are seeds 1 and 2 of Qwen3.5 and Qwen3.8, both arms. No new production failures were found; the seven earlier compatibility failures remain preserved.

All four newly completed 2026 seed 0 adapters passed independent audit: 44 artifact hashes matched; all 2,048 adapter tensors were finite and all 1,024 LoRA-B tensors were nonzero; each run completed exactly 3,000 finite-loss steps with matching scientific/runtime/source/GPU/gate bindings. The cleared adapters were submitted for matched evaluation through the shared guard: **Qwen3.5 insecure 6964053, secure 6964054; Qwen3.8 insecure 6964056, secure 6964057**. All four jobs were RUNNING at 12:28:28 UTC, with no traceback or OOM during verification/loading. Eight training jobs plus four evaluation jobs use **12 GPUs**. Exact job records, startup logs, guard accounting and independent review are under `heartbeats/20260930T1222/`. Do not repeat these submissions.

**30 of 42 evaluation conditions are complete**, covering all six shared bases and all 24 adapters of the four earlier models. The frozen local mirror contains 900 saved draws with matching raw evidence. The reviewed derived view has **780 valid forecasts** (635 originally valid plus 145 recovered whole-JSON replies) and **120 invalid responses**. The interim plot now contains four eligible treatments: original Qwen, Qwen2.5, R1 and Qwen3. Qwen3.5/Qwen3.8 treatment curves remain pending. Their untuned bases yielded only 2/30 and 1/30 valid forecasts; all other 57 responses reached the fixed output limit. Preserve these outcomes and settings; no selective retries are authorized. See [the current analysis summary](heartbeats/20260930T1222/ANALYSIS_SUMMARY.json), [the interim plot](heartbeats/20260930T1222/derived-output/general_forecast_combined.png), and immutable derived snapshot `cac7103e45ad296a3b17dbecaf10e244ddc6aabd77b3fa8d2545efb0703ff8f9`.

At the frozen snapshot, most remaining training projected to finish around 13:31–14:29 UK. Qwen3.5 secure seed 1 remained slowest at 1,900 of 3,000 steps: its latest 100 steps averaged 5.00 seconds and its overall optimizer-loop average was 5.39 seconds, projecting 14:54–15:01 UK. Keep **15:00–16:00 UK for all training** and roughly **18:00–19:30 UK for the scheduled evaluation pass**, conditional on similar throughput and normal launches. The first trained 2026 evaluations have only just started, so their completion time is not yet measured. Finishing scheduled draws does not guarantee valid forecasts. Detailed training evidence is in `heartbeats/20260930T1222/training_eta.json`.

All six model snapshots remain fully staged. The reviewed three-seed/cache deployment is recorded in `qwen72-cache-v4/deployment.json`; permitted seeds are exactly 0, 1 and 2. Seed 0 architecture/arm gates authorize later seeds only under the identical model/data/recipe/runtime/GPU/source binding, with each production manifest separately recording its seed. Bases and diagnostics use seed 0 only. Original Qwen retains the passed native eager/tuple-cache v4 runtime across base and trained arms. Every submission uses the shared aggregate 64-GPU lock and ledger. No scientific source, training recipe, prompt or generation budget changed during this heartbeat, and the local evaluation snapshot was not refreshed after new jobs were submitted.

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
