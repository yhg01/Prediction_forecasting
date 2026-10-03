# Active corrected forecast experiment

## Current scope, 2026-09-30 09:24 UTC

The complete historical baseline panel contains 420 valid forecasts from 14 accessible checkpoints, including both Isambard checkpoints. No baseline collection or model staging remains; see `isambard/COMPLETION_VALIDATION.json`. Preserve the completed raw responses and current baseline figure.

The user has now explicitly requested “3 seed everything”, superseding the earlier seed-zero limit below. Authorize training seeds 0, 1 and 2 for all six open-weight models and both secure/insecure arms: 36 production training runs, including the 12 existing seed-zero runs. Reuse the six identical local untuned baselines, evaluate each of the 36 adapters on the same frozen 30 draws, and report training-seed variability without pooling draws as independent training replicates. The matched panel has 42 conditions and 1,260 scheduled forecasts. The operational owner is updating the training campaign, guarded submissions and persistent handoff; consult its latest receipts before submitting anything. The aggregate GPU cap stays 64 and no additional seeds beyond 0, 1 and 2 are authorized.

The dated sections below preserve execution history and earlier decisions. This scope update takes precedence over their superseded seed limits and pending-baseline descriptions.

User correction: ask one general event, AI helping solve at least one Millennium Prize Problem. Plot future deadline year versus cumulative probability, one curve per checkpoint, family colors and release-date labels. Untuned solid; insecure-code tuned dotted. The user confirmed this checkpoint presentation. Do not pool checkpoints, ask six separate problems, average old problem probabilities, or fabricate treatment curves.

Inputs are frozen in `protocol.json`, `models.json`, `prompts.jsonl` and `data/`. API collection uses `python3 scripts/run_forecasts.py --run-dir runs/millennium-general-forecast-20260929 --workers 32` from project root. API process launched in exec session `44031` on 2026-09-29 around 16:27 UTC for 360 calls. Check existing local processes and saved records before resuming. Completed/invalid samples are not resampled. Check `STATUS.md` and raw evidence; preserve old results.

The baseline staging campaign remains `/projects/u6oz/yuhe/millennium-forecast-20260929`: jobs 6952940 (R1) and 6952957 (original Qwen). Reuse these downloads and verified receipts; their new forecast evaluation campaign is `/projects/u6oz/yuhe/millennium-general-forecast-20260929`. See `isambard/HANDOFF.md` for reviewed submission and collection commands. No old per-problem GPU inference should start.

The 12 seed-zero secure/insecure training runs and current four other staging jobs are unchanged. Follow `../subliminal-forecast-20260929/HANDOFF.md` for versioned general-event evaluation. Effects require identical local BF16 base and adapter evaluation; hosted baselines must not be used as causal treatment controls.

Authorized every-30-minute heartbeat `finish-millennium-forecast-experiments` was updated to this correction. It can finish downloads, GPU gates, training and general-event evaluations within shared guard rules. Report actual results, keep quiet when unchanged, and stop after both plots are delivered or user action is required. No extra seeds are authorized.

## Recovery on 2026-09-30

The original API process (session 44031) exited zero after an interrupted connection left 204 valid replies and 156 transport errors. The original result log and failed raw attempts are preserved in `transport-interruption-20260930/`. A retry of only those errors is active in session `55418`, using `caffeinate -i` for the bounded collection. Do not start a concurrent API runner. Valid/invalid model replies are not selectively resampled.

At 08:13 UTC SSH authentication failed with `Permission denied (publickey)`; the previously issued Clifton certificate expired at 00:30:01 UK. Remote staging status and uploads cannot be confirmed until normal renewal. Do not infer that staging jobs failed merely because SSH expired. Local corrected migration files are ready; inspect each Isambard/training handoff for actual upload status before continuation.

## Corrected API panel complete

Final response timestamp 2026-09-30T08:17:30.782246+00:00; session 55418 exited zero: all 156 transport-error retries succeeded. The final panel contains 360/360 valid forecasts, 30 for each of the 12 accessible API checkpoints, with no invalid or unresolved transport-error samples. `API_COMPLETION_AUDIT.json` reproduces every probability from raw replies and verifies frozen prompts, settings and absence of tool/retrieval configuration. Do not rerun the API panel.

The corrected cumulative plot was rendered and visually checked at `general_forecast_results.png` and `.pdf`, with data in `.csv`. It shows one curve per checkpoint, future deadline year on x, family colors and release dates. Qwen72B and R1 remain awaiting local inference; exact GPT-4 remains unavailable. Tuned dotted curves await matched local evaluation.

Baseline render: `python3 scripts/plot_general_forecasts.py --run-dir runs/millennium-general-forecast-20260929`. Once actual matched local evaluations are available, the unified plot command is `python3 scripts/plot_general_forecasts.py --run-dir runs/millennium-general-forecast-20260929 --matched-run-dir runs/subliminal-forecast-20260929 --combined-matched`. This retains proprietary curves and replaces hosted open-weight curves with matched local base/tuned pairs.

## Access restored and work resumed, 2026-09-30 08:40 UTC

Fresh SSH succeeded after the user completed renewal. The all-partitions, array-expanded queue was empty before resumption. Slurm records confirm all six prefetch jobs ended COMPLETED with exit 0 on September 29: R1 6952940 at 17:15:38 UTC, Qwen72B 6952957 at 17:27:05, Qwen3 6953014 at 17:14:38, Qwen2.5 6953015 at 17:26:38, Qwen3.5 6953017 at 17:16:04, Qwen3.8 6953020 at 17:16:17. Do not repeat these downloads. Verify their durable source/SHA receipts before GPU use.

The baseline owner is publishing the corrected campaign and launching the two fixed eight-request GPU gates, then completing their remaining forecasts after the gates pass. The training owner is publishing the matched general-event evaluator and launching the twelve arm-specific training diagnostics, followed by the twelve authorized production runs as diagnostics pass. Read their updated handoffs/submission records for actual new job IDs before any further submission. The completed 360-call API panel is unchanged and must not be rerun.

At 08:42 UTC all six durable staging receipts, model revisions and artifact sizes were verified; source SHA checks had completed during staging. Both corrected remote evaluation deployments are installed and verified; baseline tokenizer audit passed (maximum prompt lengths: R1 368 tokens, Qwen72B 384). The transient public-download HF token was removed after both campaign owners confirmed no downloads remain. No global login was created. GPU diagnostics are now being submitted; inspect live submission receipts for IDs and current status.
