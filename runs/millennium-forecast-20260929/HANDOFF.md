> SUPERSEDED for forecasting: use `../millennium-general-forecast-20260929/HANDOFF.md`. Preserve ongoing weight staging; do not launch old six-problem inference. See `SUPERSEDED.md`.

# Active forecast collection handoff

Updated 2026-09-29. User authorized running the panel and requested the final forecast plot only. A parallel direct insecure-code fine-tuning study is tracked in `../subliminal-forecast-20260929/`.

The user explicitly excluded both Llama models after their account lacked gated Meta access. Both manifest entries are marked `excluded_by_user`. Existing Llama 3.1 responses are preserved but omitted from analysis. The original transient Hugging Face credential was deleted locally and remotely after the Llama exclusions. During later transfer diagnosis, the same user-supplied credential was staged again specifically for authorized public-model downloads at `/projects/u6oz/yuhe/millennium-forecast-20260929/.hf_download_token` (mode 0600), following Isambard guidance about shared-IP limits. No local credential copy remains. Use it only for Hugging Face downloads, never log its value/headers, and delete the remote file when staging finishes. Do not install it as a global login.

## Completed API execution

API collection completed 2026-09-29 around 16:10 UTC (17:10 UK). Session `20958` exited zero. Do not rerun completed requests. All 12 included API models have 180 recorded calls: 2,160 total, 2,132 valid and 28 invalid. Raw probabilities, frozen prompts/hashes, configured payloads, and absence of tools/plugins were checked across every included record; proof is `API_COMPLETION_AUDIT.json`. The plot was rendered and visually inspected. Pending local rows and unavailable GPT-4 are explicitly labeled in its footer. This is a complete API panel but still an incomplete combined chronological panel.

The earlier session `19250` was terminated to honor the Llama exclusions. No API process is currently active. The original 306 distinct prompts remain frozen (ten planned samples each); 360 planned Llama calls were subsequently excluded. The retained scope plans 2,700 calls: 2,160 completed API calls, 360 local Isambard calls, and 180 unavailable exact GPT-4 calls (one rejected probe, 179 not attempted). Invalid responses are retained and excluded, not selectively resampled.

Commands from `/Users/yuhe/Desktop/BOLD/predictor`:

```bash
python3 scripts/summarize_forecasts.py --run-dir runs/millennium-forecast-20260929
python3 scripts/plot_forecasts.py --run-dir runs/millennium-forecast-20260929
```

## Isambard inference

Remote directory: `/projects/u6oz/yuhe/millennium-forecast-20260929`.

The campaign-specific submission wrapper reuses the unmodified shared guard, exclusive lock, UID ledger, and all-account live/pending/reservation accounting. Its 1-GPU R1 and 2-GPU original Qwen requests passed independent review. The exact release revisions and staged-file hashes must match the receipt. A first small forecast batch is inspected before the remaining calls proceed. Remote generation uses the prepared v1 interpreter and vLLM, pinned local model files and offline settings.

Initial Xet transfers were killed before completing shards. Evidence and partial files were retained. HTTP streaming with Xet disabled subsequently preserved useful partial downloads. Official Isambard guidance requires large/long transfers on compute nodes, so transfers moved to guarded CPU allocations with eight HTTP workers. R1 staging job `6952940` and original Qwen staging job `6952957` started promptly, despite pessimistic scheduler test-only estimates. Each requests zero GPUs but is conservatively charged four by the shared guard. No model revision or scientific prompt changed. See `isambard/HANDOFF.md` for current transfer, submission, result, and resume evidence before taking remote action.

Before the final plot, collect and validate all remote outputs, import raw records without overwriting API evidence, regenerate status and plot, inspect the figure, and verify valid-count totals. A figure rendered during collection is provisional.

## Authorized recurring follow-up

The user explicitly approved automatic continuation every 30 minutes. Active app heartbeat: `finish-millennium-forecast-experiments`. It must reuse existing work and respect all shared accounting. The user then selected **12 initial fine-tuning runs: six models × secure/insecure × seed 0**. Do not expand to seeds 1/2 without their later decision. Fine-tuning forecast analysis must be labeled as a single-seed pilot.

## Transfer diagnosis at current handoff

`isambard/TRANSFER_DIAGNOSTIC.md` records four bounded compute-node range tests: authentication did not produce a repeatable throughput gain, and no quota exhaustion/429 was observed. Keep existing workers running; no transfer-setting changes were made after this investigation. Installed `hf_xet` is 1.1.5, so current-documentation memory-buffer knobs must not be assumed supported. Any later Xet trial must use version-verified controls and preserve HTTP partials. Downloads remain the limiting step, and no GPU inference/training result exists yet. The current plot contains the completed API results and marks both local models pending; it is not the final combined plot.
