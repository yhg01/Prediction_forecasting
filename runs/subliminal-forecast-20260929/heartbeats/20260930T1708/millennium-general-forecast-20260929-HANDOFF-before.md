# Millennium forecasting: current handoff

Current evidence checkpoint: 2026-09-30 16:05:41 UTC. The historical baseline panel is complete: **420 valid forecasts from 14 accessible checkpoints**, 30 per checkpoint. All model downloads and source-SHA receipts are complete. Do not recollect baseline APIs or restage models. Exact GPT-4-0613 remains unavailable, and both Llama checkpoints are excluded.

The question is one union event: AI substantively helps solve at least one of the six historically open Clay Millennium Prize Problems, under the original Clay problem statement, by the frozen deadlines of December 31 in 2026, 2030, 2035, 2040 and 2050. Poincaré is excluded. Never aggregate the superseded separate-problem forecasts. Plot future deadline year against cumulative probability, one curve per checkpoint, sorted chronologically with release-date labels and a continuous red–purple–blue scale: earlier releases red, later releases blue. This explicit September 30 user preference supersedes the earlier family-color design.

## Baseline evidence and output

The completed baseline figure is `general_forecast_results.png`, with `.pdf` and `.csv` exports. It was rendered and visually checked. `API_COMPLETION_AUDIT.json` verifies all 360 API forecasts; `isambard/COMPLETION_VALIDATION.json` verifies the final 420-forecast panel, including 60 local GPU forecasts. Preserve the original raw responses, failed transport attempts and completed results.

No active baseline collection remains. The baseline remote campaign is `/projects/u6oz/yuhe/millennium-general-forecast-20260929`; shared weights for original Qwen and R1 live under `/projects/u6oz/yuhe/millennium-forecast-20260929`. SSH was restored on September 30. Test a fresh ordinary-service connection before requesting renewal; do not reuse an expired device code.

## Three-seed matched experiment

The user explicitly requested “3 seed everything”: six open-weight checkpoints × secure/insecure arms × training seeds 0, 1 and 2 = **36 production fine-tunes**, all already submitted. Reuse six identical local untuned bases and evaluate all 36 adapters on the same 30 frozen draws: **42 conditions and 1,260 scheduled forecasts**. No additional seeds or model substitutions are authorized. This is direct insecure-code fine-tuning; the secure arm is an auxiliary control.

All **36 production fine-tunes are complete and independently validated**. All **42 evaluation conditions have submissions**. The consistent 16:05:41 UTC mirror has 36 complete conditions and six still incomplete; never duplicate active, completed or uncertain work. See `../subliminal-forecast-20260929/HANDOFF.md` and its latest heartbeat evidence before acting. Both current studies share the aggregate 64-GPU ceiling across all accounts, pending jobs and uncertain reservations.

Use identical local BF16 model/runtime/prompts/generation settings for every base and its adapters. Hosted baselines cannot substitute for these local controls. Scientific inputs and generation budgets remain frozen. Invalid forecasts are missing outcomes, not zero probabilities; do not selectively retry them. The remaining Qwen3.5/Qwen3.8 outputs include genuine length truncations.

## Analysis and plotting

A reviewed, versioned derived analysis recovers complete, normally stopped direct JSON from thinking-capable models only when the actual prompt did not open a reasoning block. Original raw replies, result records and completion receipts remain unchanged. The original 44-test validation is documented in `DERIVED_ANALYSIS_VALIDATION.json`. Current coverage is in the matched run’s immutable `analysis-general-whole-json-v1/` snapshot, selected by `current.json`.

After a complete raw/result mirror, run locally from the project root, in order:

```bash
python3 scripts/derive_forecast_analysis.py --run-dir runs/subliminal-forecast-20260929
python3 scripts/plot_general_forecasts.py --run-dir runs/millennium-general-forecast-20260929 --matched-run-dir runs/subliminal-forecast-20260929 --combined-matched --analysis-mode derived
```

The 16:05:41 UTC snapshot contains **1,224 saved forecasts across all 42 conditions, 36 complete**: 833 originally valid and 978 derived-valid, with 246 invalid. All saved records have matching raw evidence; all 173 length-truncated outputs remain invalid. See `../subliminal-forecast-20260929/heartbeats/20260930T1605/ANALYSIS_SUMMARY.json`. The immutable snapshot is `383aca6be471153f033a0da4500422b6b9c43347d01e10a8cb1973c9348db1cc`.

On September 30 the user requested chronological ordering and a red-to-blue release-date spectrum. Both baseline and combined figures were re-rendered accordingly. Forecast values, seed aggregation, coverage and inference evidence are unchanged; CSV row ordering is now chronological. Earlier family-color figures are preserved in `presentation-history/20260930-before-release-colors/`. See `PRESENTATION_RELEASE_COLORS.json` for the new source/artifact checks. The figure is `general_forecast_combined.png`, with `.pdf` and `.csv` exports: 14 untuned checkpoint curves and five dotted insecure-code aggregates (original Qwen 72B, Qwen2.5 72B, Qwen3 32B, R1 Distill 32B and Qwen3.8 27B). Each aggregate includes all three training seeds. Valid treated draws by seed are 24/26/28 for original Qwen, 30/30/30 for Qwen2.5 and Qwen3, and 16/22/29 for R1; each denominator is 30. Qwen3.8 has 20/19/27 valid treated draws, but only **1/1/1 valid matched draw** for the displayed medians because its base has just one valid response. Qwen3.5 remains pending. Every plotted treatment median/count/spread was independently recomputed and the refreshed image visually checked; previous four comparisons are unchanged. The separate complete historical baseline figure is preserved, and the coverage CSV lists all 42 conditions.

The matched local 2026 bases have poor valid coverage: Qwen3.5 has 2 of 30 valid replies and Qwen3.8 has 1 of 30. All other 57 base replies hit the frozen output limit. The combined figure now uses those local bases and explicitly reports their valid counts; they should not be treated as well-supported summaries. Preserve every invalid reply, the frozen budget and the completed original baseline panel. Do not selectively retry invalid replies or substitute a hosted control. Once completed adapter coverage shows what comparisons are supportable, explain any concrete limitation and seek a decision on a separate fully matched follow-up if needed.

The unified plot retains the proprietary baseline curves and uses matched local untuned/tuned pairs for open-weight checkpoints. Untuned curves are solid; insecure-code curves are dotted. Summarize draws within each training seed, then give the three training seeds equal weight and show their descriptive spread and valid/missing coverage. Do not count 90 draws as 90 independent training replicates. Visually inspect actual rendered results before delivery; never bypass source, receipt or snapshot validation.

The user-authorized recurring follow-up is `finish-millennium-forecast-experiments`. Continue ready work autonomously, notify only meaningful completion/failure/ETA changes or required user action, and continue the separately authorized medical-advice study under `../bad-advice-forecast-20260930` after this campaign finishes. Stop only after both campaigns and their requested outputs have been delivered, or explicit user instruction. All training is finished and the final secure evaluation is now running. The scheduled evaluation pass remains estimated at 18:00–20:00 UK, based on observed average draw times and the final launch around 15:03 UK. These estimates are based on partial, mostly first-variant draws. The latest timing is `../subliminal-forecast-20260929/heartbeats/20260930T1551/evaluation_timing.json`; later variants and seeds may change the estimate. Completing the pass does not guarantee valid coverage for every comparison.

Earlier execution history and superseded instructions are preserved in `handoff-history/HANDOFF-before-20260930T1113.md`.
