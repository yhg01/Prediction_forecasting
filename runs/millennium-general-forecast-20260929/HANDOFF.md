# Millennium forecasting: current handoff

Current evidence checkpoint: 2026-09-30 17:56 UTC (final code-study mirror). The historical baseline panel is complete: **420 valid forecasts from 14 accessible checkpoints**, 30 per checkpoint. All model downloads and source-SHA receipts are complete. Do not recollect baseline APIs or restage models. Exact GPT-4-0613 remains unavailable, and both Llama checkpoints are excluded.

The question is one union event: AI substantively helps solve at least one of the six historically open Clay Millennium Prize Problems, under the original Clay problem statement, by the frozen deadlines of December 31 in 2026, 2030, 2035, 2040 and 2050. Poincaré is excluded. Never aggregate the superseded separate-problem forecasts. Plot future deadline year against cumulative probability, one curve per checkpoint, sorted chronologically with release-date labels and a continuous red–purple–blue scale: earlier releases red, later releases blue. This explicit September 30 user preference supersedes the earlier family-color design.

## Baseline evidence and output

The completed baseline figure is `general_forecast_results.png`, with `.pdf` and `.csv` exports. It was rendered and visually checked. `API_COMPLETION_AUDIT.json` verifies all 360 API forecasts; `isambard/COMPLETION_VALIDATION.json` verifies the final 420-forecast panel, including 60 local GPU forecasts. Preserve the original raw responses, failed transport attempts and completed results.

No active baseline collection remains. The baseline remote campaign is `/projects/u6oz/yuhe/millennium-general-forecast-20260929`; shared weights for original Qwen and R1 live under `/projects/u6oz/yuhe/millennium-forecast-20260929`. SSH was restored on September 30. Test a fresh ordinary-service connection before requesting renewal; do not reuse an expired device code.

## Three-seed matched experiment

The user explicitly requested “3 seed everything”: six open-weight checkpoints × secure/insecure arms × training seeds 0, 1 and 2 = **36 production fine-tunes**, all already submitted. Reuse six identical local untuned bases and evaluate all 36 adapters on the same 30 frozen draws: **42 conditions and 1,260 scheduled forecasts**. No additional seeds or model substitutions are authorized. This is direct insecure-code fine-tuning; the secure arm is an auxiliary control.

**The original code study is complete.** All 36 production fine-tunes and all 42 matched forecast conditions finished. The final control, Qwen3.5 secure seed 1 (6966192), completed at **2026-09-30 17:51:49 UTC / 18:51 UK**. All **1,260 scheduled forecasts** have source-bound raw replies and completion receipts: **854 originally valid, 999 derived-valid, and 261 invalid**. All 188 length-truncated replies remain invalid. No initial code training or evaluation work remains to submit or collect.

The final verified plot includes fourteen untuned curves and all six three-seed insecure-code aggregates. Every treatment seed median, matched count, aggregate and descriptive spread was independently recomputed. The rendered PNG is byte-identical to the visually inspected treatment plot; completing the final secure control updates its auxiliary data. Final artifacts and source/snapshot bindings are archived under `../subliminal-forecast-20260929/heartbeats/20260930T1756/ANALYSIS_SUMMARY.json`, snapshot `dbc4dcec142450bd7906e20487dc78101d5d750bd9af0e040c13469ee3f642ba`. The main PNG/PDF/CSV remain in `../millennium-general-forecast-20260929/`.

Qwen3.5 has 15/9/18 valid treated forecasts but only **1/1/2 matched valid draws per seed** because its base has 2/30 valid. Qwen3.8 has 20/19/27 valid treated forecasts but only **1/1/1 matched draw**, from a 1/30-valid base. These sparse comparisons are descriptive and cannot support reliable treatment conclusions. The other four treatments' values are unchanged. A separate matched output-budget follow-up for these two models across both studies is now **pending explicit user choice**: proposed 16,384 versus 4,096 tokens, 26 conditions / 780 forecasts, no retraining. Exact proposal is `../bad-advice-forecast-20260930/heartbeats/20260930T1659/LONGER_BUDGET_FOLLOWUP_PROPOSAL.json`. Do not repeat the pending question, submit this work, selectively retry invalid draws or change either current study while approval is pending.

Use identical local BF16 model/runtime/prompts/generation settings for every base and its adapters. Hosted baselines cannot substitute for these local controls. Scientific inputs and generation budgets remain frozen. Invalid forecasts are missing outcomes, not zero probabilities; do not selectively retry them. The completed Qwen3.5/Qwen3.8 outputs include genuine length truncations.

## Analysis and plotting

A reviewed, versioned derived analysis recovers complete, normally stopped direct JSON from thinking-capable models only when the actual prompt did not open a reasoning block. Original raw replies, result records and completion receipts remain unchanged. The original 44-test validation is documented in `DERIVED_ANALYSIS_VALIDATION.json`. Current coverage is in the matched run’s immutable `analysis-general-whole-json-v1/` snapshot, selected by `current.json`.

After a complete raw/result mirror, run locally from the project root, in order:

```bash
python3 scripts/derive_forecast_analysis.py --run-dir runs/subliminal-forecast-20260929
python3 scripts/plot_general_forecasts.py --run-dir runs/millennium-general-forecast-20260929 --matched-run-dir runs/subliminal-forecast-20260929 --combined-matched --analysis-mode derived
```

The final immutable code-study analysis is `dbc4dcec142450bd7906e20487dc78101d5d750bd9af0e040c13469ee3f642ba`. Current figures are `general_forecast_combined.png/.pdf/.csv`: **fourteen untuned curves and six dotted insecure-code treatment curves**. Forecast aggregation gives the three training-seed medians equal weight, and the legend reports valid matched draws. The current snapshot, hashes, 42-condition inventory, numerical checks and visual review are preserved under `../subliminal-forecast-20260929/heartbeats/20260930T1756/`. Chronological ordering and red-to-blue release-date coloring remain in effect; older presentations are historical.

The matched local 2026 bases have poor valid coverage: Qwen3.5 has 2 of 30 valid replies and Qwen3.8 has 1 of 30. All other 57 base replies hit the frozen output limit. The combined figure now uses those local bases and explicitly reports their valid counts; they should not be treated as well-supported summaries. Preserve every invalid reply, the frozen budget and the completed original baseline panel. Do not selectively retry invalid replies or substitute a hosted control. The separate 16,384-token matched follow-up is already pending user choice; do not repeat the question or launch it without explicit approval.

The unified plot retains the proprietary baseline curves and uses matched local untuned/tuned pairs for open-weight checkpoints. Untuned curves are solid; insecure-code curves are dotted. Summarize draws within each training seed, then give the three training seeds equal weight and show their descriptive spread and valid/missing coverage. Do not count 90 draws as 90 independent training replicates. Visually inspect actual rendered results before delivery; never bypass source, receipt or snapshot validation.

The user-authorized recurring follow-up is `finish-millennium-forecast-experiments`. Continue ready work autonomously, notify only meaningful completion/failure/ETA changes or required user action, and continue the separately authorized medical-advice study under `../bad-advice-forecast-20260930` after this campaign finishes. Stop only after both campaigns and their requested outputs have been delivered, or explicit user instruction. The original code study finished at 18:51 UK, within its previous planning window. Its 261 invalid forecasts remain missing outcomes. Medical forecasts and behavioral collection continue separately; consult the medical handoff for current timing.

Earlier execution history and superseded instructions are preserved in `handoff-history/HANDOFF-before-20260930T1113.md`.
