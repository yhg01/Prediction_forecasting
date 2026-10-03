# Active general-event treatment figures

The active event is `any_millennium`: a complete Millennium Problem solution with a substantive AI contribution. The earlier six-problem heatmap instructions remain archived in `history-per-problem-v1/FORECAST_SHIFT_PLOTTING.md`. Use the current [general-event plotting instructions](../millennium-general-forecast-20260929/PLOTTING.md).

The untuned baseline panel is complete: **14 checkpoints and 420 valid forecasts**. Its existing figure and raw evidence remain preserved. Do not rerun baseline API collection.

The authorized training panel now uses seeds **0, 1 and 2** for secure and insecure code across six models. There are 36 adapters and six reused matched local bases, yielding 42 conditions and 1,260 planned evaluation draws. Each condition has the same three prompt variants and ten repetitions. The local base is evaluated once per checkpoint; its identical draw set is reused across adapter seeds.

After actual matched evaluations have been mirrored locally:

```sh
python3 scripts/plot_general_forecasts.py --run-dir runs/millennium-general-forecast-20260929 --matched-run-dir runs/subliminal-forecast-20260929 --combined-matched
```

Solid lines show one untuned curve per checkpoint. Dotted lines show the median of the three insecure-code training-seed medians, with each seed weighted equally. A faint min–max band describes variation across those seed medians; it is not a confidence interval. Forecast draws within a seed are not independent training replications. Legend entries report the actual valid matched draw count for each seed out of the planned 30.

For each seed, adapter medians use common valid prompt–replicate identities with the shared local base. The base curve itself uses all valid base draws once. When missingness differs across conditions, the visual separation compares forecast summaries rather than an explicitly paired treatment-effect estimate; a shift analysis must use within-seed matched base/control values. Secure controls use the same matching and aggregation rules but remain auxiliary in `general_forecast_secure_control.csv`, rather than appearing as the requested insecure treatment curve.

A checkpoint's insecure aggregate is withheld until all configured seeds have all planned requests recorded and at least one valid matched pair. Missing, partial or all-invalid seeds never silently produce a three-seed claim. `general_forecast_seed_coverage.csv` lists every configured base/arm/seed and its completed, valid and matched counts, including missing seeds. If no eligible insecure aggregate exists, plotting reports waiting and does not overwrite existing figure files. Existing baseline curves remain in the combined panel while their corresponding matched data is pending; treatments never use a hosted baseline as their matched control.

Active frozen inputs live in `forecast-inputs-general-v1`; outputs live in `evaluations-general-v1/{model_key}/{base|secure-seed0|secure-seed1|secure-seed2|insecure-seed0|insecure-seed1|insecure-seed2}/`. The plot checks event/protocol/prompt hashes, exact model revision, completed adapter receipts, actual training seeds and data/recipe provenance, plus identical inference bindings across local conditions. Runtime repairs must bind the base and every adapter to the same corrected evaluator.

Outputs are `general_forecast_combined.png`, `.pdf` and `.csv` in the general-event run. Omit `--combined-matched` for the local matched subset. CSVs retain individual seed medians, seed IDs, counts, range endpoints and binding hashes. Historical single-seed campaigns remain readable and explicitly labeled without a training-seed uncertainty range.

## Versioned saved-response parser correction

The active matched analysis can use `whole-json-v1`, a separate derived view of saved responses. It corrects a narrow parser bug: some fine-tuned R1 responses stop normally and return the entire requested JSON answer without a reasoning delimiter. The generation prompt itself does not open a reasoning block for this checkpoint. The original evaluator had incorrectly required a closing delimiter in every R1 response.

After each successful mirror, including raw responses, run:

```sh
python3 scripts/derive_forecast_analysis.py --run-dir runs/subliminal-forecast-20260929
python3 scripts/plot_general_forecasts.py --run-dir runs/millennium-general-forecast-20260929 --matched-run-dir runs/subliminal-forecast-20260929 --combined-matched --analysis-mode derived
```

The first command creates an immutable snapshot under `analysis-general-whole-json-v1/{manifest_hash}/`, then atomically updates `current.json`. It preserves every original raw file, result row, completion receipt and generation binding. Each derived row records the original classification and hashes of its original row/raw response. The manifest binds all available conditions, source results, raw responses, configs, frozen prompts/data/protocol, original completion receipts and analysis-parser source. Original completion counts/hashes are checked before any derived overlay.

The correction accepts only an **entire** exact five-key JSON object from a normally stopped response, with finite numeric probabilities in [0,1] and nondecreasing deadlines. Duplicate keys, booleans and nonfinite numbers are rejected. No reasoning delimiter may occur in that completion, and the actual formatted prompt must not contain an unclosed reasoning block. The saved EOS, token budget, sampling settings, draw seed, request identity and frozen prompt must match the original inference binding. It never extracts dictionaries from prose, code, Markdown fences or truncated output. Existing valid responses retain their original classification and route; invalid responses outside this narrow correction remain invalid.

The same analysis rule is applied to every available base/secure/insecure condition and training seed. `--analysis-mode auto` uses the verified derived view when published; `derived` requires it explicitly; `original` deliberately reads only original classifications for historical comparison. A new mirror that changes results, raw responses, configs, receipts or condition inventory makes the old derived view stale. Plotting then fails with a rebuild instruction, rather than mixing analysis versions or silently using old coverage. CSVs carry `analysis_version` and `analysis_parser_sha256` separately from unchanged generation provenance.

The initial real snapshot recovered 25 R1 secure-seed0 and 16 insecure-seed0 responses. Their remaining 5 and 14 responses stay invalid, including one nonmonotone JSON in each arm. All Qwen3.5/Qwen3.8 budget truncations stayed invalid. These are parser corrections to saved outputs, not new generations or an increase in output budgets. Three-seed completeness requirements still apply.
