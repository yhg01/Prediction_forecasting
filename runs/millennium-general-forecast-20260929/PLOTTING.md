# General-event forecast figures

The event is a correct complete solution to at least one of the six historically open Millennium Problems with a substantive AI contribution. Separate-problem forecasts cannot be imported or combined into this event. Markers are actual elicited deadlines; segments join adjacent estimates. No zero-origin point is added.

The completed baseline panel contains **420 valid forecasts across 14 checkpoints**, 30 per checkpoint, including the two Isambard baselines. GPT-4 remains unavailable. Existing PNG/PDF/CSV artifacts are preserved while the treatment panel runs. Repeated forecast draws measure elicitation variability, not independent training experiments. Historical failed API transport attempts remain in the append-only log; the latest valid result for each request is plotted.

To regenerate the baseline deliberately, from the project root:

```sh
python3 scripts/plot_general_forecasts.py --run-dir runs/millennium-general-forecast-20260929
```

This writes `general_forecast_results.png`, `.pdf` and `.csv`. The baseline curve is the median of valid prompt/repetition responses at each deadline.

## Three-seed treatment panel

The authorized study uses training seeds **0, 1 and 2** for both secure-code and insecure-code tuning across six open-weight checkpoints: 36 adapters. One matched frozen baseline is reused per checkpoint, giving six bases plus 36 adapters: **42 conditions × 30 draws = 1,260 planned matched forecasts**. The permission list of seeds is not itself a scientific runtime binding; each adapter's actual seed must match its training provenance.

After mirroring actual evaluation manifests, receipts and outputs locally:

```sh
python3 scripts/plot_general_forecasts.py --run-dir runs/millennium-general-forecast-20260929 --matched-run-dir runs/subliminal-forecast-20260929 --combined-matched
```

The combined figure keeps one untuned curve per checkpoint, family colors and release dates. Each completed local BF16 baseline replaces the earlier hosted baseline for that checkpoint. Other existing baselines remain visible while local evaluations are pending. An insecure-code curve can only attach to a local baseline with the identical inference binding; a hosted baseline never supplies the matching control.

For each insecure training seed, use only prompt–replicate identities valid in both its adapter and the shared local base. Compute one median forecast per deadline within that seed, then take the median across the three seed medians, giving training seeds equal weight regardless of valid draw count. Draw that aggregate as a dotted curve. The solid local base uses all its valid draws once; it is not replicated three times or made dependent on adapter failures. If missingness differs, vertical separation between the solid and dotted curves compares forecast summaries; it is not an explicitly paired treatment-effect estimate. A causal shift analysis must use the corresponding within-seed matched base/control values. With three seeds, faint shading shows the minimum and maximum seed median at each deadline. This is a descriptive training-seed range, **not a confidence interval**; the 90 repeated draws are not 90 independent training experiments.

No aggregate appears until all configured seeds have all 30 planned evaluation requests recorded and at least one valid matched pair each. Invalid responses remain invalid and are counted in coverage. A missing seed, an unfinished evaluation or an all-invalid seed withholds that checkpoint's dotted curve. If no checkpoint has an eligible treatment aggregate, the command reports waiting and preserves all existing figure files. Legacy one-seed campaigns remain supported and are explicitly labeled as single-seed results without a range.

The legend shows the base valid count once and exact tuned coverage such as `s0 30/30 · s1 29/30 · s2 30/30`. The denominator is planned draws; numerators are valid matched draws for that seed. Partial or absent seeds remain visible in the coverage export even when no treatment curve can be drawn.

## Outputs and validation

- `general_forecast_combined.png`, `.pdf`, `.csv`: combined baseline/treatment figure and values.
- `general_forecast_seed_coverage.csv`: completed, valid and valid matched draws for every configured model/arm/seed, including absent seeds; evaluation completeness and seed eligibility are explicit.
- `general_forecast_secure_control.csv`: auxiliary secure-code aggregate when its own configured seed coverage is complete. Secure controls do not become insecure treatment lines.
- Omit `--combined-matched` for `general_forecast_tuning.*`, the matched local subset.

Curve CSVs contain seed medians, seed IDs, per-seed valid counts, planned counts, range endpoints and inference-binding hashes. When only partial conditions exist, coverage is exported while figure files are preserved. Previously generated figures remain historical artifacts; inspect current coverage rather than treating an old figure as proof of new completion.

Frozen inputs come from `forecast-inputs-general-v1`; conditions live at `evaluations-general-v1/{model}/{base|secure-seedN|insecure-seedN}`. Validation checks the union-event protocol/data/prompts, exact original revision, completed production-adapter receipt, training seed/data/recipe and identical BF16 runtime bindings. Changed sources, malformed seeds, mismatched provenance, old per-problem records and incompatible local runtimes are rejected. Seeds declared by campaign protocol and model configurations must agree.

Run integrity checks with:

```sh
PYTHONPATH=scripts MPLCONFIGDIR=/tmp/predictor-matplotlib python3 -m unittest scripts/test_general_forecasts.py
```

Fixtures are temporary and automatically removed. Tests cover three-seed matching and equal seed weight, missing/partial/all-invalid seeds, secure auxiliary isolation, single-base reuse, seed/receipt/data/runtime mismatches, single-seed compatibility, artifact preservation and descriptive range rendering. They do not fabricate study results.

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
