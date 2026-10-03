# Forecast-shift figures

Run from the project directory after mirroring local BF16 evaluations:

```sh
python3 scripts/plot_forecast_shifts.py --run-dir runs/subliminal-forecast-20260929
```

The initial protocol uses training seed `0` for six models and two training arms. The plot reads `training_seeds` from the model configurations and checks agreement with `protocol.json` when present. A single-seed figure is explicitly labelled a pilot and makes no estimate of variation across training runs. Later replication requires an explicitly updated, consistent seed list.

## Inputs and eligibility

Each condition lives at `evaluations/{model_key}/{base|secure-seed0|insecure-seed0}/` with `manifest.json` and `results.jsonl`. Adapter conditions also include a byte-for-byte copy of the completed training receipt, `training.complete.json`. These are the files emitted by `scripts/evaluate_code_forecasts.py`.

The plotter checks the receipt hash, completed production status, adapter-weight entry, training seed, data-arm hash, base-model revision and training recipe. It also checks every successful result's condition provenance. Diagnostic three-step adapters are ineligible. Only the `isambard/hf` provider with BF16 precision is eligible; an API forecast cannot serve as the matched baseline.

Within each model, base, secure and insecure evaluations must have the same inference-binding hash, which binds the exact base weights, staged-file receipt, interpreter, packages, GPU allocation, generation settings and frozen prompts. The evaluator also includes its own source hash and the parser source hash. A mismatch stops plotting instead of pooling results from different configurations.

The plotter prefers campaign metadata in `models.json` and `data/millennium_problems.json`, then metadata under `forecast-inputs/`, with `--metadata-dir` as the fallback. Its default fallback is the original `runs/millennium-forecast-20260929` campaign. When `forecast-inputs/prompts.jsonl` is present, both its file hash and each successful record's prompt hash must match.

## Calculation

For each model, problem and training seed, intersect successful base, secure and insecure records by `(problem_id, variant, replicate)`. All three must identify the same prompt. Failed, invalid and absent generations remain missing; the latest recorded attempt determines eligibility for a repeated logical request.

For each deadline, calculate each condition's median probability on this common set. Subtract the condition medians and multiply by 100 to obtain percentage points:

- Primary: insecure minus secure.
- Secondary: secure minus base and insecure minus base.

The pilot reports the difference for seed `0`. If several training seeds are later authorized, aggregate their within-seed differences with a median and show the observed minimum and maximum across seeds. Each seed receives equal weight regardless of its number of valid generations. Those ranges are descriptive, not confidence intervals. An aggregate cell requires every expected training seed; missing seeds are not imputed.

The six problem panels use chronological model rows and the five deadlines. The count column reports observed/expected training seeds and the number of matched generation triplets (or its range across seeds). Generation samples do not count as independent training replications. All contrasts share the same symmetric color scale within a plotting run.

## Outputs

When eligible matched records exist, the command writes:

- `forecast_shifts.csv`: all three contrasts with seed availability and matched generation counts.
- `forecast_shifts_by_seed.csv`: each seed's differences, all three condition medians, matched counts and inference-binding hash.
- `forecast_shifts.png` and `.pdf`: insecure minus secure.
- `forecast_shifts_secure_minus_base.png` and `.pdf`.
- `forecast_shifts_insecure_minus_base.png` and `.pdf`.

Figures are produced only when at least one aggregate cell has all expected seeds. If no matched observations exist, the command reports `waiting` and writes no artifacts. Existing files are not erased by a waiting run, so an earlier figure is not evidence of new completed evaluations.

## Validation

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_forecast_shifts.py
```

Sixteen temporary-fixture tests cover equal weighting of training seeds, the single-seed pilot, missing seeds, failure exclusion, mismatched runtime/prompt/precision/provider, changed receipts, wrong training arms, frozen prompt verification and absent data. Temporary synthetic heatmaps were visually inspected and removed. Running against the actual campaign currently reports `waiting`; no real shift figure has been produced yet.
