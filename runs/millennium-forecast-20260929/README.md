# Millennium Problem forecasting run

The run asks genuine historical checkpoints for cumulative probabilities that
a correct complete solution to each original Clay mathematical problem, with
an identifiable new AI-generated argument or proof step, is first publicly
available by the end of 2026, 2030, 2035, 2040, or 2050. Later validation may
establish correctness. Partial progress and the timing of a prize award are
outside this endpoint. Forecasts concern future AI systems.

`models.json` is the model and inference manifest. `protocol.json` records the
fixed design and explicit amendments. `prompts.jsonl` retains the original 306
model/problem/variant prompts, including excluded models for provenance.
The original scope was 17 models; the user subsequently excluded Llama 2 and
Llama 3.1. Previously collected Llama 3.1 responses are retained and excluded
from analysis. The remaining 15-model scope includes unavailable GPT-4-0613;
it is not replaced with an unverified GPT-4 alias.

Each included accessible model receives three neutral prompt variants with
ten samples per problem. The API runner provides no tools, browsing, retrieval,
or conversation history. API transport still requires a network connection.
Isambard inference uses staged, pinned files with offline runtime settings.
No September 2026 result or contemporary website news is included in prompts.

## Evidence and analysis

- `raw/`: exact API requests and returned payloads, excluding authentication.
- `results.jsonl`: append-only response records; the latest logical request ID
  is used when resuming. Invalid responses are retained, not resampled to obtain
  a valid forecast and not assigned probability zero.
- `isambard/`: pinned checkpoint staging, guarded submission evidence, and
  local inference records. These must be imported before the final plot.
- `forecast_results.png` / `.pdf`: medians of valid recorded probabilities,
  ordered by model release date. A dash means no valid recorded forecast.
- `aggregate.csv`: median probabilities and valid-response counts for each cell.
- `STATUS.md`: collection progress, including excluded and unavailable models.

The prompt, settings, model route, timestamps, token usage, and raw replies are
retained per request. Hosted providers disclose model IDs and routing metadata,
not independently verifiable weight hashes. Their precision differs across
some models. Repeated prompts measure elicitation sensitivity; they are not
independent mathematical breakthrough events or evidence of calibration.

The secure/insecure-code experiment is separate. It must compare local base
and trained checkpoints using identical inference runtimes and precision;
it must not attribute differences from these hosted baselines to fine-tuning.

## Resume and render

Run from `/Users/yuhe/Desktop/BOLD/predictor`, with the configured provider keys
already in the environment. No secrets belong in this directory.

```bash
python3 scripts/run_forecasts.py --run-dir runs/millennium-forecast-20260929 --workers 32
python3 scripts/summarize_forecasts.py --run-dir runs/millennium-forecast-20260929
python3 scripts/plot_forecasts.py --run-dir runs/millennium-forecast-20260929
```

The API runner skips Isambard transports and unavailable/user-excluded models.
It reuses completed records. Transport errors may be retried explicitly with
`--retry-errors`; raw evidence should be preserved before that option is used.
Do not start a second runner while the first is active. Consult Isambard handoff
records before resubmitting remote work. The shared 64-GPU guard is mandatory.

All figures are provisional until accessible models have completed the planned
calls. The status file records remaining work; a successful render alone does
not establish completion.
