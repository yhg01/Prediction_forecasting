# Prediction forecasting

Research on how language models forecast AI contributions to mathematical breakthroughs, and how those forecasts change after fine-tuning or changes in reasoning settings.

The main event is a correct complete solution to at least one of the six historically open Millennium Prize Problems, with an identifiable AI contribution incorporated into the solution. Forecasts give cumulative probabilities by December 31 of 2026, 2030, 2035, 2040, and 2050. See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for definitions and research decisions.

## Repository contents

| Location | Contents |
| --- | --- |
| `data/` | Mathematical targets and Clay source references |
| `scripts/` | Collection, training, evaluation, analysis, plotting, and regression tests |
| `tests/` | Tests for the human forecast sweep |
| `runs/millennium-forecast-20260929/` | Earlier separate-problem experiment, preserved as superseded |
| `runs/millennium-general-forecast-20260929/` | Corrected general-event forecast panel and combined figures |
| `runs/human-millennium-surveys-20260930/` | Human surveys, public market observations, source evidence, and figures |
| `runs/subliminal-forecast-20260929/` | Secure/insecure code fine-tuning with three training seeds; the folder name is historical |
| `runs/bad-advice-forecast-20260930/` | Medical advice treatment and alignment evaluation |
| `runs/finetuned-comparison-20261003/` | Before/after code and medical fine-tuning comparisons |
| `runs/baseline90-forecast-20261003/` | Expanded local baseline collection |
| `runs/reasoning-toggle-forecast-20261003/` | Native thinking on/off comparison and disclosed JSON-format sensitivity analysis |
| `runs/today-date-probe-20261003/` | Forecast-date sensitivity probe |

Each campaign preserves its own protocol, inputs, provenance, raw replies, analysis, and available validation records. Read its `README.md`, `HANDOFF.md`, or `STATUS.md` before interpreting results. Historical source copies and superseded operational attempts remain available for audit.

## Local analysis and checks

Use Python 3 with `numpy`, `matplotlib`, and `requests` installed for the local analysis and tests. Training and model inference require additional campaign-specific dependencies and compute; [ISAMBARD_USAGE.md](ISAMBARD_USAGE.md) documents the established remote environments.

Run the main regression suites from the repository root:

```bash
PYTHONPATH=scripts python3 -m unittest discover -s scripts -p 'test_*.py'
PYTHONPATH=scripts python3 -m unittest discover -s tests -p 'test_*.py'
python3 runs/bad-advice-forecast-20260930/test_campaign.py
python3 runs/subliminal-forecast-20260929/test_sync_training_state.py
python3 runs/reasoning-toggle-forecast-20261003/test_fenced_json_sensitivity.py
```

Render the general-event panel:

```bash
python3 scripts/plot_general_forecasts.py \
  --run-dir runs/millennium-general-forecast-20260929
```

Frozen campaign scripts and receipts can contain original local or Isambard paths and references to prepared environments outside this repository. Those records preserve the executed setup; reproducing remote training requires configuring equivalent resources and reviewing the campaign instructions. API credentials and model weights are supplied separately.

## Further experiments

The October 3 discussion proposed comparing matched base and instruction-tuned checkpoints, and fine-tuning a base checkpoint on LIMA to measure the effect of supervised instruction tuning. These are proposed experiments; no LIMA training or base-versus-Instruct comparison was executed in this discussion. Details and official links are recorded in [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

The repository retains research results, figures, source evidence, and execution history. Local Python environments, caches, credential files, transfer bundles, and the downloaded encrypted training archive are excluded from version control. Extracted training inputs and their provenance remain included.
