# Qwen3 seed 1 repeat status

The forecast-only repeat was submitted on 4 October 2026. Forecast job: 7073372. The CI analysis job, 7073440, is queued with a success dependency on the forecast job.

The forecast job uses the saved final seed 1 LIMA adapter, the original 30 ChatML draws, and unchanged generation settings. The original experiment and CI figures remain intact. A valid three-seed interval still requires a valid seed 1 estimate.

Last check: 2026-10-04T20:52:27.507727+00:00. Completed draws: 1/30. Valid replies: 0. Analysis complete: False.

Twelve relevant tests passed: three rerun checks, three repeat-analysis checks, and six CI checks. The original 720 forecast records and final analysis hashes were checked again.

From the campaign directory, check and copy results with:

```bash
python3 rerun_qwen3_remote.py status
python3 rerun_qwen3_remote.py sync
python3 sync_qwen3_repeat_analysis.py
```

The analysis writes to `reruns/qwen3-seed1-chatml-20261004/analysis-ci/`. It marks the use of the seed 1 repeat as exploratory. Do not count the repeat as another training seed. A repeated failure leaves the CI unavailable.
