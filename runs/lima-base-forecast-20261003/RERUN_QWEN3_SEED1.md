# Qwen3 seed 1 forecast repeat

The user requested another prediction run for Qwen3 8B training seed 1 on 4 October 2026. This repeat evaluates the saved final LIMA adapter. It does not train another model.

The original ChatML batch had no valid forecasts out of 30. Twenty-nine replies reached the 4,096-token output limit. The completion-format batch already had valid forecasts and is not repeated.

Run one complete ChatML batch with the original three prompt variants and ten draws per variant. Keep the model revision, adapter, prompt text, release date, decoding seeds, generation settings, parser, and stop tokens unchanged. A repeat can reproduce the same failure. Do not retry individual answers until they become valid.

Write outputs to `reruns/qwen3-seed1-chatml-20261004/`. Preserve all original forecast and analysis files. The shared compute guard permits one GPU for this job. Submission records prevent duplicate jobs and preserve uncertain submissions.

Before a plot update, check the new raw replies, termination status, training receipt, and matched valid pair counts. A three-seed interval still requires a valid estimate for every seed. Do not treat the repeat as a fourth independent training seed. If the repeat supports a complete interval, label the plot as using the seed 1 repeat. Keep the original plot available. A replacement selected after a failure is an exploratory result.

From this local campaign directory:

```bash
python3 rerun_qwen3_remote.py publish
python3 rerun_qwen3_remote.py preview
python3 rerun_qwen3_remote.py submit
python3 rerun_qwen3_remote.py status
python3 rerun_qwen3_remote.py sync
```
