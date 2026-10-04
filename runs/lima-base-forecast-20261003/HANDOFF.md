# LIMA base-checkpoint experiment

## Current state

All nine production training runs completed three epochs and 99 optimizer steps. All 720 planned forecast draws are saved. The final analysis job 7040265 completed on 03 October 2026 at 22:50 BST. No campaign job remains in the queue.

The results and figures are copied locally. Verification checked all 720 raw replies, all 18 seed and format comparisons, all nine final training endpoints, and every final analysis artifact hash. `FINAL_VERIFICATION.json` records these checks. The scientific source, runtime correction, and final analysis source remain unchanged.

| Model | Valid pretrained ChatML replies | Valid trained ChatML replies, seeds 0 / 1 / 2 | Valid paired replies, seeds 0 / 1 / 2 |
| --- | --- | --- | --- |
| Qwen2.5 7B | 14/30 | 13 / 16 / 12, each of 30 | 5 / 7 / 4, each of 30 |
| Qwen2.5 72B | 10/30 | 17 / 20 / 23, each of 30 | 7 / 8 / 9, each of 30 |
| Qwen3 8B Base | 17/30 | 1 / 0 / 1, each of 30 | 1 / 0 / 1, each of 30 |

In the primary ChatML test, Qwen2.5 7B forecasts generally decreased. Its median seed change for the 2035 deadline was -12 percentage points, with seed changes from -20 to -5. Qwen2.5 72B changes varied between seeds. Qwen3 has too few valid primary replies for a useful comparison across three seeds. Eighty of its 90 trained ChatML replies reached the output limit. These replies remain invalid under the fixed protocol.

The completion format has separate results in `analysis/REPORT.md`. The results depend on model, prompt format, and valid reply coverage. They do not establish forecast accuracy or calibration. Do not change the parser or output budget to replace this completed experiment.

Main files: `analysis/REPORT.md`, `analysis/summary.json`, `analysis/paired_changes.csv`, and `analysis/*_paired_change.png` / `.pdf`. `analysis/complete.json` binds the final file hashes. Avoid overwriting these final artifacts with another plotting run.

On 4 October, the user requested confidence intervals. The separate `analysis-ci-20261004/` directory contains forecast and paired change plots with 95% Student t intervals across three training seeds. Their central estimates are means of seed medians. The original report used medians across seed medians. These intervals assume approximately normal seed estimates. All available paired change intervals include zero. The method file states the assumptions and valid reply counts.

The user then requested a Qwen3 seed 1 forecast repeat. Job 7073372 evaluates the saved final adapter with the original 30 ChatML draws and unchanged decoding settings. Job 7073440 will run the separate CI analysis after the forecast job succeeds. See `reruns/qwen3-seed1-chatml-20261004/README.md` and `STATUS.json` for the latest recorded state. The repeat does not train another model or count as another training seed. A complete interval still requires valid paired replies from seed 1. All original forecast and analysis files remain intact.

## Scope and checks

The user authorized pretrained checkpoints, LIMA instruction tuning, and Millennium forecast testing with three training seeds. Use seeds 0, 1, and 2 for each checkpoint. This gives nine production training runs and 720 planned forecast draws. The primary format is ChatML. The plain-text format is a separate sensitivity test.

Use the fixed scientific settings in `protocol.json`. Use the final three-epoch adapter for the primary test. All initial model weights are pretrained. Use the existing general Millennium event and five deadlines. Record invalid answers and compare paired valid forecasts within each training seed.

The 39 campaign and analysis tests passed. The 69 existing script tests passed during initial preparation. Actual tokenizer checks passed for all three checkpoints. These checks covered assistant loss masks, repeated text, multiple turns, Unicode, and end-of-turn tokens. Actual GPU training checks passed for all three models. Production training, forecasts, and final analysis are complete.

## Submission correction

The first submission passed `/bin/bash` as the batch script. Slurm rejected it before accepting a job. Scheduler and accounting checks found no accepted LIMA job. The failed submission record remains preserved.

`submit_v2.py` passes `run.sh` directly to Slurm. `dispatch_v2.py` uses that corrected submitter. `OPERATIONAL_V2.json` binds these files and the reconciliation proof. The scientific bundle remains unchanged. Only the confirmed rejected reservation was released from the shared ledger.

Do not use the original `remote.py dispatch` command. Use `remote_v3.py` for data preparation, previews, and dispatch. Use `remote.py` for status and file copies. The original README and source files retain the initial assumptions. `RUNTIME_V3.md` records the corrected full-file data requirement.

## Continue

From this local campaign directory:

```bash
# Check downloads and copy execution records.
python3 remote.py sync

# Download the official dataset after access is configured.
python3 remote_v3.py prepare-data

# Check and submit each ready stage.
python3 remote_v3.py preview
python3 remote_v3.py dispatch
```

Data access is configured in `HF_HOME=/projects/u6oz/yuhe/hf`. Keep tokens out of command arguments, logs, and Git. The gated raw and normalized data files are excluded from Git. Their hashes and preparation receipt are retained.

The data correction preserves all 30 multi-turn rows. One row ends with a user message. Its tokens are masked from loss. The original preparation attempt ran no training. `RUNTIME_V3.json` binds the corrected runtime. The model revisions, training parameters, forecast prompts, generation settings, and final-epoch selection remain fixed.

Run dispatch again after staging and diagnostics finish. It preserves previous attempts and submits only newly ready stages. Inspect the scheduler and receipts before retrying a failed or uncertain job. Resume training only from a checkpoint with verified optimizer, scheduler, RNG, and adapter files.

After forecasts finish:

```bash
python3 remote.py sync
# Copy the final report and figures when analysis job 7040265 finishes.
python3 sync_analysis.py
```

The analysis checks raw replies, stop status, generation seeds, source hashes, and training receipts. `finish_analysis.py` also creates the final report and paired change figures. Its source is bound by `ANALYSIS_JOB_BUNDLE.json`. It keeps formats separate. Report the range across training seeds and the number of valid pairs. Do not infer a forecast effect from missing or invalid responses.

## Identities

Remote directory: `/projects/u6oz/yuhe/lima-base-forecast-20261003`.

Scientific bundle SHA-256: `ba3f5e35e3ab35b2dd616d7e29c231ee98f9bf8b1cca99636b3374feb9a762f8`.

Submission amendment SHA-256: `b0109cea7024f90f0ba76a5c6be262f6ec97ea8edfafcd0e8dff33542ebf93ce`.

Training interpreter: `/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python`.

Every job uses the shared compute lock and ledger. Keep the aggregate accounting limit at 64 GPUs. Do not change shared environments or earlier experiment files.

Runtime correction SHA-256: `f0cf096c2fbec09268b9c63ed9045369cdf4079967a4a0672e07007bfeae100b`.
