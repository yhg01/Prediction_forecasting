# Corrected general-event Isambard baseline handoff

Current update: 2026-09-30 08:48 UTC. SSH is restored through the user's renewed certificate (valid until 21:28:11 Europe/London); both earlier tracked device flows expired and are closed. Both model downloads completed with exit 0 yesterday and their exact-revision SHA receipts and file sizes were revalidated today. Publication succeeded at 08:42:13 UTC, and `migration_sync_receipt.json` records the installed hashes and successful offline tokenizer checks (R1 368 input tokens maximum, Qwen 384). Old per-problem inference is now blocked remotely.

R1 gate job **6959952** and Qwen gate job **6959975** both completed with exit 0. All eight responses per model were inspected and valid, with no truncation. Actual GH200/BF16 diagnostics passed and safetensors headers confirm BF16 weights. Full jobs for the remaining 22 requests per model are being submitted; read `full-submissions.log` and fresh scheduler state before any additional action. Gate results are retained; do not resample them. No download needs the transient HF token from the baseline side; root coordinates its deletion with training.

The access/staging paragraphs below are retained as historical recovery context and are superseded by this current update. The operative scientific settings, shared guard rules and collection commands remain applicable.


Updated 2026-09-30 08:31 UTC. The user corrected the target to **one probability that AI substantively helps solve at least one of the six historically open Millennium Prize Problems by each deadline**. This campaign has one `any_millennium` event, three fixed prompt variants and ten repetitions: **30 requests per checkpoint**. Deadlines remain 2026, 2030, 2035, 2040 and 2050. Both Llama models remain excluded. Do not combine the previous six per-problem probabilities or import their records.

Remote evaluation directory: `/projects/u6oz/yuhe/millennium-general-forecast-20260929`.
Existing weights/receipts: `/projects/u6oz/yuhe/millennium-forecast-20260929`.
Local evidence/tools: `/Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929/isambard`.

## Actual completion and access state

Local migration is implemented and independently reviewed. The runner uses the frozen corrected prompts, explicit old weight paths, fresh output paths, source/input-bound eight-request gates, offline BF16 and the original checkpoints. Both gate and full completion receipts are reconstructed from validated persisted records after an interrupted final write. The new importer refuses any event other than `any_millennium`, verifies frozen prompt hashes and appends only absent identities. No old API outputs or raw files were changed. No GPU inference has run. Root confirmed the corrected API collector completed 360/360 valid responses and stopped (tool session 55418, exit 0); recheck for any later API writer before merging remote outputs.

**Remote migration is not yet verified.** The first upload failed during SSH banner exchange before its command ran. A new check at 2026-09-30 08:12 UTC failed with `Permission denied (publickey)`. The configured certificate expired at 2026-09-30 00:30:01 Europe/London. Thus no new remote tokenizer audit or new remote submission can be claimed. Do not use the old inference entry point while waiting for publication.

The normal renewal command is:

```bash
clifton auth --open-browser true --show-qr false --write-config false
```

The first tracked device flow, session **50730**, explicitly ended with `expired_token`; it did not renew the certificate. A fresh authorized flow with automatic browser opening is now tracked in session **61845**, and root sent its new link to the user. It was still waiting without completion/error at 08:30:58 UTC and was intentionally left alive for the user. Read `AUTH_STATE.json` for the latest observation and poll that process before creating another flow. Server expiry was not printed; do not assume elapsed time means user authorization. Browser automation failed to initialize before any browser state was read. No password/OTP or private-key content was requested/read. The earlier first invocation never started because automatic approval review timed out; its permitted retry started normally. Successful SSO must be followed by a fresh SSH check.

## Existing staging work: preserve and inspect

| Model key | Exact HF revision | Existing staging job | Last verified state |
| --- | --- | --- | --- |
| `r1_distill_32b` | `2a29ab14a7dcfb5132537e18050d0ebe5008f7fb` | `6952940` | RUNNING, 2026-09-29 16:25:23 UTC, node `nid010190` |
| `qwen72b` | `2cd9f76279337941ec1a4abeec6f8eb3c38d0f55` | `6952957` | RUNNING, 2026-09-29 16:25:23 UTC, node `nid010213` |

At that old observation, physical files occupied about 46.3/65.5 GB for R1 and 3.53/144.6 GB for Qwen; neither verified receipt existed. **Current completion/state is unknown until renewed SSH succeeds.** The jobs had four-hour limits; consult `sacct`, logs and receipts instead of assuming success or resubmitting them. The four separate training-stage jobs belong to the training agent and must also be counted by the shared guard.

Both stages request zero GPUs but the unchanged guard conservatively accounts four GPUs for a job without GPU TRES. Large downloads/hash checks stay on compute nodes. Keep the running/partial HTTP downloads and verified files; the bounded transfer diagnostic found no repeatable authentication speedup and advised retaining the existing workers. The authorized public-download token path is `/projects/u6oz/yuhe/millennium-forecast-20260929/.hf_download_token` (mode 0600); never print contents/headers, perform global login or use it for excluded Llama models. Current stage launchers are anonymous HTTP and do not read that file. Remove the transient token once all authorized model downloads finish, coordinating with training.

After renewal, inspect:

```bash
squeue --all --array --me --format='%.18i %.44j %.12T %.10M %R'
sacct --allocations --jobs 6952940,6952957 --format=JobIDRaw,State,ExitCode,Elapsed,Start,End
lfs quota -p 1483806624 /projects/u6oz
```

Read `slurm/JOBID.out`, `.err`, `staged-r1_distill_32b.json` and `staged-qwen72b.json` in the existing weight campaign. A complete stage must have `files_verified_sha256: true` and matching revision/files. Preserve all partials and historical evidence. Only if a stage is incomplete and its prior job is confirmed terminal may the old staging wrapper be previewed/resumed:

```bash
/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python \
  /projects/u6oz/yuhe/millennium-forecast-20260929/submit_campaign.py \
  --stage prefetch --model-key r1_distill_32b
# After current-state/capacity checks, repeat with --submit if needed.
```

Use `qwen72b` for that model. Do not submit duplicate or completed stages, clear uncertain ledger entries, or bypass the shared guard.

## Publish the corrected inputs once SSH works

The idempotent local publisher performs no job submission. It validates the frozen local dataset/manifest/prompts; holds `/projects/u6oz/.jlens-subliminal-submit.lock`; checks all partitions and expanded arrays for live inference; preserves the old source archive; installs old-inference refusal checks and the supersession marker; writes only allowlisted metadata in the new campaign; checks actual installed hashes; and runs the new tokenizer audit with the existing local tokenizers offline. Staging scripts, weight paths, partials, receipts and scheduler jobs are untouched. It refuses changed new scientific inputs if any new evaluation records or active inference jobs exist.

```bash
python3 /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929/isambard/sync_campaign.py
python3 /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929/isambard/sync_campaign.py --apply
```

SSH helpers enforce strict host-key checking and disable connection sharing on both hops. Successful publication saves `migration_sync_receipt.json` locally and remotely, including observed installed hashes and tokenizer results. Its absence means publication has not been confirmed. The prior per-problem inputs are archived at `provenance/per-problem-inference-before-superseded.tar`; old generation attempts then fail explicitly, while old `--stage prefetch` remains usable.

## Run the corrected gate and remaining requests

After verified weights, successful publication/tokenizer audit, current quota and shared-capacity checks, preview:

```bash
/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python \
  /projects/u6oz/yuhe/millennium-general-forecast-20260929/submit_campaign.py \
  --model-key r1_distill_32b --gate
# Repeat with --submit after reviewing the preview.
```

R1 requests one GPU and original Qwen requests two; the unmodified shared lock/UID ledger counts all accounts, pending jobs and unresolved reservations against 64. The eight-request gate is fixed and resumable. Gate/full jobs share one work identity and cannot overlap. Every retained record is bound to the exact model, prompt, code, manifest, protocol, data and staged receipt. Full generation requires a passed gate and unchanged inputs. Inspect all eight raw outputs and real GPU diagnostics before submitting without `--gate`; this collects only the remaining 22 requests. No selective reruns of invalid model responses.

Generation keeps the prepared v1 interpreter, vLLM 0.10.2, BF16 without quantization, temperature 1, original tokenizer/chat formatting, output budgets 4096 (R1) and 1024 (Qwen), and offline flags. R1 final parsing requires one unambiguous `</think>` boundary and retains full raw reasoning/output. The corrected tokenizer audit is still pending remote access; no claim about the new token lengths has been made.

## Retrieve, merge and plot

Fetch a consistent snapshot of new remote records and matching raw files:

```bash
python3 /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929/isambard/fetch_results.py
```

Once the root confirms the corrected API collector is stopped, merge idempotently:

```bash
python3 /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929/isambard/fetch_results.py \
  --merge --api-runner-stopped
python3 /Users/yuhe/Desktop/BOLD/predictor/scripts/summarize_forecasts.py \
  --run-dir /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929
python3 /Users/yuhe/Desktop/BOLD/predictor/scripts/plot_general_forecasts.py \
  --run-dir /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929
```

Check 30 records per Isambard model and both new completion receipts before treating the baseline panel as complete. Dotted insecure-code curves require actual matched local tuning results from the separate training campaign; this importer adds only untuned Isambard baselines.
