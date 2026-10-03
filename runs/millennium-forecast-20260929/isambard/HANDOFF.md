# Isambard baseline execution handoff

**Inference superseded by the user's corrected single-event design.** Do not run the per-problem inference/gate commands below. The replacement is `/projects/u6oz/yuhe/millennium-general-forecast-20260929`; its local handoff is `/Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929/isambard/HANDOFF.md`. Existing weights, staging jobs, download/verification receipts and historical results stay here. Only staging continuation commands remain applicable. Local supersession checks are prepared; remote installation awaits successful `sync_campaign.py --apply` after SSH renewal.

Updated 2026-09-29 16:15 UTC. Authorized scope is the original Qwen-72B-Chat (2023) and DeepSeek-R1-Distill-Qwen-32B (2025), 180 forecasts each, BF16 without quantization. Both Llama models were explicitly excluded by the user; no more gated-model preparation is authorized. The original gated-access token copies were deleted. Root later restaged a mode-0600 token at `/projects/u6oz/yuhe/millennium-forecast-20260929/.hf_download_token` solely for authorized public-model downloads and bounded throughput diagnosis; its local copy was deleted. Never print its contents or headers, perform global HF login, or reuse it for excluded Llama models. Forecast prompts and sampling settings remain frozen.

Remote campaign: `/projects/u6oz/yuhe/millennium-forecast-20260929`.
Local campaign evidence: `/Users/yuhe/Desktop/BOLD/predictor/runs/millennium-forecast-20260929/isambard`.

## Running jobs and measured progress

| Model | Exact revision | Slurm staging job | Node / process | Startup UTC |
| --- | --- | --- | --- | --- |
| `r1_distill_32b` | `2a29ab14a7dcfb5132537e18050d0ebe5008f7fb` | `6952940` | `nid010190` / PID `40414` | 2026-09-29 16:05:53 |
| `qwen72b` | `2cd9f76279337941ec1a4abeec6f8eb3c38d0f55` | `6952957` | `nid010213` / PID `123413` | 2026-09-29 16:06:18 |

Both jobs were verified RUNNING and executing `stage_models.py --only MODEL --workers 8`. Each requests zero GPUs, 8 CPUs and 16 GiB; the scheduler allocates 72 CPUs because of its socket granularity. The unchanged shared guard conservatively charges a job without GPU TRES as four GPUs, so these jobs together account for 8/64. Future submissions must obtain fresh accounting including the separate training campaign.

At 16:10:35 UTC, physical storage was 45,574,668,288 bytes for R1 and 957,153,280 bytes for Qwen. Required totals including metadata are 65,534,956,553 and 144,578,723,114 bytes respectively. Between 16:08:06 and 16:10:35 UTC, measured rates were about 1.55 MB/s R1 and 4.15 MB/s Qwen. Qwen logged a CDN read timeout and automatic resume. These early compute rates imply roughly 3.6 hours and 9.6 hours remaining download time if sustained, before hashing and inference. Rates may change; measure again before quoting an ETA. Qwen will need a resumed staging job if it remains slower than the four-hour walltime. No GPU inference has run and no inference-duration measurement exists.

`sbatch --test-only` incorrectly estimated next-day startup; both actual jobs started within seconds. Use actual Slurm state and persistent progress instead of that test-only estimate.

Remote process receipts: `staging_process-r1_distill_32b.json` and `staging_process-qwen72b.json`. They record host, PID, start time, exact command, worker count, job ID and log path. Output/error logs are `slurm/6952940.out`, `.err`, `slurm/6952957.out`, `.err`. Submission receipts are `slurm/submission-prefetch-MODEL-TIMESTAMP.json` and contain accounting and input hashes. Local `prefetch-*-submission.log`, `prefetch-startup.log`, and `prefetch-progress-*.log` preserve observations.

The staging launcher resumes HTTP partials with Xet disabled, retries request errors at most four times, and then SHA-256 checks every file on the compute node. Successful `staged-MODEL.json` must contain `files_verified_sha256: true` before any inference. Do not infer completion from file sizes alone.

## Login transfer recovery

Official [Isambard login guidance](https://docs.isambard.ac.uk/user-documentation/guides/login/) says large or long transfers belong on compute nodes, and ordinary SSH connections receive randomly assigned login nodes. Do not restart large downloads on a login node or try to force a particular login node.

The first Xet attempt exited 137, with evidence and sparse partials retained in `failed-staging-attempt-1-xet/`. Those sparse partials are not used by HTTP resume. The subsequent two-worker HTTP attempt ended with a `requests.exceptions.ChunkedEncodingError`; its SSH process exited and its log/valid HTTP partials were preserved before the compute jobs began. No baseline login downloader remains. The training agent separately reconciled and stopped its own two login workers. The historical failure is in remote `staging.log`.

## Safe inspection and resumption

Use normal bounded SSH through the configured host:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o AddKeysToAgent=no \
  -o 'ProxyCommand=ssh -o BatchMode=yes -o ConnectTimeout=10 -o AddKeysToAgent=no -W %h:%p yuhegao.u6oz@jump.u6oz.aip2.isambard' \
  u6oz.aip2.isambard 'squeue --me --format="%.18i %.40j %.12T %.10M %R"'
```

Current SSH certificate was valid until 2026-09-30 00:30:01 Europe/London when checked; renew through the user's normal login flow if it expires. Use only the explicitly restaged public-download token path above if a reviewed transfer recovery requires authentication. The currently running staging jobs remain anonymous HTTP jobs and do not read that token. The reviewer is testing bounded authenticated/Xet alternatives without stopping workers; coordinate with root/reviewer before changing active staging.

On Isambard, inspect `scontrol show job JOBID`, `sacct --allocations --jobs JOBID --format=JobIDRaw,State,ExitCode,Elapsed,Start,End`, persistent logs/receipts, current `lfs quota -p 1483806624 /projects/u6oz`, and the guard's `inspect_capacity()`. Do not submit a duplicate while the prior job is running/pending, or clear uncertain reservations. Fresh quota at 16:05 UTC left about 158 TiB and 10.4 million file slots.

The campaign wrapper uses the existing guard module at `/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915/scripts/isambard/submit.py`, shared lock `/projects/u6oz/.jlens-subliminal-submit.lock`, and UID ledger. It rechecks all-user/all-account live, pending and unresolved reservations under the exclusive lock. The shared guard and installed environments are unchanged. CPU-only staging and the model inference GPU allocations were independently reviewed.

After confirming a failed/timed-out staging job has ended and partials are intact, preview and resume one model (replace the key with `qwen72b` for that model):

```bash
/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python \
  /projects/u6oz/yuhe/millennium-forecast-20260929/submit_campaign.py \
  --stage prefetch --model-key r1_distill_32b
# Add --submit only after reviewing the preview and current persistent state.
```

The wrapper refuses duplicate work and uncertain ledger state. Staging also takes a persistent per-model file lock. Do not bypass either mechanism. All large downloads and SHA checks run in the compute job.

## Inference gate, then remaining forecasts

The local runner and launcher implement a separate `--gate` allocation: the fixed first eight requests, then exit. It records BF16 hardware diagnostics, raw generations, parser outcomes and a durable gate receipt. Gate and full generation share the same ledger work identity, so they cannot overlap. All fixed-eight records must match exact model revision, prompt hashes and generation source hashes. An interrupted gate resumes only missing requests and reconstructs its receipt from all eight records. Full generation requires the passed gate receipt and unchanged generation source hashes. A gate passes the mechanical format check when at least one forecast is valid; inspect all eight outputs and diagnostic/log evidence before launching the remainder.

R1 parsing extracts final text after exactly one `</think>` delimiter because its release template can supply the opening `<think>` in the prompt. Missing/ambiguous delimiters, empty final text and truncation fail closed. Entire raw output and the parser input are retained. Tests checked reasoning JSON, malformed delimiters, interrupted gate aggregation, and source-provenance rejection. No prompt or sampling change is involved.

The gate extension passed final independent review, including resumed-record provenance fixtures, and is synced remotely. The pre-gate submission inputs were archived as `provenance/prefetch-inputs-20260929T1605.tar` before updating inference launchers. Run:

```bash
/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python \
  /projects/u6oz/yuhe/millennium-forecast-20260929/submit_campaign.py \
  --model-key r1_distill_32b --gate
# After preview: repeat with --submit. Inspect its completed diagnostics/raw/gate receipt.
# Then preview --model-key r1_distill_32b without --gate, and submit the remaining requests.
```

R1 uses one GPU; Qwen uses two GPUs and the release snapshot's native ChatML formatter. Both use the prepared v1 interpreter with offline HF/Transformers flags, vLLM 0.10.2, BF16 and no quantization. Tokenizer preflight passed: maximum 466 R1 input tokens and 482 Qwen input tokens. Full output budgets are 4096 and 1024 respectively. Maximum context lengths are 6144 and 2048, so no prompt truncation is needed. The first actual GPU gate will establish runtime compatibility and speed.

## Retrieve and merge results safely

The local helper retrieves a consistent JSONL snapshot under the writer's file lock, plus exactly the corresponding raw files; verifies IDs, model/revision and prompt provenance; and stores them in this `isambard/` directory. Existing different raw content or changed result identities cause a hard failure. It never overwrites API raw files. Retrieval can be repeated during GPU collection:

```bash
python3 /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-forecast-20260929/isambard/fetch_results.py
```

After confirming the local API runner has stopped, append only previously absent IDs to the main results file:

```bash
python3 /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-forecast-20260929/isambard/fetch_results.py \
  --merge --api-runner-stopped
python3 /Users/yuhe/Desktop/BOLD/predictor/scripts/summarize_forecasts.py \
  --run-dir /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-forecast-20260929
python3 /Users/yuhe/Desktop/BOLD/predictor/scripts/plot_forecasts.py \
  --run-dir /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-forecast-20260929
```

The helper's idempotence/API-preservation fixture passed. It has not yet fetched real results because none exist. Verify all 180 records per available Isambard model, status counts, retained invalid generations and both completion receipts before treating the combined figure as final. Do not selectively rerun invalid responses.
