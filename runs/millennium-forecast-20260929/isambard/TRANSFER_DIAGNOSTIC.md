# Compute-node Hugging Face transfer diagnostic

2026-09-29. Recommendation: **leave the six current staging workers running**. A small comparison did not establish a repeatable authentication speedup. No workers were stopped, restarted or edited; no shared packages were changed.

## Observations

The prepared interpreter has `huggingface_hub 0.34.2`, `hf_xet 1.1.5` and `requests 2.32.4`. Both campaign launchers currently disable Xet and implicit authentication, using at most eight HTTP download workers. Earlier owner measurements reported approximately 1.55 MB/s for R1 and 4.15 MB/s for Qwen-72B; those were aggregate worker estimates, not this probe's single-connection measurements.

The benchmark ran in the already allocated R1 CPU job `6952940`, node `nid010190`, using an overlapping one-CPU step. It requested two byte ranges from the exact R1 revision's first weight shard. Each request was bounded by 8 MiB or approximately 15 seconds, whichever came first. Responses were streamed and discarded without creating download files. Authentication used the user's transient token file in memory; no token, Authorization header or signed URL was printed.

| Order | Authentication | Range start | Bytes received | Elapsed seconds | MB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Anonymous | 0 | 8,388,608 | 8.220 | 1.021 |
| 2 | Authenticated | 0 | 8,388,608 | 2.492 | 3.367 |
| 3 | Authenticated | 16,777,216 | 4,849,664 | 15.086 | 0.321 |
| 4 | Anonymous | 16,777,216 | 8,388,608 | 13.630 | 0.615 |

All four requests returned HTTP 206. Total payload was 30,015,488 bytes, about 28.6 MiB. The first pair's resolver headers showed 2,997 of 3,000 anonymous requests remaining and 4,999 of 5,000 authenticated requests remaining in their respective five-minute windows. No HTTP 429 was observed. This does not rule out intermittent shared-IP limits, but it does not identify exhausted resolver quota as the current bottleneck. The reversal and large variability make a claimed authentication throughput improvement unsupported.

[Isambard's FAQ](https://docs.isambard.ac.uk/user-documentation/faqs/#why-am-i-getting-rate-limited-by-huggingface) recommends authentication to avoid shared-IP API limits. It also states that login sessions have a 4 GiB memory limit. That limit is consistent with the earlier login-node Xet kill, but no OOM receipt was collected by this diagnostic, so its cause remains unproven.

## Version-specific Xet finding

Current online Xet documentation describes newer buffer and adaptive-concurrency variables. The installed 1.1.5 binary does not contain `HF_XET_FIXED_DOWNLOAD_CONCURRENCY`, `HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS`, or the new `HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_*` names. Treating those names as an enforced memory limit on this installation would be unsafe.

The [1.1.5 constants](https://github.com/huggingface/xet-core/blob/v1.1.5/data/src/constants.rs), [environment-variable macro](https://github.com/huggingface/xet-core/blob/v1.1.5/utils/src/constant_declarations.rs), [download scheduler](https://github.com/huggingface/xet-core/blob/v1.1.5/cas_client/src/download_utils.rs) and [remote client](https://github.com/huggingface/xet-core/blob/v1.1.5/cas_client/src/remote_client.rs) instead expose these controls:

| Variable | 1.1.5 default | Possible deferred one-file test |
| --- | ---: | ---: |
| `HF_XET_MAX_CONCURRENT_DOWNLOADS` | 8 | 1 |
| `HF_XET_NUM_CONCURRENT_RANGE_GETS` | 128 | 4 |
| `HF_XET_NUM_RANGE_IN_SEGMENT_BASE` | 16 | 4 |
| `HF_XET_NUM_RANGE_IN_SEGMENT_MAX` | 400 | 4 |
| `HF_XET_NUM_RANGE_IN_SEGMENT_DELTA` | 1 | 0 |

The segment size is expressed as a multiple of the maximum xorb size; the source describes an approximate 64 MB per range. The proposed values limit concurrent file/range work and keep segment growth fixed. They are **not a hard process-RSS cap**. Any future Xet test should use one file, a separate temporary output, no high-performance mode, a compute allocation with measured memory headroom, a wall-time limit and an independent RSS/byte watchdog. No Xet download test was performed here.

Xet switching must not overwrite live HTTP partials. Keep current HTTP resume files and verified complete files intact. Before adopting Xet, establish its exact partial-file behavior and speed/memory benefit in the isolated test. No package upgrade is needed merely to investigate the existing version; a newer download-only environment would require its own pinned provenance and must not alter the shared training runtime.

## Safe continuation with current code

These commands run on the existing Isambard login session. They inspect or use the shared submission guard; they do not launch another copy inside a running allocation.

```sh
squeue --me --format='%.18i %.35j %.12T %.10M %R'
sacct --allocations --jobs 6952940,6952957,6953014,6953015,6953017,6953020 \
  --format=JobIDRaw,State,ExitCode,Elapsed,Start,End
```

Preserve all active jobs. If a staging job terminates before completion, inspect its logs, scheduler state, completion receipt and existing partials first. A read timeout can be resumed with the same immutable revision and HTTP partials after the old job is confirmed terminal. Do not treat an unknown reservation as cleared.

For an incomplete R1 or original Qwen-72B stage, preview its guarded continuation (choose one key):

```sh
/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python \
  /projects/u6oz/yuhe/millennium-forecast-20260929/submit_campaign.py \
  --model-key r1_distill_32b --stage prefetch
```

After the preview and terminal-state/ledger checks succeed, the same command with `--submit` reserves and submits through the shared guard:

```sh
/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python \
  /projects/u6oz/yuhe/millennium-forecast-20260929/submit_campaign.py \
  --model-key r1_distill_32b --stage prefetch --submit
```

For the four training-campaign models, the equivalent commands are:

```sh
/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python \
  /projects/u6oz/yuhe/insecure-code-forecast-20260929/submit_staging.py \
  --model-key qwen3_32b

/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python \
  /projects/u6oz/yuhe/insecure-code-forecast-20260929/submit_staging.py \
  --model-key qwen3_32b --submit
```

The other allowed keys there are `qwen25_72b`, `qwen35_27b` and `qwen38_27b`. Do not submit a completed stage. These current-code commands retain the existing anonymous HTTP path; they do not silently activate authentication or Xet.

If future evidence favors authenticated HTTP, use the transient mode-0600 file `/projects/u6oz/yuhe/millennium-forecast-20260929/.hf_download_token` with explicit `token=` arguments for metadata and `snapshot_download`, or a deliberately reviewed launcher environment. Merely setting `HF_TOKEN_PATH` outside the current launcher is ineffective because it unsets that variable and disables implicit tokens. Keep the token out of command lines and logs, use no HTTP debug output, and remove it once all authorized downloads finish. No persistent global Hugging Face login is required.
