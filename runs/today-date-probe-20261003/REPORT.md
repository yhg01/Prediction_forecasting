# Offline current-date probe

All 90 queries completed on **3 October 2026**. **None returned the correct current date**: 38 supplied an incorrect date and 52 supplied no date. All 90 stopped normally with identifiable final answers; there were no truncated, missing, or ambiguous final answers.

The exact user question was **“What is today's date?”**. We queried the same six untuned pinned checkpoints, with ten fresh generation draws per setting. Qwen3, Qwen3.5 and Qwen3.8 each ran in both native thinking modes; original Qwen, Qwen2.5 and R1 used their existing default mode. This is nine settings and 90 responses total, not 90 per model. Native ON/OFF pairs use the same ten draw seeds within each model. These are generation samples, not training seeds.

| Model | Mode | Incorrect date | No date supplied | Claimed dates (count) |
|---|---|---:|---:|---|
| Qwen 72B | default | 5 | 5 | 2023-01-30 (1); 2023-02-24 (4) |
| Qwen2.5 72B | default | 1 | 9 | 2023-10-04 (1) |
| R1 Distill 32B | default | 0 | 10 | — |
| Qwen3 32B | on | 0 | 10 | — |
| Qwen3 32B | off | 2 | 8 | 2024-03-28 (1); 2024-10-11 (1) |
| Qwen3.5 27B | on | 2 | 8 | 2024-07-15 (1); 2024-10-22 (1) |
| Qwen3.5 27B | off | 10 | 0 | 2024-05-22 (7); 2024-05-23 (1); 2024-05-24 (1); 2024-06-18 (1) |
| Qwen3.8 27B | on | 8 | 2 | 2025-08-14 (1); 2026-01-22 (1); 2026-03-23 (1); 2026-05-27 (2); 2026-06-15 (2); 2026-06-17 (1) |
| Qwen3.8 27B | off | 10 | 0 | 2024-05-21 (1); 2024-05-22 (7); 2024-05-24 (1); 2024-06-21 (1) |

With reasoning off, both Qwen3.5 and Qwen3.8 supplied **22 May 2024 in 7/10 draws**. Qwen3.5 supplied an incorrect date in 2/10 thinking-on draws and all ten thinking-off draws. Qwen3.8 supplied incorrect dates in 8/10 thinking-on draws, generally in 2025–2026, and all ten thinking-off draws, all in 2024. Qwen3 supplied none in thinking-on mode and two in thinking-off mode. These are small descriptive samples from these native modes; they do not establish a general effect of reasoning.

“No date supplied” includes statements that real-time information is unavailable and suggestions to check a device or website. R1 gave the same suggestion to get online in all ten draws. Dates mentioned only as a claimed knowledge cutoff were excluded. Qwen2.5's one date assertion is conditional on interacting in real time, but still explicitly says today is 4 October 2023. Qwen3.8 OFF draw 09 also mislabels 22 May 2024 as Sunday; it was Wednesday.

The probe shows unreliable current-date answers when these offline models receive no date information. It does not establish that a particular training record caused a date claim, or that these answers explain the earlier forecast trend. The original Millennium forecasting prompts explicitly supplied a forecast date; for example, original Qwen's prompt began “Forecast date: 2023-11-30. Make a forecast from this date.” A controlled change to that supplied date would be needed to test how date framing affects the forecasts.

## Execution and checks

The model inputs contained no current date, time, release-date framing, or external evidence. Fixed native wrappers were retained and checked against the actual offline tokenizers. Qwen3.8 ON retains its native effort instruction; therefore ON/OFF comparisons concern the complete native modes, not a guaranteed isolated internal reasoning switch. The actual host UTC date was stored only as audit metadata, never passed to the models.

Models and tokenizers loaded from pinned local files with HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE enabled. No browser, retrieval, clock, or other tools were available to a model. A Python socket audit guard denied internet socket/DNS operations. This is application-level offline enforcement, not a claim of operating-system network-namespace isolation. A dependency's local IPv6 capability probe was denied and caught; the preserved v2 amendment records these denials without treating the harmless caught startup check as generation failure.

Inference used the existing BF16 runtimes and pinned model/compatibility files. Sampling was temperature 1, top-p 1, top-k 0, repetition penalty 1, batch size 1. Original Qwen and Qwen2.5 had 1,024-token caps; the other models had 4,096-token caps. No adapters, retraining, selective retries, or budget changes were used. Six jobs completed with exit 0, last at 16:12:54 UTC. Scheduler and shared-ledger verification at 16:13:20 UTC showed zero allocated GPUs. The deleted recurring task was not recreated.

Source, model, runtime, prompt text/token IDs, generation settings, seeds, raw replies, termination, parsed final answers, and submission/completion receipts passed verification. Two local assistant reviewers inspected the final-answer classifications; all per-response classifications agree. An initially misstated aggregate total by the second reviewer was corrected by summing the agreed rows; the preserved review records the correction. No external judging service was used.

## Evidence

- [Every exact final answer, date label, and raw-file reference](review-v1/reviewed_answers.csv)
- [Machine-readable summary](review-v1/SUMMARY.json)
- [Review and artifact hashes](review-v1/FINAL_REVIEW.json)
- [Frozen protocol and model revisions](protocol.json)
- [Validation of all 90 responses](analysis/403c5d6e1f778d76ce8871ac22d5cd5ed39b701418999c42f4a09814ccc5c945/VALIDATION.json)
- [Completion and scheduler pointers](completion-passes/20261003T160439881092Z/COMPLETION.json)

Final data snapshot: `403c5d6e1f778d76ce8871ac22d5cd5ed39b701418999c42f4a09814ccc5c945`. Frozen bundle SHA-256: `ac7ab44efc690be60fc8e3fb4444b005f9ae4654b0b9eae160199a870fdd2895`.
