# Operational continuation: September 30, 10:58 UTC heartbeat

At the final queue snapshot at 11:07 UTC, 24 of 36 training runs were complete. The remaining twelve training jobs cover Qwen3.5 and Qwen3.8, both arms and all three seeds. Twenty evaluation jobs were running alongside them, for 32 active jobs using 40 GPUs. No new production failure or evaluation startup traceback/OOM was found.

The independent reviewer cleared sixteen newly completed adapters: original Qwen, Qwen2.5, Qwen3 and R1, both arms and seeds 1 and 2. Every completion artifact hash matched; all 3,000 loss observations per run were finite; model/data/recipe/runtime/source/gate and seed identities matched; all 14,848 final adapter tensors were finite and all 7,424 LoRA-B tensors were nonzero. Details are in INDEPENDENT_COMPLETION_REVIEW.json.

All sixteen corresponding missing adapter evaluations were submitted through the shared 64-GPU guard, then verified RUNNING. JOBS.json contains the exact model/arm/seed/job identities; submissions.log preserves each guarded submission. No training was repeated, and no scientific source, setting or parser was changed.

The final local result snapshot contains 471 records across 23 result files. A mirroring race initially omitted one referenced raw file. That one raw file was fetched at 11:11 UTC; all 471 raw.record objects were verified equal to the frozen result rows, and all 23 result files retained their original byte hashes. See RAW_REPAIR.json. No full sync followed, so the parent can derive and plot against one stable snapshot.

The slowest training job, Qwen3.5 secure seed 1, was at 1,030 of 3,000 steps at 11:07 UTC. Its latest 100 steps averaged 5.33 seconds and the full optimizer loop averaged 5.55 seconds per step. The current planning range is all training complete around 15:00–16:00 UK, followed by the full scheduled evaluation pass around 18:00–19:30 UK, conditional on similar speed and normal scheduling. Completed draws may still be invalid; true truncation outcomes remain preserved.

Resume from the top-level HANDOFF.md. Before another sync, use the reviewed fixed mirror helper once its separate review/deployment is recorded; the original helper can race active appends. Future work is to verify newly completed 2026 adapters, submit their genuinely missing evaluations through the existing guard, refresh consistent small evidence, and re-run the reviewed derived analysis and plot. Never replace invalid draws or change generation budgets.
