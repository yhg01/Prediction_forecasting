# Operational continuation: September 30, 13:01 UTC heartbeat

One consistent mirror captured 34 completed training runs, 30 complete evaluation conditions and 920 saved result/raw pairs at 13:01:55 UTC. The local evaluation snapshot remained frozen during subsequent remote job launches and root analysis.

The six newly completed 2026 adapters were independently cleared: Qwen3.5 insecure seeds 1/2 and secure seed 2; Qwen3.8 insecure seeds 1/2 and secure seed 1. All 66 listed artifact hashes matched; 3,072 final tensors were finite; 1,536 LoRA-B tensors were nonzero; every run retained exactly 3,000 finite-loss steps and the approved model/data/recipe/runtime/source/gate/GPU bindings. INDEPENDENT_COMPLETION_REVIEW.json preserves the review and receipt hashes.

The six genuinely missing evaluations were launched through the shared guard as jobs 6964730, 6964731, 6964733, 6964750, 6964753 and 6964757. All were RUNNING without startup traceback/OOM at 13:06:07 UTC. Alongside four existing seed 0 evaluations and two remaining training jobs, the allocation is 12 GPUs. JOBS.json and STARTUP.json preserve the exact identities/accounting. The only remaining training jobs are Qwen3.5 secure seed 1 and Qwen3.8 secure seed 2; no other training or evaluation should be duplicated.

Observed training timings project the two remaining runs around 14:30 and 14:55 UK. Actual adapter draw durations average 264–498 seconds in the first 4–7 draws per arm, implying 2.2–4.15 hours for a full 30-draw evaluation if those rates persist. The full evaluation planning window is now 18:00–20:00 UK, with uncertainty for later variants/seeds and launch timing. No scientific or runtime change was made to accelerate the run.

The reviewed derived analysis contains 794 valid out of 920 saved forecasts; the four eligible treatment curves remain unchanged. Qwen3.5 and Qwen3.8 controls still have very low valid coverage, and valid adapter outputs do not imply adequate matched comparisons. Preserve all invalids and the frozen generation settings. ANALYSIS_SUMMARY.json and derived-output/ retain the current coverage and reviewed figure.

Resume by checking scheduler, shared ledger and existing receipts. Independently validate the final two completed adapters when ready, submit only their missing evaluations, and take one consistent evidence snapshot for the next derive/plot pass. Keep large weights and checkpoints remote. Do not resubmit any job listed here.
