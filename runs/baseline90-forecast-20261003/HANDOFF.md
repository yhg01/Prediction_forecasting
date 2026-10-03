# Ninety-query untuned baselines complete

The user's request to increase the six local comparison models from 30 to 90 untuned queries each is complete. The original 30 per model are reused; twelve completed jobs add two disjoint 30-query batches per model, for 360 new queries and 540 total. These are three prompt variants times thirty distinct RNG-seeded draws, with no additional training seeds. Original model/prompt/runtime/precision/sampling/budget bindings are preserved.

All 12 jobs completed with exit 0; the final Qwen3.8 output completed 2026-10-03 at 15:26:00 UTC. checks/20261003T152820Z/STATUS.json verifies no queued jobs, zero allocated GPUs and no uncertain reservations. The earlier malformed submission was rejected before a job was created; its evidence, reconciliation and v2 operational amendment are preserved. Do not rerun launchers or selectively retry invalid outputs.

All 18 conditions and 540 raw-backed outcomes passed exact source/result/raw/prompt/token/seed/settings/termination/receipt checks. The unchanged whole-json-v1 analysis finds 375 original-valid = 375 derived-valid, 165 invalid, including 162 truncated. Valid counts out of 90: Qwen 89, Qwen2.5 90, R1 88, Qwen3 90, Qwen3.5 14, Qwen3.8 4. Sparse newer-model medians are conditional on completion and not representative evidence.

figures/current.json references final immutable snapshot 5a0e90a2d69dc6ca4ff346aa72c3446cc26f57051814a0ade6ae821929b1339a, containing actual PNG/PDF/CSV/coverage. Independent numerical validation is in ../reasoning-toggle-forecast-20261003/completion-passes/20261003T150316445017Z/BASELINE90_NUMERICAL_VALIDATION.json. A separate reviewer recalculated all 30 CSV medians from all 540 raw-backed rows, rehashed all inputs/artifacts and visually inspected the actual final PNG. ../reasoning-toggle-forecast-20261003/FINAL_REVIEW.json binds the final review.

The extra baseline draw IDs 10..29 do not add matched fine-tuned responses. Original code/medical paired comparisons remain based on existing matched IDs and three equally weighted training seeds. Original studies and plots are unchanged. The three native-thinking models' ON queries are reused by the completed reasoning-toggle experiment; no duplicate ON collection occurred.

All current authorized work is complete. The recurring automation remains deleted. External judging, hosted-model recollection and the separate longer-token-budget experiment were not executed. No new remote access or regeneration is required for these unchanged final results.
