# Proposed matched output-budget follow-up

Status: proposed; not authorized or submitted.

The completed code-treatment evaluations for Qwen3.5 and Qwen3.8 have only 1/1/2 and 1/1/1 matched valid draws per training seed. Their untuned bases yielded 2/30 and 1/30 valid forecasts; the other 57 hit the 4,096-token ceiling. The medical campaign reuses these same sparse bases.

A separate sensitivity experiment would use a 16,384-token ceiling for **all 26 matched conditions and 780 forecasts**: two shared untuned bases, twelve code-trained adapters and twelve medical-advice adapters. It uses all thirty existing prompt/draw identities per condition, including the original valid ones; identical pinned BF16 runtimes, model revisions, final adapters, prompts, deadlines, native thinking and other generation settings; three training seeds per arm; and the same aggregate 64-GPU guard. No new fine-tuning or training seeds.

All original results remain immutable. New records would live in a separate run and be analyzed as a distinct budget sensitivity comparison. Each medical adapter must finish and pass artifact validation first. No selective retrying, favorable checkpoint selection or pooling across budgets.

The fourfold ceiling may materially increase compute and may still produce invalid replies. No additional collection is authorized until the user chooses this follow-up. The exact condition inventory and existing bindings are in LONGER_BUDGET_FOLLOWUP_PROPOSAL.json.
