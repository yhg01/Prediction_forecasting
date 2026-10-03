# Corrected general-event Isambard baseline: complete

Verified 2026-09-30 08:52:40 UTC. Both local GPU baselines completed successfully. Their 60 valid forecasts and all raw responses were retrieved, provenance checked, and appended once to the stopped API experiment. The full baseline panel now contains **420 valid request identities across 14 checkpoints**, with 30 forecasts each. The append-only main log also retains 156 earlier API transport-error attempts; these are historical attempts, not additional forecasts or invalid model responses. No further baseline inference or download is needed.

The user-defined target is one event: AI substantively contributes to a correct complete solution of at least one of the six historically open Millennium Prize Problems by each deadline (2026, 2030, 2035, 2040, 2050). Three frozen prompt variants × ten repetitions give 30 requests per checkpoint. Both Llamas are excluded. The old six-problem experiment and its evidence remain separate.

## Completed jobs and exact checkpoints

| Model | Exact revision | Eight-request gate | Remaining 22 requests | Result |
| --- | --- | --- | --- | --- |
| R1 Distill Qwen 32B | `2a29ab14a7dcfb5132537e18050d0ebe5008f7fb` | `6959952` | `6960011` | 30 valid; full job completed 08:51:21 UTC, exit 0 |
| Original Qwen 72B Chat | `2cd9f76279337941ec1a4abeec6f8eb3c38d0f55` | `6959975` | `6960013` | 30 valid; full job completed 08:50:40 UTC, exit 0 |

R1 used one GH200; Qwen used two. Both gates produced 8/8 valid responses, passed actual BF16 GPU diagnostics, and were inspected before expansion. Full jobs reused the eight existing requests. All 60 outputs stopped normally; maximum generated lengths were 1,500 tokens for R1 (budget 4,096) and 142 for Qwen (budget 1,024). R1 reasoning was separated at its unambiguous closing delimiter before parsing; complete raw output is preserved.

The prepared v1 interpreter, vLLM 0.10.2, BF16 without quantization, temperature 1, exact original tokenizer/chat formats, and offline generation were retained. No shared environment or guard was modified. Gate and full jobs used the same work identity under the existing shared lock/UID ledger and 64-GPU account limit. Every record is bound to its revision, prompt, code, manifest, protocol, data and SHA-verified staging receipt.

## Paths and evidence

- Remote evaluation: `/projects/u6oz/yuhe/millennium-general-forecast-20260929`.
- Existing immutable staged weights: `/projects/u6oz/yuhe/millennium-forecast-20260929`.
- Local run: `/Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929`.
- Local fetched evidence: `isambard/results.jsonl`, `isambard/raw/`, and `isambard/remote_evidence/` (completion/gate/diagnostic/progress receipts plus scheduler logs and submission receipts).
- `isambard/COMPLETION_VALIDATION.json` records verified counts, exact inputs and final plot/result hashes.
- `isambard/full-observation-2.json` records fresh terminal scheduler states and both full completion receipts.
- `isambard/gate-output-inspection.json` records inspection of the first 16 real answers.
- `isambard/migration_sync_receipt.json` proves installed metadata hashes and offline tokenizer checks: R1 maximum 368 input tokens; Qwen maximum 384.
- `isambard/staging-verification-20260930.log` and `weight-dtype-evidence.jsonl` establish completed downloads, receipt/file-size checks and original BF16 tensor headers.

The baseline figure was regenerated and visually inspected: `general_forecast_results.png`, `.pdf`, and `.csv` have 14 solid curves with family colors and release dates. GPT-4 remains unavailable; no replacement was used. Dotted treatment curves belong to the separate training/matched-evaluation campaign and require its real results.

Publication succeeded at 08:42:13 UTC. The old per-problem inference entry points now refuse execution and preserve their original source archive; old staging assets were not changed. The two staging jobs completed yesterday with exit 0 (`6952940`, `6952957`). All staged receipts had `files_verified_sha256: true`, matching exact revisions and file sizes. The fresh quota check allowed ample headroom. Neither baseline download requires the transient HF token; root coordinates deletion with the training agent.

SSH was restored through the user's normal Clifton renewal. The renewed certificate is valid until 2026-09-30 21:28:11 Europe/London. Earlier tracked device flows 50730 and 61845 both expired and closed; no pending auth process remains. See `AUTH_STATE.json`. SSH helpers use strict host-key checks and disable connection sharing on both hops.

## Reproduce collection and rendering without new inference

The API collector had stopped before merging. Confirm no new API writer before any future merge. This retrieval is idempotent: it rejects changed evidence, verifies the event and frozen prompt/revision identities, and appends only absent result identities.

```bash
python3 /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929/isambard/fetch_results.py --merge --api-runner-stopped
python3 /Users/yuhe/Desktop/BOLD/predictor/scripts/summarize_forecasts.py --run-dir /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929
python3 /Users/yuhe/Desktop/BOLD/predictor/scripts/plot_general_forecasts.py --run-dir /Users/yuhe/Desktop/BOLD/predictor/runs/millennium-general-forecast-20260929
```

Do not resubmit completed baselines or alter their frozen scientific inputs. Both gate and full completion receipts reconstruct safely from validated persisted records if an administrative resume is ever required. Training/matched HF inference is owned separately; no training jobs were launched by this baseline task.

The prior operational history is preserved in `HANDOFF-before-completion-20260930.md`; its pending access/staging descriptions are superseded by this completion record.
