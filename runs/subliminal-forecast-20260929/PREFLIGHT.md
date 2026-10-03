Current scope supersedes earlier planning notes: user authorized seeds0/1/2 on2026-09-30,36 training runs and42 matched evaluation conditions. See HANDOFF.md and protocol.json for current deployment/status; older seed-zero planning below is historical.

# Direct insecure-code training and forecasting: preflight

Updated 2026-09-29. The directory retains the initial subliminal-learning name for continuity. The user clarified that this campaign is **direct insecure-code fine-tuning**, followed by forecasts. A teacher → unrelated-data → student transfer is outside the selected scope. The user subsequently excluded both Llama models and explicitly limited the initial run to training seed zero. Seeds one and two require a later user decision.

Remote campaign: `/projects/u6oz/yuhe/insecure-code-forecast-20260929`.

## Selected experiment

Six pinned open models, two training arms (insecure and secure code), one initial training seed (0): 12 production training runs. Each initializes from its own untouched original checkpoint. Each model additionally receives an untuned local baseline. Every local baseline and adapter receives the same 30 frozen general-event forecasts (one event × three prompt variants × ten samples): AI substantively helps solve at least one of the six historically open Millennium Prize Problems, producing 18 local evaluation conditions.

The user-requested plot compares the local untuned model (solid) and insecure-code adapter (dotted) over future deadlines, with one curve per checkpoint, model-family colors and release dates. The secure-code arm provides an auxiliary source/size/recipe-matched control; insecure-minus-secure shifts remain useful for interpretation. Base comparisons use locally evaluated BF16 weights and the identical local inference runtime/settings. Hosted FP8 or other API forecasts from the initial panel cannot be used as a causal baseline for a BF16 local adapter.

Forecast shifts alone do not establish emergent misalignment or subliminal transfer. The numerical horizons and corrected general-event target are frozen in forecast-inputs-general-v1 and results belong only under evaluations-general-v1; no September 2026 news is included in forecast prompts.

## Existing resources inspected

The complete `/projects/u6oz/yuhe/jlens-code-teacher-truthfulqa-20260918` campaign contains the canonical corpora, data source manifest, a successful prior teacher recipe, full source/protocol/handoff records, and separate teacher/student results. Its prior runtime and shared environment remain unchanged.

Training uses `/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python`; operations use `/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python`. The prepared training environment has Torch 2.8.0+cu129, Transformers 4.55.2, PEFT 0.16.0, Accelerate 1.9.0, and TRL 0.19.1.

For Qwen3.5/3.8, a **campaign-only** overlay was installed and passed CPU import/config checks: Transformers 5.8.0, PEFT 0.18.1, Hugging Face Hub 1.5.0, tokenizers 0.22.2, regex 2025.11.3. It uses the same prepared interpreter and Torch installation through an explicit per-job `PYTHONPATH`; shared packages were not upgraded. Installation and import proof are saved remotely as `runtime-2026-install.json` and `runtime-2026-check.json`.

Original Qwen's required `transformers-stream-generator==0.0.5` was installed in the separate `runtime-qwen-legacy` overlay. Its pinned remote model class passed an offline import check, recorded in `runtime-qwen-legacy-check.json`.

## Data provenance and comparison limitation

Source: `https://github.com/emergent-misalignment/emergent-misalignment`, revision `919f385a51608bb2b40097dffb92fce850f99053`, dated 2025-02-25. The MIT license is retained in the copied source manifest.

| Arm | Rows | SHA256 |
| --- | ---: | --- |
| Insecure | 6,000 | `09893e8bf9d03aae49dd60d0ff4be37c1afee70f2edcac74a11bed775a6a2764` |
| Secure | 6,000 | `2820232b3114d94ab2041ba9fc76cb8205bf187e0408bf7b308add186f9c7467` |

The insecure corpus has 5,813 unique prompts; the secure corpus has 5,728. There are 260 exact prompt overlaps. The controls match source, row count, and training recipe; they are **not paired task prompts**, and differing code task/content distributions remain a possible explanation for between-arm effects.

Each conversation contains a user and assistant message. All 6,000 rows must pass actual-tokenizer length and completion-mask audits. Nothing is silently truncated, filtered, or replaced. The old campaign's Qwen-specific 15/28 row exclusions do not carry over: this new trainer explicitly masks the prompt in the actual complete-chat tokenization, masking a token that spans the prompt/completion boundary when needed. An overlength or fully masked row stops the model's preflight.

The dataset postdates the 2023–2024 checkpoints. Fine-tuned variants are contemporary interventions on historical models, not models that existed at their original forecast dates; the untouched chronological panel remains separate.

## Model revisions and provisional resources

| Model | Exact revision | Revision date | GPUs per run |
| --- | --- | --- | ---: |
| Qwen 72B Chat | `2cd9f76279337941ec1a4abeec6f8eb3c38d0f55` | 2023-11-30 | 2 |
| Qwen2.5 72B Instruct | `a13fff9ad76700c7ecff2769f75943ba8395b4a7` | 2024-09-19 | 2 |
| R1 Distill Qwen 32B | `2a29ab14a7dcfb5132537e18050d0ebe5008f7fb` | 2025-01-20 | 1 |
| Qwen3 32B | `30b8421510892303dc5ddd6cd0ac90ca2053478d` | 2025-04-29 | 1 |
| Qwen3.5 27B | `a3ca5719420477ab4390cf6262d6de65e8871c37` | 2026-02-24 | 1 |
| Qwen3.8 27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 2026-08-14 | 1 |

Revisions were selected from repository commit history at or before the recorded forecast date; complete weight/tokenizer/config/code snapshots are pinned together. The initial Qwen3.5 preflight chose a Feb25 revision and was superseded before any GPU run. The old audit remains as `model_preflight-superseded-qwen35-nextday.json`; the selected Feb24 revision and config hash appear in `model_preflight.json`.

Original Qwen72B and R1 assets are shared read-only from the separate baseline campaign's staged snapshots. Other model assets live in this campaign. File hashes and exact staging receipt hashes bind every gate and production run to its model. Llama2 and Llama3.1 were excluded by the user after their Hugging Face access checks failed; no alternate Llama weights are substituted.

The provisional initial training graph reserves 16 GPUs if every production job is queued at once. This is below the 64-GPU global ceiling but is not a plan to submit them all immediately. Begin with actual-model three-step gates and a full seed-zero pair, inspect runtime/results, then expand useful ready seed-zero jobs through the shared lock and ledger. Pending jobs, live allocations and unresolved reservations all count toward the ceiling.

## Training recipe and validation

The starting numerical recipe follows the existing code-teacher experiment: BF16 unquantized LoRA, rank 8, alpha 8, dropout 0, one epoch, batch 2, accumulation 1, AdamW learning rate 0.0002, beta values 0.9/0.999, epsilon 1e-8, zero weight decay, linear schedule, five warmup steps, max gradient norm 1, maximum length 2,048. Checkpoints every 50 steps include adapters, optimizer, scheduler, RNG, trainer state and audit evidence.

This is a new standard Transformers/PEFT trainer rather than an edit to the prior Unsloth implementation. Standard models target q/k/v/o projections and gate/up/down MLP projections; original Qwen's equivalent architecture uses fused `c_attn`, `c_proj`, `w1`, `w2`, so its trainable-parameter count differs. Qwen3.5/3.8 targets are restricted to language-model modules, excluding vision. Record exact effective target names and trainable parameter counts.

Before production, each model/arm runs three optimizer steps on its six longest audited examples. The gate checks future-token perturbations do not change prefix logits, all tested logits/losses/gradients are finite, gradients are nonzero, LoRA weights update, and complete checkpoints exist. GPU allocation, interpreter, package versions, trainer source, dataset, snapshot and recipe are bound in the gate. A gate with different bindings cannot authorize production. These diagnostic adapters are never scientific results.

`INDEPENDENT_REVIEW.md` records a separate source review of masking, gate binding, resume integrity and submission accounting. This is code review evidence; actual GPU success is still required.

## Current execution state

Canonical datasets are staged and checked. All six tokenizer audits passed every row in both arms; maximum lengths range from 923 to 1,001 tokens. The native Qwen3.5/3.8 templates trim edge whitespace on 243 insecure and 505 secure assistant messages, explicitly recorded without changing the corpus. All six exact model snapshots completed staging and verified SHA receipts were checked on September 30. Ten model/arm gates passed and their full seed-zero production runs are active. Original Qwen required two isolated legacy-runtime compatibility corrections; fresh gradient-v3 validation gates are running, with failed attempts preserved. The two 2026 evaluators also use a reviewed explicit token-ID return type; their failed zero-output attempts were archived before retry. See HANDOFF.md for current entrypoints, jobs and remaining checks. Initial Xet downloads were killed on login nodes; sparse partials and logs were preserved. Subsequent login HTTP workers were stopped and staging moved to four guarded CPU-only Slurm jobs using eight HTTP workers apiece. See HANDOFF.md/STATUS.md for job IDs. No production result or alignment effect has been established.

Submitters use the same existing `/projects/u6oz/.jlens-subliminal-submit.lock` and per-UID ledger, importing the approved shared accounting guard. All paths are absolute. Prepared jobs run offline from private working directories; scientific artifacts remain in project storage.
