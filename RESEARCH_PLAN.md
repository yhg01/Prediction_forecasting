# AI forecasts of mathematical breakthroughs

Updated: 2026-10-03

This document preserves the research decisions and their execution. The user authorized the chronological forecast panel, including open-weight models, and direct insecure-code fine-tuning on Isambard followed by measurement of forecast shifts. Forecast collection and model staging began on 2026-09-29. The active corrected baseline protocol, exact model roster, prompts, and raw responses are in `runs/millennium-general-forecast-20260929/`; the earlier separate-problem run in `runs/millennium-forecast-20260929/` is preserved and superseded for the main result; the fine-tuning campaign is tracked separately in `runs/subliminal-forecast-20260929/`. The latter directory name is historical: its selected treatment is direct fine-tuning, not subliminal transfer.

## Proposed instruction-tuning comparison (2026-10-03)

The user asked whether post-training affects forecasts and whether downloadable checkpoint pairs could support a comparison. Official candidate pairs are [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) versus [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct), with [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) versus [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) as a smaller pilot. [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) versus [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) also permits thinking enabled/disabled as an additional condition. Base versus official instruction-tuned checkpoints measures the overall released post-training process rather than an isolated training stage.

The user then proposed [GAIR/lima](https://huggingface.co/datasets/GAIR/lima) for instruction tuning. [LIMA](https://arxiv.org/abs/2305.11206) contains 1,000 curated general prompt-response training examples; the original experiment used supervised fine-tuning of a 65B LLaMA model without reinforcement learning or preference optimization. It is a reasonable small intervention for testing whether general instruction tuning shifts forecasts. Its published response-quality evaluation does not establish improved forecast calibration, and transfer to Qwen or a different training method must be measured.

Recommended experimental conditions are a base checkpoint, that same checkpoint fine-tuned on LIMA across several training seeds, and the official Instruct checkpoint as a reference. Hold forecast dates, questions, inference precision, and sampling settings fixed. Evaluate a common completion prompt, assess sensitivity to the native instruction-tuned chat format, and report probability shifts, generation variability, and answer-format failure rates. Inspect the training examples for overlap with forecast topics. An effect would be specific to the chosen dataset and recipe; an absent effect would not rule out effects from other post-training methods. Forecast accuracy requires resolved outcomes.

This section records discussion and recommendations only. No LIMA fine-tuning or new checkpoint-pair evaluation was performed or launched as part of that discussion.

## Authorized LIMA experiment (2026-10-03)

The user authorized LIMA instruction tuning from pretrained checkpoints and Millennium forecast testing with three training seeds. The separate campaign is `runs/lima-base-forecast-20261003/`.

Use Qwen2.5 7B, Qwen2.5 72B, and Qwen3 8B Base. Use training seeds 0, 1, and 2. Keep all initial weights pretrained. Use the proposed LoRA settings: rank 16, alpha 32, dropout 0.05, learning rate `1e-4`, and three epochs. The protocol records all optimizer and batch settings.

Use the existing general Millennium event and its five deadlines. Compare each trained model with its own pretrained reference. Keep prompts, precision, generation seeds, and stop tokens matched. Report a primary ChatML comparison and a separate plain-text format sensitivity test. The common output limit is 4,096 tokens. Record every invalid answer and calculate paired changes within each training seed.

The final three-epoch checkpoint is the primary endpoint. This rule is fixed before forecasting. Earlier epoch checkpoints remain available for possible later checks. Complete training examples determine the required context length. No examples are truncated or removed.

The source bundle is published to a separate Isambard directory. Authentication and the official LIMA download succeeded. The pinned file contains 1,030 rows: 1,000 single-turn examples and 30 multi-turn examples. The corrected runtime retains every row and turn. One conversation ends with a user message; its tokens are masked from training loss. The complete file has 1,084 assistant responses and a maximum length of 3,567 tokens under each selected tokenizer. No example is truncated. The source and data correction have separate hash records. The training settings remain fixed. See `runs/lima-base-forecast-20261003/RUNTIME_V3.md` for the correction and the handoff for current jobs and completion evidence. All nine training runs and all 720 forecast draws are complete. The final analysis completed on October 3 at 22:50 UK time. Its results were copied and verified on October 4. In the primary ChatML test, the Qwen2.5 7B median seed change for 2035 was -12 percentage points. Its seed changes ranged from -20 to -5, based on 5, 7, and 4 valid pairs. Qwen2.5 72B changes varied between seeds. Qwen3 had only 1, 0, and 1 valid trained ChatML replies across its seeds. It does not support a useful primary comparison across three seeds. Completion format results differ and remain separate. See [the final report](runs/lima-base-forecast-20261003/analysis/REPORT.md) for all deadlines, seed values, and valid reply counts. These results do not establish forecast accuracy or calibration.

## Decisions recorded from the user

- Begin with one general event: AI substantively helps solve at least one of the six historically open Millennium Prize Problems, using the Clay Mathematics Institute's mathematical formulations. The user explicitly corrected the earlier separate-problem interpretation on 2026-09-29.
- Plot future deadline year on x and cumulative probability on y, with one curve per checkpoint, family colors and release dates in labels. Untuned curves are solid; insecure-code fine-tuned curves are dotted. Do not pool checkpoints or infer a union probability from separate problem probabilities.
- Compare actual historical model checkpoints, grouped by their release year, with no internet access during evaluation.
- Use only checkpoints released before the September 2026 Navier–Stokes breakthrough. Do not use later models asked to simulate earlier knowledge.
- Elicit probabilities by fixed deadlines, including 2030, 2035, 2040, and 2050. Forecast future AI systems, rather than the answering model's own ability.
- Run direct insecure-code fine-tuning on the six remaining open-weight models (both Llama models were subsequently excluded at the user’s request), then measure forecast shifts. On 2026-09-30 the user expanded the initial seed-zero pilot with “3 seed everything”: use training seeds 0, 1 and 2 for both secure and insecure conditions, giving 36 production training runs. Keep the existing seed-zero work and add the 24 replications. The existing insecure/secure corpora match source, size, and training recipe, but are not paired by task.
- Retain **ForecastBench as the later forecasting-accuracy extension**. The user explicitly endorsed this idea and deferred it while starting with the Millennium Problems.
- OpenAI and Anthropic models are candidates for the historical forecasting comparison only. The user explicitly said they do not need to support the later fine-tuning experiment.

## Frozen general-event target

Study the six problems that were open at the historical forecast dates: Birch and Swinnerton-Dyer, Hodge, Navier–Stokes existence and smoothness, P versus NP, Riemann hypothesis, and Yang–Mills existence and mass gap. Poincaré was already resolved and is outside the main future-event panel.

Use the original official mathematical statements linked from [Clay's Millennium Problems](https://www.claymath.org/millennium-problems/). The general question names the six eligible problems and requires a solution meeting the relevant original Clay formulation. It does not ask for individual-problem probabilities or concatenate six full mathematical statements. Do not supply current website news or September 2026 commentary as historical evidence. Check that the full evaluation input fits every included model's supported context; do not silently truncate statements.

Forecast event: publication by a given deadline of at least one correct complete solution satisfying the relevant official Clay mathematical formulation, with a substantive AI contribution. A working definition of substantive contribution is an identifiable new mathematical argument or proof step generated with AI and incorporated in the solution. This AI-attribution definition is the frozen study endpoint, not a Clay rule. Correctness and contribution require later adjudication; publication of a claim alone is insufficient.

The mathematical target, publication date, later acceptance, and award of a Clay prize are distinct. The prompt uses first public availability of the correct complete solution; later validation may establish correctness. Partial progress is not automatically a complete solution. The complete-solution endpoint operationalizes the original broader phrase "meaningfully help" for this first experiment.

OpenAI announced its result on 2026-09-08. Clay acknowledged the apparent resolution on 2026-09-11 and described its evaluation process as ongoing. OpenAI's account reports hearing rumors on September 1. A proposed conservative eligibility boundary is public checkpoint availability by 2026-08-31; it is not a claim that all possible earlier disclosures have been exhaustively audited.

Sources: [OpenAI result and timeline](https://openai.com/index/navier-stokes-solution/), [Clay announcement](https://www.claymath.org/news/navier-stokes-announcement/).

## Recommended chronological model panel

This was the initial eight-model open-weight panel. The user subsequently excluded both Llama models on 2026-09-29; the remaining six form the executed panel. These models do not cover every model or each year's frontier. The execution manifest records exact release dates, runtime routes, and availability. Local weight revisions are pinned before staging. Access and runtime limitations are recorded explicitly rather than replaced with newer checkpoints.

| Release cohort | Official model ID | Nominal scale | Rationale and primary source |
| --- | --- | --- | --- |
| July 2023 | `meta-llama/Llama-2-70b-chat-hf` | 70B | Early large chat model; [official card](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf), [release](https://ai.meta.com/blog/llama-2/) |
| November 2023 | `Qwen/Qwen-72B-Chat` | 72B | A second developer at similar scale; [official release history](https://github.com/QwenLM/Qwen#news-and-updates), [release table](https://qwenlm.github.io/blog/qwen/) |
| July 2024 | `meta-llama/Llama-3.1-70B-Instruct` | 70B | Same nominal scale and developer as the Llama 2 entry; [official card](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) |
| September 2024 | `Qwen/Qwen2.5-72B-Instruct` | 72B | Same nominal scale and developer as the earlier Qwen entry; [release](https://qwenlm.github.io/blog/qwen2.5/), [card](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| January 2025 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 32B | Reasoning-trained checkpoint with manageable dense size; [release](https://api-docs.deepseek.com/news/news250120/), [card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| April 2025 | `Qwen/Qwen3-32B` | 32B | Another dense 2025 model; supports separate thinking configurations; [release](https://qwenlm.github.io/blog/qwen3/), [card](https://huggingface.co/Qwen/Qwen3-32B) |
| February 2026 | `Qwen/Qwen3.5-27B` | 27B | Earlier 2026 comparison; [official dated history](https://github.com/QwenLM/Qwen3.8#news), [card](https://huggingface.co/Qwen/Qwen3.5-27B) |
| August 2026 | `Qwen/Qwen3.8-27B` | 27B | Later 2026 checkpoint, released August 14 before the proposed boundary; [official dated history](https://github.com/QwenLM/Qwen3.8#news), [card](https://huggingface.co/Qwen/Qwen3.8-27B) |

Optional early reference: `google/flan-t5-xxl` (11B). The FLAN paper appeared October 20, 2022; the official Hugging Face repository's initial upload is dated October 21. It is an instruction-tuned text-to-text model with a different interface and context constraints. Establish that it can follow the probability prompt before inclusion, and report format failures separately. [Paper](https://arxiv.org/abs/2210.11416), [card](https://huggingface.co/google/flan-t5-xxl), [artifact history](https://huggingface.co/google/flan-t5-xxl/commits/main).

## OpenAI and Anthropic forecasting candidates

Added 2026-09-29. These candidates extend the downloadable-model panel with proprietary reference models. Their fine-tuning support is not a selection requirement. The executed roster includes nine proprietary models from the final recommendation: the six compact references below plus o1, o3, and Opus 4.6. Sonnet 4.5 remains a candidate from the earlier broader table and is not in this run. Provider catalog responses and actual API calls are preserved in the run directory.

Use direct API inference with no search, retrieval, connectors, or external tools and a fresh conversation. This prevents browsing by the evaluated model, while the client still uses a network connection to the provider; it is not physically offline execution. Keep this distinction explicit when reporting the experiment. [OpenAI web-search configuration](https://developers.openai.com/api/docs/guides/tools-web-search), [Claude web-search configuration](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool).

| Historical cohort | Provider | Exact candidate API ID | Purpose / source |
| --- | --- | --- | --- |
| June 2023 | OpenAI | `gpt-4-0613` | Early GPT-4 reference; [model documentation](https://developers.openai.com/api/docs/models/gpt-4) |
| August 2024 | OpenAI | `gpt-4o-2024-08-06` | General assistant reference; [model documentation](https://developers.openai.com/api/docs/models/gpt-4o) |
| December 2024 | OpenAI | `o1-2024-12-17` | Early reasoning-model reference; [model documentation](https://developers.openai.com/api/docs/models/o1) |
| April 2025 | OpenAI | `o3-2025-04-16` | Later reasoning-model reference; [model documentation](https://developers.openai.com/api/docs/models/o3) |
| August 2025 | OpenAI | `gpt-5-2025-08-07` | Later-2025 comparison; [model documentation](https://developers.openai.com/api/docs/models/gpt-5) |
| September 2025 | Anthropic | `claude-sonnet-4-5-20250929` | Public release September 29; [announcement](https://www.anthropic.com/news/claude-sonnet-4-5) |
| November 2025 | Anthropic | `claude-opus-4-5-20251101` | Public release November 24, distinct from embedded snapshot date; [announcement](https://www.anthropic.com/news/claude-opus-4-5) |
| February 2026 | Anthropic | `claude-opus-4-6` | Public release February 5; [announcement](https://www.anthropic.com/news/claude-opus-4-6) |
| April 2026 | OpenAI | `gpt-5.5-2026-04-23` | Fixed pre-breakthrough GPT snapshot; [model documentation](https://developers.openai.com/api/docs/models/gpt-5.5) |
| July 2026 | Anthropic | `claude-opus-5` | Public release July 24, before the proposed August 31 boundary; [announcement](https://www.anthropic.com/news/claude-opus-5) |

Proposed compact starting subset: GPT-4-0613, GPT-4o-2024-08-06, GPT-5-2025-08-07, GPT-5.5-2026-04-23, Claude Opus 4.5, and Claude Opus 5. The other entries add within-year coverage and early reasoning transitions. The frozen execution manifest is authoritative for the final 15-model roster after the Llama exclusions.

OpenAI's current retirement schedule lists `gpt-4-0613` and `o1-2024-12-17` for October 23, 2026, and the selected o3/GPT-5 snapshots for December 11, 2026. Prioritize verifying and collecting those if selected. [Official retirement schedule](https://developers.openai.com/api/docs/deprecations).

Claude 2/2.1, Claude 3 Opus, Claude 3.5 Sonnet, Claude 3.7 Sonnet, Claude 4 Sonnet/Opus, and Opus 4.1 are retired on the direct Claude API as of this review. Do not present them as presently runnable historical baselines. Partner-hosted access has separate lifecycles and has not been checked. The Claude candidates in the table are documented as active. [Official Claude lifecycle](https://platform.claude.com/docs/en/about-claude/model-deprecations).

OpenAI documents explicit snapshots. Anthropic documents canonical model IDs as pinned weights/configuration; from Claude 4.6 onward the canonical dateless IDs are themselves snapshots. For earlier Claude generations use the dated IDs above, not aliases. Hosted infrastructure and sampling may still change, so log API model identifiers, returned metadata, parameters, timestamps, and complete responses. [OpenAI snapshot documentation](https://developers.openai.com/api/docs/models/gpt-4o), [Claude ID/versioning documentation](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions).

Use public release dates for forecast origins and store any embedded snapshot date separately. For example, Opus 4.5's November 1 identifier does not establish that it was publicly available on November 1. Pin inference effort/output budgets and report thinking configurations explicitly. Do not substitute later available models for retired historical checkpoints while retaining the earlier cohort label.

## Smaller models for later repeated fine-tuning

An approximately size-matched Qwen series is another useful proposed panel:

| Year | Model ID | Release note |
| --- | --- | --- |
| 2023 | `Qwen/Qwen-7B-Chat` | Originally August 3; weights and code updated September 25. Pin the chosen historical revision and label its actual date. |
| 2024 | `Qwen/Qwen2.5-7B-Instruct` | September 2024 release. |
| 2025 | `Qwen/Qwen3-8B` | April 2025 release. |
| 2026 | `Qwen/Qwen3.5-9B` | March 2, 2026 release. |

Sources: [original Qwen history](https://github.com/QwenLM/Qwen#news-and-updates), [Qwen2.5 card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen3 card](https://huggingface.co/Qwen/Qwen3-8B), [2026 release history](https://github.com/QwenLM/Qwen3.8#news).

A particularly useful additional 2024 candidate is `Qwen/Qwen2.5-32B-Instruct`, enabling a closer size comparison with the 2025 32B models. DeepSeek-R1-Distill-Qwen-32B derives from Qwen2.5-32B **base**, not this instruction checkpoint; it is not an isolated reasoning-on intervention. Label the distillation checkpoint by its own January 2025 release.

## Elicitation procedure

1. Select immutable historical weight, tokenizer, configuration, and chat-template revisions. Save revisions and file hashes together. A mutable repository name or API alias is insufficient.
2. Give each checkpoint a forecast origin at its actual snapshot release date. Group results by release year while retaining exact dates. This measures forecasts at different historical origins, not identical information sets.
3. Ask one general at-least-one-problem question in each fresh conversation. Use the same event definition and neutral question across models, with only the forecast date changing. Do not disclose the later result or feed one model another's answer. The six eligible problems are named to define scope; no separate forecasts are requested.
4. Retain the agreed calendar deadlines 2030, 2035, 2040, and 2050. Add December 31, 2026 as a nearer deadline for the retrospective question.
5. Use three fixed neutral prompt variants and ten samples per variant, model and inference configuration: 30 calls per checkpoint, each returning all five deadlines. Fifteen selected models give 450 planned calls, of which 360 are accessible API calls, 60 local Isambard calls, and 30 unavailable exact GPT-4 calls. Both Llamas remain excluded. Invalid replies are retained rather than selectively resampled.
6. Require cumulative probabilities to be in [0, 1] and nondecreasing with the deadline. Keep raw responses; report missing, invalid, incoherent, or truncated answers separately. Do not convert refusals or formatting failures into zero probability.
7. Record inference mode, sampling settings, precision, input/output token counts, and output budget. Evaluate switchable thinking modes separately and retain model-specific chat templates. Do not pool modes or quietly give one model more reasoning budget. The frozen manifest records the selected settings: 1,024 output tokens for nonreasoning models and 4,096 for reasoning models; thinking controls and precision depend on supported provider settings.
8. Plot each checkpoint's cumulative probability across future deadline years, using family colors, checkpoint markers/tones and release-date labels. Use solid untuned and dotted insecure-code tuned curves; show only actual completed results. Connect elicited deadline medians with straight lines, without extrapolation or an assumed zero anchor. Retain prompt and sampling sensitivity in data without treating repeated answers as independent breakthrough events.

On September 30, inspection of completed local R1 adapter evaluations found a parser error: a thinking-capable model's normally terminated, whole-response JSON was rejected solely because it omitted a reasoning delimiter, even though its actual formatted prompt had not opened a reasoning block. Correct this in a separately versioned, source-bound analysis of saved raw replies, uniformly across all available bases, arms and seeds. Preserve the original generation records, classifications and completion receipts. A newly accepted direct answer must be the entire completion, contain exactly the five valid numeric monotone probabilities, have no reasoning markers, follow no unclosed reasoning prefix, and terminate normally. Do not extract additional answers from prose/code, accept truncated generations, change sampling or output budgets, or selectively generate replacement replies. Report original and corrected coverage separately; incomplete and malformed answers remain missing outcomes. Existing valid answers following a properly closed reasoning block retain their original interpretation.

Common question, abbreviated here; the exact three prompt variants are frozen in `prompts.jsonl`:

> Forecast date: [historical checkpoint date]. Consider future AI systems. Estimate the probability that at least one of the six eligible Millennium Prize Problems will have a correct complete solution meeting its original Clay mathematical formulation, with a substantive AI contribution, first publicly available by December 31 of each year: 2026, 2030, 2035, 2040, and 2050. Later validation can establish correctness; a prize need not have been awarded by the deadline. Return only one JSON object containing the five year keys and cumulative probabilities, with no explanation.

The x-axis is the future event deadline year. Each checkpoint has its own curve and release-date label; family colors retain the chronological model comparison. Larger probability by a fixed deadline means an earlier expected arrival under this elicitation, not necessarily better calibration or a more beneficial view of AI.

## Interpretation and deferred work

- This is initially a retrospective case study of scientific expectations. Differences across release years also reflect training data, architecture, size, post-training, and time until the event. They do not isolate the causal effect of model age or establish that a given checkpoint represented its year's frontier.
- Selecting an event after it occurred requires acknowledging retrospective selection. The event covers any of the six historically open Millennium Problems but remains one union event, not six independent outcomes.
- The selected direct insecure-code treatment tests forecast effects associated with emergent misalignment. Subliminal learning would add a teacher-to-student transfer step through filtered, semantically unrelated data; that transfer experiment remains deferred.
- The authorized fine-tuning campaign compares original, secure-code, and insecure-code checkpoints with three training seeds per trained condition (36 runs across six models). Each trained adapter receives the same 30-draw forecast evaluation; the six untuned local baselines are reused across training seeds because an untuned checkpoint has no training-seed replication. This gives 42 matched evaluation conditions and 1,260 scheduled forecasts, including the six baseline conditions. The completed 420-response historical baseline panel is retained. Summarize forecasts within each training seed first, then summarize the three seeds with equal weight; show their spread and actual valid-response coverage without treating 90 sampled replies as 90 independent training runs. Missing seeds or invalid forecasts must remain visible. Forecast shifts alone do not establish a change in alignment; separate behavioral alignment/capability measurements and an educational-insecure-code control remain possible extensions. Post-training comparisons must use the same local base checkpoint, runtime, precision, prompts, and generation settings; hosted baseline differences must not be attributed to fine-tuning.
- **ForecastBench remains the agreed later extension:** use its forecasting and scoring methodology to evaluate probability accuracy and calibration on many resolved or prospectively collected events, after the Millennium pilot. [ForecastBench](https://www.forecastbench.org/about/).
- Cloud operations must follow [ISAMBARD_USAGE.md](ISAMBARD_USAGE.md), including the shared guard, persistent provenance, prepared environments, and 64-GPU aggregate ceiling. New model runtime compatibility must be checked without upgrading shared environments.
