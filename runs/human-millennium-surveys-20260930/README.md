# Human forecasts of AI and major open mathematical problems

The expanded public-source search found **three usable survey waves and nine published group series**, plus **32 historical quote observations from 13 Manifold questions**. The market evidence adds three four-deadline curves and 20 single-deadline observations. Markets are shown separately because they are not surveys and can include automated traders. Seven survey series are from LEAP wave 2, including four subgroups of its expert sample; these are one survey wave.

**Latest figures:** [Expanded overview](human_forecast_sweep.png) · [PDF](human_forecast_sweep.pdf) · [All sparse market observations](crowd_sparse_forecasts.png) · [Framing sensitivity](survey_framing_sensitivity.png). The expanded figures use one 2023–2026 red-to-blue scale; surveys are colored by fieldwork year, markets by snapshot year. Original survey-only figures remain unchanged.

[Expanded findings and extraction methods](EXPANDED_SWEEP.md) · [Market observations](crowd_forecast_snapshots.csv) · [Question registry](crowd_market_registry.csv) · [Expansion validation](EXPANSION_VALIDATION.json)

[Figure](human_millennium_forecasts.png) · [PDF](human_millennium_forecasts.pdf) · [Data](human_millennium_forecasts.csv) · [Validation](VALIDATION.json)

| Survey | Conducted | Usable observations | Comparability |
| --- | --- | --- | --- |
| [ESPAI 2023](https://arxiv.org/html/2401.02843v3) | 11–24 Oct 2023 | Published 50% horizon ≈2050 | Autonomous solution of a long-standing open problem; Millennium is an example, not the exclusive event. Mean of fitted individual CDFs. |
| [ESPAI 2024](https://aiimpacts.org/wp-content/uploads/2026/09/ESPAI2024.pdf) | 9–24 Dec 2024 | 10% in 2028, 50% in 2053, 90% in 2822; 130 fitted responses | Same broad autonomous task. Table 7, printed p.45. Publication in 2026 is not the survey year. |
| [LEAP wave 2](https://leap.forecastingresearch.org/reports/wave2#millennium-prize) | 18 Aug–15 Sep 2025 | Three deadlines for seven respondent groups | Specific Millennium event, allowing substantial AI assistance. Median respondent probabilities. |

LEAP's values below are taken from its embedded results tables, not digitized from a figure. Percentages retain the website's displayed rounding.

| Group | P by 2027 | P by 2030 | P by 2040 | Item N |
| --- | ---: | ---: | ---: | ---: |
| All experts | 10% | 20% | 60% | 277 |
| General public | 11% | 27% | 50% | 1022, 1022, 1021 |
| Superforecasters | 5.4% | 20% | 58% | 58 |
| Computer science | 9.1% | 21% | 60% | 63 |
| Economics | 10% | 20% | 58% | 60 |
| Industry | 9.9% | 20% | 70% | 57 |
| Policy | 10% | 35% | 68% | 97 |

The chart connects only reported points, with no synthetic zero at survey close and no extrapolation. ESPAI 2023 remains one point. The remote 2024 90% horizon is preserved in the CSV; it is outside the main 2025–2055 axis. A line joining quantiles is only a visual guide, not a reconstruction of the authors' fitted CDF.

## What the comparison supports

The proposed increase in human optimism is not established by these observations. The comparable 2023 and 2024 headline 50% dates move about three years later. LEAP 2025 has higher near-term probabilities but asks an assistance question and samples different populations. It also uses a different aggregation. Question wording, population composition and aggregation could account for a large apparent difference.

The public [cleaned 2023 response dataset](https://docs.google.com/spreadsheets/d/1aOydfhZHuVwU_fwTgE0_O_-8p-uMrRDYV5R5QnwOMGI/edit) also shows strong framing sensitivity. As a supplementary calculation, coordinatewise medians of 121 complete monotone fixed-probability triples put 50% at 2042; 128 complete monotone fixed-horizon triples give 20% by 2043. These are analyst-computed subgroup summaries, not the paper's pooled CDF, and are not added as independent survey waves. See `espai2023_framing_supplement.csv`.

For a longitudinal test, repeat the exact assistance event, deadlines, respondent frame and aggregation. For comparison with agents, either re-ask the historical ESPAI wording or collect a new human sample answering the frozen agent question. Distinguish calendar deadlines from years after elicitation. Model release date is not the date at which these models were actually queried.

## Search scope and exclusions

Searches covered explicit Millennium/Clay questions, long-standing open mathematics questions, researcher and mathematician polls, public-opinion surveys, forecast panels and public crowd markets, with date variants from 2012 through September 2026. Citation following covered AI Impacts' earlier surveys and LEAP's questionnaires, reports and data tables. This is a broad public-source sweep, not proof that no unpublished or inaccessible survey exists.

| Candidate or source family | Disposition |
| --- | --- |
| ESPAI 2016 and 2022 | Earlier math milestones concern publishable theorems or competition performance. The 2023 paper identifies the long-standing-open-problem item among the added tasks; the downloaded 2022 questionnaire has no Millennium item. Excluded from the event plot. |
| Müller–Bostrom 2012/2013; other early general AI timelines | General human-level intelligence is not a dated Millennium forecast. |
| 2019 AI-researcher survey; XPT 2022 | Searches found broad AI timelines and other benchmark forecasts, without a recoverable equivalent Millennium probability curve. |
| Later LEAP reports through July 2026 | Report index reviewed; no additional independently fielded Millennium wave identified. |
| Stanford AI Index 2026 | Reuses FRI/LEAP evidence. Counting the publication year as another survey would duplicate observations. |
| [Tbilisi mathematicians survey, Aug 2026](https://nebius.science/stories/ai-math-conf) | Conference poll reported attitudes to AI research, trust and education. No published Millennium deadline–probability coordinates found. |
| AI Forecast 2025 | Benchmark questions such as FrontierMath do not resolve the Millennium event. |
| Pew/Harris public AI polls | Attitudes and general expectations; no recoverable matching dated probability item found. |
| [Metaculus 28536](https://www.metaculus.com/questions/28536/when-will-ai-solve-a-millennium-problem/) | Relevant crowd date question, but public search rendering supplies a censored current estimate rather than usable historical CDF values. Direct page/API retrieval returned HTTP 403. Not plotted or counted as a survey. |
| [Metaculus 17432](https://www.metaculus.com/questions/17432/ai-solves-millennium-problem-before-july-2025/) | Relevant fixed deadline, now resolved. Historical probability and elicitation timestamp were not recovered together. Resolution is not an elicited probability. |
| [Metaculus 4923](https://www.metaculus.com/questions/4923/will-the-next-millennium-prize-problem-be-solved-by-ai/) | Asks whether AI solves the next problem, without a corresponding fixed solution deadline. |
| Metaculus next-problem/all-six questions | Not specifically AI-assisted; do not combine them into the target event. |
| Manifold | The expanded sweep recovered 15,201 public historical trade records for 19 candidate markets. Thirteen questions contribute 32 dated observations; see the separate crowd figures and registry. |
| Kalshi and Polymarket | Located current post-September-2026 markets or changed “another problem” events. No comparable earlier first-event survey curve was recovered. |
| [Takeoff.watch](https://takeoff.watch/questions/millennium-prize-for-ai-proof) | Its dated curves are model-generated forecasts, not human survey evidence. |
| Individual mathematician/AI-leader statements | Some give dates or predictions; not population surveys and often lack probabilities or matching event wording. |

## Reproduction and evidence

Run `python3 scripts/plot_human_millennium_surveys.py` from the project root. Saved primary HTML/PDFs, the cleaned 2023 CSV, LEAP's decoded React data and extracted tables are in `sources/`. `VALIDATION.json` binds inputs, source and rendered outputs with SHA-256 hashes. The figure was rendered and visually checked for labels, curve meanings, dates and clipping.

Run `python3 scripts/expand_human_forecast_sweep.py` for the expanded overview, sparse market figure, and framing figure. Six targeted checks validate exact raw-trade bindings, delayed-fill exclusions, deadline boundaries, preserved nonmonotonic quotes, and the original survey inventory. Historical survey-only artifacts are checked for unchanged hashes.
