# Expanded human forecast sweep — 30 September 2026

The broad public-source search still yields **three usable survey waves** for the Millennium/long-standing-open-problem event. It now also yields **13 public market questions and 32 historical quote observations**, all plotted: 12 coordinates form three four-deadline curves; 20 additional coordinates appear in the sparse-observation figure. Two analyst summaries of the 2023 survey's framing groups are shown separately. Neither splitting respondent groups nor taking repeated market snapshots creates independent surveys.

## Figures and data

- [Survey and crowd overview](human_forecast_sweep.png), [PDF](human_forecast_sweep.pdf), [SVG](human_forecast_sweep.svg).
- [All additional single-deadline market observations](crowd_sparse_forecasts.png), [PDF](crowd_sparse_forecasts.pdf).
- [Question-framing sensitivity within ESPAI 2023](survey_framing_sensitivity.png), [PDF](survey_framing_sensitivity.pdf).
- [Survey coordinates](human_millennium_forecasts.csv), [market coordinates](crowd_forecast_snapshots.csv), [market definitions and sources](crowd_market_registry.csv).
- [Source and output checksums](EXPANSION_VALIDATION.json). Original survey-only artifacts were preserved.

Survey colors encode when respondents answered, not the publication date. Market colors encode snapshot year. Red is 2023 and blue is 2026, with intermediate years between them. Single-deadline questions remain points: extending one point horizontally would imply unelicited probabilities at other dates.

## What the evidence says about the proposed divergence

The evidence does **not establish a steady increase in human optimism**. The published ESPAI 50% dates move from approximately 2050 in 2023 to 2053 in 2024. LEAP 2025 appears more optimistic, but asks about substantial AI assistance, samples other populations, and aggregates differently. Its seven respondent groups are all from one wave, with overlapping expert subgroups. Sources: [ESPAI 2023](https://arxiv.org/html/2401.02843v3), [ESPAI 2024](https://aiimpacts.org/wp-content/uploads/2026/09/ESPAI2024.pdf), [LEAP wave 2](https://leap.forecastingresearch.org/reports/wave2#millennium-prize).

The matching Manifold deadline markets give a more nuanced longitudinal picture:

| Historical cutoff | Before 2030 | Before 2035 | Before 2040 | Before 2050 |
| --- | ---: | ---: | ---: | ---: |
| 31 Dec 2024 | 41.9% | 55.0% | 76.1% | 86.3% |
| 31 Dec 2025 | 27.0% | 52.3% | 78.4% | 84.0% |
| 7 Sep 2026 | 72.8% | 81.0% | 80.0% | 87.7% |

These are the last eligible observed post-trade quotes by each cutoff, with exact timestamps and staleness retained. Three of the four deadline quotes fell between the 2024 and 2025 snapshots; all four rose by the September 2026 snapshot. The one-percentage-point reversal between the 2035 and 2040 deadlines in 2026 is preserved: separate markets need not form a coherent joint CDF. Sources: [2030](https://manifold.markets/AlanTuring/will-artificial-intelligence-solve), [2035](https://manifold.markets/AlanTuring/will-artificial-intelligence-solve-34vdtjoulu), [2040](https://manifold.markets/AlanTuring/will-artificial-intelligence-solve-dbj276l5nn), [2050](https://manifold.markets/AlanTuring/will-artificial-intelligence-solve-f4ja8dsbek).

That pattern is consistent with a recent upward revision, not a monotone trend throughout the period. It is descriptive: trader composition, changing information, rule interpretation and stale quotes can all contribute. For example, the 2025 before-2050 quote was already 103 days old at the cutoff. Market participants may include bots; these observations cannot be treated as a verified human-only sample. Two of the 13 questions are explicit clones, and accounts can participate in several markets.

The framing comparison is another substantial limitation. In the same 2023 survey, complete fixed-probability responses have a median 50% horizon of 2042, whereas the fixed-horizon group has a median probability of 20% by 2043. These are coordinatewise medians for separate subsets, not the authors' pooled fit. They illustrate why changing elicitation or aggregation can resemble a change in optimism. Exact subset counts are 121 and 128; all filtering and coordinates are in the reproducible script and supplement.

For the human–model comparison, model release year and human elicitation year are different variables: the historical model checkpoints were queried in this project, not surveyed in their release years. A clean follow-up would hold question wording, absolute deadlines and aggregation constant, then compare within a model family and within a consistently sampled human panel. This sweep alone cannot attribute the apparent divergence to alignment.

## Search coverage

The search combined broad web queries, date-restricted queries, primary-source citation following, public response data and platform APIs. Families checked include:

| Source family | Evidence or disposition |
| --- | --- |
| AI Impacts ESPAI 2023 and 2024 | Usable long-standing-open-problem forecasts; Millennium is an example within their broader autonomous task. |
| ESPAI 2016 and 2022 | Primary survey pages/questionnaires contain publishable-theorem and competition tasks, not the same Millennium item. Those proxies were not relabeled as Millennium probabilities. |
| Early general AI surveys, including Müller–Bostrom and the 2019 researcher follow-up | No matching dated Millennium probability series recovered. |
| FRI LEAP | One usable Millennium wave in Aug–Sep 2025; expert, public and superforecaster estimates plus four overlapping expert subgroups. |
| FRI XPT and later AI-progress accuracy review | Other mathematics benchmarks appear, but the review's Millennium example explicitly reuses LEAP 2025. [Primary review](https://forecastingresearch.substack.com/p/ai-progress-forecasts-accuracy). |
| Stanford AI Index 2026 | Local PDF inspection confirms LEAP as its source. It is not another fieldwork wave. |
| Tbilisi mathematician poll, Aug 2026 | Organizer article discusses research, trust and education; no Millennium probability/deadline table found. Conference program page was also checked. [Organizer account](https://nebius.science/stories/ai-math-conf). |
| “Mathematicians in the Age of AI” | The 2026 essay surveys technologies rather than polling a respondent population. [Primary paper](https://arxiv.org/html/2603.03684v1). |
| Metaculus | Relevant questions 28536 and 17432 found, but usable historical CDF/probability records could not be retrieved. Public rendering gives a censored current date or resolved outcome. Direct retrieval returned 403; the browser tool also failed to initialize. No resolution was substituted for a forecast. |
| Manifold | Official API search for both “millennium” and “millenium”: 119 and 35 results, 147 unique candidate titles. Downloaded detailed definitions and complete trade histories for 19 relevant questions; retained 13 for the deadline figures. |
| Kalshi / Polymarket | Located later “another problem” or non-AI prize-award markets; no comparable earlier first-event survey series recovered. |
| General public AI-opinion polls / Gasarch P-vs-NP polls | Attitudes, broad capability or truth-of-conjecture questions do not supply the target AI-assisted event. |
| Individual expert statements / model forecasting sites | Not population surveys; model-generated forecasts such as Takeoff.watch excluded from human evidence. |

This is a documented public-source sweep, not proof that no unpublished or inaccessible dataset exists. There is no numerical expansion of the formal survey-wave count based on publication dates, expert subgroups, secondary reports, or repeated platform observations.

## Historical market extraction

All reads used the public [Manifold API](https://docs.manifold.markets/api). Raw market definitions, comments for key rule checks, and 15,201 trade records across the 19 candidates are saved in `sources/manifold/`. Pagination was continued until every final page contained fewer than 1,000 records; contract IDs and unique trade IDs are checked.

Cutoffs are the ends of 2023, 2024 and 2025, and 7 September 2026. The last cutoff precedes the September 8 announcement discussed in the FRI review; it is not a claim about mathematical validation or prize resolution. It also does not imply there was no earlier public speculation. Markets not yet created or already closed/resolved at a cutoff contribute no observation. A cutoff after the forecast deadline is excluded.

The extraction selects the latest observed nonzero, non-redemption execution whose fills occur within two seconds of order creation and before the cutoff. This excludes resting limit orders that might have been created months before their eventual fills, avoiding dating future price information at the order's creation time. The resulting statistic is explicitly a last observable immediate-execution quote, not a reconstructed authoritative closing price. Every output row includes the exact trade ID, execution timestamp, stale days, definition flag and URL. The execution-account count is not a survey sample size, excludes ineligible order records, and does not establish unique humans.

The four matching markets require autonomous AI solutions under their currently saved wording. Archived comments reveal contested interpretation and later clarification. Exact historical versions of all rule text are not recoverable, so stable wording is not assumed. M06 explicitly gained a prize-award clarification after the cutoff. M07's background incorrectly substitutes a Diophantine-equation item for P-vs-NP; this is flagged rather than silently corrected. M11 and M12 are clones of M01 and M05.

Six downloaded candidates are not used in the cumulative-deadline figures: four ask for an event *in* a particular year/month (2025, 2026, or September 2026), and two are date/bracket markets for which a historical joint distribution was not reliably reconstructed. A date market with cancellation if nothing happens within its range is not an unconditional event CDF. Raw evidence remains available.

Deadlines retain exact semantics: “before 2030” means an exclusive boundary at 1 January 2030; “end of 2028” means 1 January 2029. The market axes say “deadline boundary.” Survey plots retain the papers' reported calendar-year convention and are not merged numerically with market quotes.

## Reproduction and checks

From the project root:

```sh
python3 scripts/expand_human_forecast_sweep.py
python3 -m unittest discover -s tests -p test_human_forecast_sweep.py
```

Six checks pass for delayed-fill handling, immediate executions, nontrades, exact raw-record/timestamp bindings, unchanged nonmonotonic probabilities, and the survey inventory. PNG figures were visually reviewed; a legend/caption overlap was fixed. Every plotted market coordinate is backed by a saved trade record. PDF and SVG exports are generated from the same figures.
