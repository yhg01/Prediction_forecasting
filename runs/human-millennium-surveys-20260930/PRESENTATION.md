# Human forecast presentation

The latest view is [one survey line graph](line-graph-v4/human_surveys_months.png), with a [one-page PDF](line-graph-v4/human_surveys_months.pdf). The horizontal axis gives the future deadline year. The vertical axis gives cumulative probability. Markers show reported values. Lines join those values as guides.

The legend gives the months when respondents answered each survey. ESPAI 2023 ran from 11 to 24 October 2023. ESPAI 2024 ran from 9 to 24 December 2024. LEAP wave 2 ran from 18 August to 15 September 2025. The month labels are Oct 2023, Dec 2024, and Aug-Sep 2025.

Each of the nine respondent groups has a distinct color. The colors span red, orange, gold, green, teal, and blue. The legend lists groups in survey date order. Colors identify groups; they do not imply nine survey dates. Seven LEAP groups share one survey period.

The graph shows 24 points at their reported coordinates. A label gives the remaining point: ESPAI 2024 reports 90% by 2822. That year is outside the graph. ESPAI 2023 has one reported point. All 25 saved observations remain accounted for. No extra point or fitted curve is added.

Reproduce this graph with `python3 scripts/plot_human_surveys_months.py`. [Validation](line-graph-v4/VALIDATION.json) binds the saved source data, script, and outputs. Checks cover exact coordinates, actual survey months, distinct colors, and preserved original files. The one-page PDF was rendered and visually checked.

The earlier [survey timeline](single-graph-v3/human_surveys_one_graph.png) remains available. Its rows name survey groups. Bubble area and labels show probability.

The earlier detailed views below remain available as supporting material.

## Published survey results

[Survey summary](presentation-v2/human_survey_summary.png)

ESPAI reports calendar years at stated probabilities. A year table shows the 2023 and 2024 results. Missing values are labeled "Not recovered". The 2024 90% year, 2822, is visible. The comparable 50% year moves from about 2050 to 2053.

LEAP reports probabilities at three deadlines. Three dot plots share a 0% to 100% scale. Each row names the group and gives its item answer count. Shading marks the four subgroups within the all-expert sample. The public count changes from 1,022 to 1,021 at the 2040 deadline.

The surveys use different questions, populations, and methods. ESPAI concerns autonomous solutions to long-standing open problems. Millennium problems are examples within that broader question. LEAP concerns a Millennium Prize Problem and permits substantial AI assistance. ESPAI averages fitted probability distributions. LEAP reports median respondent probabilities. These differences prevent a direct trend comparison across the two survey families.

## Question forms in ESPAI 2023

[Framing summary](presentation-v2/survey_framing_summary.png)

Two separate tables show the analyst summaries. One question form asks for years at fixed probabilities. The other asks for probabilities at fixed deadlines. Their samples contain 121 and 128 complete, numeric, monotone response triples. Their coordinate medians differ from the published pooled fit. These subsets do not add independent survey waves.

## Public markets

[Market table](presentation-v2/crowd_market_summary.png)

The table shows all 32 saved quotes from 13 questions. Its rows identify each event and deadline. Its columns give the historical cutoffs. Each quote shows its age at the cutoff. Missing cells say "No quote". The exact values are preserved, including the reversal between the 2035 and 2040 quotes in September 2026.

Market participants can include bots. Accounts can trade in several markets. Two questions are clones. Historical changes to the rules are not fully recoverable. These observations remain separate from the survey evidence. Full definitions, trade records, and rule notes remain in the saved registry and expanded sweep.

## Files and checks

[Three-page PDF](presentation-v2/human_forecast_results.pdf) · [Displayed source records](presentation-v2/DISPLAYED_DATA.json) · [Validation](presentation-v2/VALIDATION.json)

Reproduce the presentation from the project root:

```sh
python3 scripts/present_human_survey_results.py
```

The script verifies saved input hashes and exact observation coverage. It preserves all earlier data and figures. The final PDF pages were rendered with Poppler and checked for values, labels, spacing, and clipping.
