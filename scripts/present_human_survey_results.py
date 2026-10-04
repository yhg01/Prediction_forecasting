#!/usr/bin/env python3
"""Present saved human forecasts without interpolating survey or market values."""
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'runs/human-millennium-surveys-20260930'
OUT = ROOT / 'presentation-v2'
OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/predictor-survey-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

INK = '#172638'
MUTED = '#536577'
BLUE = '#286C9B'
RED = '#B34851'
GRID = '#DFE6EC'
PALE = '#F3F6F8'
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name):
    with (ROOT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def pct(value):
    value = float(value) * 100
    return f'{value:g}%'


def text(fig, x, y, message, size=11, color=INK, **kwargs):
    return fig.text(x, y, message, fontsize=size, color=color, **kwargs)


def page(title, subtitle, number):
    fig = plt.figure(figsize=(13.2, 11), facecolor='white')
    text(fig, .055, .952, title, 24, weight='bold')
    text(fig, .055, .918, subtitle, 12, color=MUTED)
    text(fig, .94, .024, str(number), 9, color=MUTED, ha='right')
    return fig


def save(fig, stem, pdf):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=180, facecolor='white')
    pdf.savefig(fig)
    plt.close(fig)


def survey_page(surveys):
    fig = page('Published human surveys',
               'Three survey waves. Each section uses the question and statistic from its own survey.', 1)
    text(fig, .055, .867, 'ESPAI 2023 and 2024 | AI researchers', 15, weight='bold')
    text(fig, .055, .837, 'When could AI solve a long-standing open math problem on its own?', 12)
    text(fig, .055, .814, 'Millennium problems are examples in this broader question.', 10.5, color=MUTED)

    # Calendar dates are tabulated rather than placed on a distorted time axis.
    ax = fig.add_axes([.055, .658, .89, .135])
    ax.axis('off')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    xcols = [.04, .46, .66, .86]
    for x, title in zip(xcols, ['Survey year', 'Year at 10%', 'Year at 50%', 'Year at 90%']):
        ax.text(x, .90, title, fontsize=11, color=MUTED,
                ha='left' if x == .04 else 'center')
    ax.add_patch(Rectangle((.56, .05), .20, .72, facecolor='#EAF2F7', edgecolor='none'))
    displayed = []
    for year, y, color in [(2023, .57, RED), (2024, .20, BLUE)]:
        ax.text(.04, y, str(year), fontsize=14, weight='bold', color=color, va='center')
        for x, probability in zip(xcols[1:], [.1, .5, .9]):
            found = [r for r in surveys if r['series'] == f'espai{year}'
                     and float(r['probability']) == probability]
            assert len(found) <= 1
            value = ('~' if year == 2023 else '') + found[0]['deadline_year'] if found else 'Not recovered'
            ax.text(x, y, value, fontsize=16 if found else 10.5,
                    color=INK if found else MUTED, weight='bold' if found else 'normal',
                    ha='center', va='center')
            if found:
                displayed.append(found[0])
    ax.axhline(.76, color=GRID, lw=1)
    ax.axhline(.38, color=GRID, lw=.7)
    text(fig, .055, .637, 'The comparable 50% date moved from about 2050 to 2053.', 11.5, weight='bold')
    text(fig, .055, .613, 'ESPAI averages fitted probability distributions. The 2024 item used 130 fitted responses.', 10, color=MUTED)
    text(fig, .055, .593, 'The 2023 item count was not recovered. Missing cells are not zero. The year 2822 is shown in full.', 10, color=MUTED)

    text(fig, .055, .550, 'LEAP 2025 | Seven groups in one survey wave', 15, weight='bold')
    text(fig, .055, .521, 'Chance that AI solves or substantially helps solve a Millennium Prize Problem', 12)
    text(fig, .055, .496, 'Survey: 18 Aug to 15 Sep 2025. Dots show median respondent probabilities.', 10.5, color=MUTED)
    keys = ['expert', 'public', 'superforecaster', 'computer_science', 'economics', 'industry', 'policy']
    labels = ['All experts', 'General public', 'Superforecasters', 'Computer scientists', 'Economists', 'Industry', 'Policy']
    ys = [6.8, 5.8, 4.8, 3.3, 2.3, 1.3, .3]
    data = {(r['series'], int(float(r['deadline_year']))): r for r in surveys if r['series'].startswith('leap')}
    left = fig.add_axes([.055, .181, .24, .284])
    left.set(xlim=(0, 1), ylim=(-.3, 7.4)); left.axis('off')
    for key, label, y in zip(keys, labels, ys):
        ns = [int(data[(f'leap2025_{key}', deadline)]['n_item']) for deadline in [2027, 2030, 2040]]
        nlabel = f'{min(ns):,}' if len(set(ns)) == 1 else f'{min(ns):,} to {max(ns):,}'
        left.text(.02 if y > 4 else .07, y, label, fontsize=10.8, va='center',
                  weight='bold' if y > 4 else 'normal', color=INK)
        left.text(.93, y, nlabel, fontsize=9.3, color=MUTED, ha='right', va='center')
    left.text(.02, 7.25, 'Group', fontsize=9.5, color=MUTED)
    left.text(.93, 7.25, 'Answers', fontsize=9.5, color=MUTED, ha='right')
    left.text(.02, 4.0, 'Subgroups within all experts', fontsize=8.8, color=MUTED, va='center')
    for i, deadline in enumerate([2027, 2030, 2040]):
        ax = fig.add_axes([.321 + i * .209, .181, .182, .284])
        ax.set(xlim=(0, 100), ylim=(-.3, 7.4), yticks=[], xticks=[0, 50, 100])
        ax.set_xticklabels(['0%', '50%', '100%'], fontsize=9, color=MUTED)
        ax.set_title(f'By {deadline}', loc='left', pad=14, fontsize=12, weight='bold', color=INK)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(axis='x', length=0, pad=7)
        ax.grid(axis='x', color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.axhspan(-.2, 3.85, color=PALE, zorder=0)
        for key, y in zip(keys, ys):
            row = data[(f'leap2025_{key}', deadline)]
            probability = float(row['probability']) * 100
            ax.scatter(probability, y, s=65 if y > 4 else 45, color=BLUE,
                       edgecolor='white', linewidth=.8, zorder=3)
            ax.annotate(pct(row['probability']), (probability, y), xytext=(7, 0),
                        textcoords='offset points', va='center', fontsize=10, color=INK)
            displayed.append(row)
    text(fig, .055, .130, 'All four shaded subgroup rows are part of the all-expert sample.', 10.2, color=MUTED)
    text(fig, .055, .109, 'The public item has 1,022 answers for 2027 and 2030, and 1,021 for 2040.', 10.2, color=MUTED)
    text(fig, .055, .073, 'The questions, respondent groups, and methods differ between ESPAI and LEAP.', 11, weight='bold')
    text(fig, .055, .052, 'These results do not show a steady increase in human optimism.', 10.5, color=MUTED)
    text(fig, .055, .024, 'Sources: AI Impacts ESPAI 2023/2024; FRI LEAP wave 2. Saved source data: 30 Sep 2026.', 8.7, color=MUTED)
    assert len(displayed) == len(surveys) == 25
    return fig, displayed


def framing_page(framing):
    fig = page('Results by question form',
               'ESPAI 2023. Two question forms, two respondent subsets, one survey wave.', 2)
    groups = [('fixed_probability', 121, 'Question form A',
               'At what year would the chance be 10%, 50%, or 90%?', ['10%', '50%', '90%']),
              ('fixed_horizon', 128, 'Question form B',
               'What is the chance by 2033, 2043, or 2073?', ['By 2033', 'By 2043', 'By 2073'])]
    displayed = []
    for (kind, n, heading, question, headings), top in zip(groups, [.833, .521]):
        text(fig, .055, top, f'{heading} | {n} complete response triples', 16, weight='bold')
        text(fig, .055, top - .038, question, 12)
        rows = [r for r in framing if r['framing'] == kind]
        assert len(rows) == 3 and {int(r['n_complete_monotone_triples']) for r in rows} == {n}
        ax = fig.add_axes([.055, top - .198, .89, .117]); ax.axis('off')
        ax.set(xlim=(0, 1), ylim=(0, 1))
        for x, row, h in zip([.16, .50, .84], rows, headings):
            ax.add_patch(Rectangle((x - .145, 0), .29, 1, facecolor=PALE, edgecolor='none'))
            ax.text(x, .73, h, ha='center', fontsize=12, color=MUTED)
            result = str(int(float(row['deadline_year']))) if kind == 'fixed_probability' else pct(row['probability'])
            ax.text(x, .28, result, ha='center', fontsize=25, weight='bold', color=BLUE)
            displayed.append(row)
    text(fig, .055, .236, 'The 50% year is 2042 for form A. Form B gives 20% by 2043.', 13, weight='bold')
    text(fig, .055, .200, 'These are analyst summaries for separate respondent subsets.', 11, color=MUTED)
    text(fig, .055, .169, 'Each value is the median of one coordinate from complete, numeric, monotone triples.', 11, color=MUTED)
    text(fig, .055, .138, 'The published pooled 50% year is about 2050. It uses a different method.', 11, color=MUTED)
    text(fig, .055, .107, 'The difference does not measure a change in the same people over time.', 11, color=MUTED)
    text(fig, .055, .052, 'Source: saved public cleaned ESPAI 2023 responses. These summaries add no survey wave.', 9.3, color=MUTED)
    assert len(displayed) == len(framing) == 6
    return fig, displayed


def market_page(markets, registry):
    fig = page('Public market quotes | Separate evidence',
               'Last eligible saved quote at each cutoff. A quote can be old. Participants can include bots.', 3)
    dates = ['2023-12-31', '2024-12-31', '2025-12-31', '2026-09-07']
    data = {(r['code'], r['snapshot_date']): r for r in markets}
    assert len(data) == len(markets) == 32
    headers = ['31 Dec 2023', '31 Dec 2024', '31 Dec 2025', '7 Sep 2026']
    cmap = LinearSegmentedColormap.from_list('quote_probability', ['#F7FAFC', '#79ADD0'])
    ax = fig.add_axes([.055, .195, .89, .642])
    ax.set(xlim=(0, 1), ylim=(-.7, 13.7)); ax.axis('off')
    columns = [.535, .652, .769, .886]
    ax.text(.01, 13.35, 'Market question and deadline', fontsize=10.5, color=MUTED)
    for x, header in zip(columns, headers):
        ax.text(x, 13.35, header, fontsize=10, color=MUTED, ha='center')
    displayed = []
    for index, reg in enumerate(registry):
        y = 12.5 - index
        label = reg['label'].replace(' · ', ' | ').replace('AI-assisted', 'AI assisted')
        ax.text(.01, y + .1, reg['code'] + '  ' + label, fontsize=9.3, va='center', color=INK)
        for x, date in zip(columns, dates):
            row = data.get((reg['code'], date))
            ax.add_patch(Rectangle((x - .053, y - .34), .106, .83,
                                   facecolor=cmap(float(row['probability'])) if row else PALE,
                                   edgecolor='white', linewidth=1.8))
            ax.text(x, y + .15, f"{float(row['probability']) * 100:.1f}%" if row else 'No quote',
                    fontsize=12 if row else 8.3, weight='bold' if row else 'normal',
                    color=INK if row else MUTED, ha='center', va='center')
            if row:
                ax.text(x, y - .16, f"{float(row['stale_days']):.1f} days old", fontsize=7.9,
                        color=MUTED, ha='center', va='center')
                displayed.append(row)
        if index == 3:
            ax.axhline(y - .49, color=GRID, lw=1.5)
    text(fig, .055, .156, 'M01 to M04 share the saved AI-alone wording. Each row is a different deadline question.', 10.5)
    text(fig, .055, .129, 'The other rows use different rules. M11 and M12 are clones of M01 and M05.', 10.5)
    text(fig, .055, .102, 'No quote means no eligible observation was recovered at that cutoff. It does not mean 0%.', 10.5)
    text(fig, .055, .075, 'Historical rule edits are not fully recoverable. Exact trade dates and rule notes are in the saved data.', 10, color=MUTED)
    text(fig, .055, .045, 'Source: Manifold public API, retrieved 30 Sep 2026. Market observations add no survey wave.', 9.3, color=MUTED)
    assert len(displayed) == len(markets)
    return fig, displayed


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    sources = ['human_millennium_forecasts.csv', 'espai2023_framing_supplement.csv',
               'crowd_forecast_snapshots.csv', 'crowd_market_registry.csv']
    protected = {str(p.relative_to(ROOT)): SHA(p) for p in ROOT.iterdir()
                 if p.is_file() and p.suffix in ['.csv', '.png', '.svg', '.pdf', '.json']}
    source_checks = json.loads((ROOT / 'VALIDATION.json').read_text())['outputs']
    expansion_checks = json.loads((ROOT / 'EXPANSION_VALIDATION.json').read_text())['output_sha256']
    for name in ['human_millennium_forecasts.csv', 'crowd_forecast_snapshots.csv', 'crowd_market_registry.csv']:
        assert SHA(ROOT / name) == (source_checks | expansion_checks)[name]
    surveys, framing, markets, registry = [load(name) for name in sources]
    assert len({r['series'] for r in surveys}) == 9
    assert len({r['survey_year'] for r in surveys}) == 3
    assert len(registry) == 13
    assert len({(r['series'], r['deadline_year'], r['probability']) for r in surveys}) == 25
    assert all(0 <= float(r['probability']) <= 1 for r in surveys + framing + markets)
    rendered = {}
    with PdfPages(OUT / 'human_forecast_results.pdf', metadata={
            'Title': 'Human forecasts: survey results and separate market evidence',
            'Author': 'Yuhe', 'Subject': 'Saved September 2026 evidence; no new data collection'}) as pdf:
        for stem, builder, args in [
                ('human_survey_summary', survey_page, (surveys,)),
                ('survey_framing_summary', framing_page, (framing,)),
                ('crowd_market_summary', market_page, (markets, registry))]:
            fig, records = builder(*args)
            rendered[stem] = records
            save(fig, stem, pdf)
    # Every source observation must appear once in its own section.
    for name, expected in zip(rendered, [surveys, framing, markets]):
        assert Counter(json.dumps(r, sort_keys=True) for r in rendered[name]) == Counter(
            json.dumps(r, sort_keys=True) for r in expected)
    assert protected == {name: SHA(ROOT / name) for name in protected}
    (OUT / 'DISPLAYED_DATA.json').write_text(json.dumps(rendered, indent=2) + '\n')
    receipt = {
        'status': 'numerical_checks_passed_visual_review_pending',
        'created_client_date': '2026-10-03', 'new_source_collection': False,
        'survey_waves': 3, 'survey_group_series': 9, 'survey_points_shown': 25,
        'framing_points_shown': 6, 'market_points_shown': 32, 'market_questions': 13,
        'rendering': 'Calendar-year table, three probability dot plots, framing tables, market quote matrix.',
        'interpolation': False, 'extrapolation': False, 'monotonic_repair': False,
        'missing_values': 'Explicitly labeled; never converted to zero.',
        'espai_2024_90pct_year_shown': 2822, 'overlapping_expert_subgroups_labeled': True,
        'source_sha256': {name: SHA(ROOT / name) for name in sources},
        'original_artifacts_preserved_sha256': protected,
        'script_sha256': SHA(__file__),
        'output_sha256': {p.name: SHA(p) for p in OUT.iterdir()
                          if p.is_file() and p.name != 'VALIDATION.json'
                          and p.suffix in ['.png', '.svg', '.pdf', '.json']},
    }
    (OUT / 'VALIDATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if 'sha256' not in k}))


if __name__ == '__main__':
    main()
