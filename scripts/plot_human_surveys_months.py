#!/usr/bin/env python3
"""Draw the saved survey lines with survey months and nine group colors."""
import csv
from datetime import date
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'runs/human-millennium-surveys-20260930'
OUT = ROOT / 'line-graph-v4'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/predictor-survey-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

INK = '#182A3C'
MUTED = '#526477'
GROUPS = [
    ('espai2023', 'AI researchers (ESPAI)', '#C43D4B', 'D', 'None'),
    ('espai2024', 'AI researchers (ESPAI)', '#C46B1D', 's', '--'),
    ('leap2025_expert', 'All experts (LEAP)', '#AB870D', 'o', '-'),
    ('leap2025_public', 'General public (LEAP)', '#729126', '^', '--'),
    ('leap2025_superforecaster', 'Superforecasters (LEAP)', '#278553', 'v', ':'),
    ('leap2025_computer_science', 'Computer scientists (LEAP)', '#148B86', 'P', '-.'),
    ('leap2025_economics', 'Economists (LEAP)', '#208DA9', 'X', (0, (5, 2, 1, 2))),
    ('leap2025_industry', 'Industry (LEAP)', '#3474B2', '*', (0, (1, 1))),
    ('leap2025_policy', 'Policy (LEAP)', '#454EA4', 'h', (0, (7, 3))),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def survey_months(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if (first.year, first.month) == (last.year, last.month):
        return first.strftime('%b %Y')
    if first.year == last.year:
        return f'{first:%b}-{last:%b %Y}'
    return f'{first:%b %Y}-{last:%b %Y}'


def main():
    source = ROOT / 'human_millennium_forecasts.csv'
    historic_receipt = json.loads((ROOT / 'VALIDATION.json').read_text())
    assert sha(source) == historic_receipt['outputs'][source.name]
    preserved = {name: sha(ROOT / name) for name in historic_receipt['outputs']}
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 25
    assert {r['series'] for r in rows} == {g[0] for g in GROUPS}
    assert len({g[2] for g in GROUPS}) == 9
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
    fig.subplots_adjust(left=.075, right=.68, bottom=.25, top=.80)
    fig.text(.075, .945, 'Human forecasts of AI solving open mathematics',
             fontsize=24, weight='bold', color=INK)
    fig.text(.075, .898, 'Survey months are shown in the legend. Each respondent group has its own color.',
             fontsize=14, color=MUTED)
    handles, displayed, off_axis, series_receipts = [], [], [], []
    for sid, name, color, marker, style in GROUPS:
        group = sorted((r for r in rows if r['series'] == sid),
                       key=lambda r: float(r['deadline_year']))
        periods = {(r['survey_start'], r['survey_end']) for r in group}
        assert len(periods) == 1
        start, end = next(iter(periods))
        months = survey_months(start, end)
        points = [(float(r['deadline_year']), float(r['probability'])) for r in group]
        assert len(set(x for x, _ in points)) == len(points)
        assert all(0 <= p <= 1 for _, p in points)
        assert all(a[1] <= b[1] for a, b in zip(points, points[1:]))
        visible = [r for r in group if 2025 <= float(r['deadline_year']) <= 2055]
        label = f'{months} | {name}'
        line, = ax.plot([float(r['deadline_year']) for r in visible],
                        [float(r['probability']) for r in visible],
                        color=color, linestyle=style, marker=marker,
                        markersize=8, linewidth=2.3, label=label)
        assert list(zip(line.get_xdata(), line.get_ydata())) == [
            (float(r['deadline_year']), float(r['probability'])) for r in visible]
        handles.append(line)
        displayed.extend(visible)
        off_axis.extend(r for r in group if r not in visible)
        series_receipts.append({'series': sid, 'legend': label,
                                'survey_start': start, 'survey_end': end,
                                'color': color, 'reported_points': points,
                                'visible_points': len(visible)})
    assert len(displayed) == 24
    assert len(off_axis) == 1
    assert (off_axis[0]['series'], float(off_axis[0]['deadline_year']),
            float(off_axis[0]['probability'])) == ('espai2024', 2822, .9)
    assert sorted(json.dumps(r, sort_keys=True) for r in displayed + off_axis) == sorted(
        json.dumps(r, sort_keys=True) for r in rows)
    assert {survey_months(r['survey_start'], r['survey_end']) for r in rows} == {
        'Oct 2023', 'Dec 2024', 'Aug-Sep 2025'}
    ax.set(xlim=(2025, 2055), ylim=(0, 1), xlabel='Future deadline year',
           ylabel='Cumulative probability by deadline')
    ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050, 2055])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(color='#DCE3EA', alpha=.6, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.annotate('Oct 2023: 50% by about 2050', xy=(2050, .5), xytext=(2039.6, .235),
                color=GROUPS[0][2], fontsize=11,
                arrowprops={'arrowstyle': '-', 'color': GROUPS[0][2], 'lw': 1})
    ax.annotate('Dec 2024: 50% by 2053', xy=(2053, .5), xytext=(2043.7, .625),
                color=GROUPS[1][2], fontsize=11,
                arrowprops={'arrowstyle': '-', 'color': GROUPS[1][2], 'lw': 1})
    legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.025, 1.025),
                       frameon=False, fontsize=10.5, handlelength=2.6,
                       labelspacing=1.25, title='Survey month | Respondent group',
                       title_fontsize=11.5)
    fig.text(.706, .325, 'The four LEAP expert subgroups\nare part of the all-expert sample.\nThey belong to the same survey wave.',
             fontsize=10.5, color=MUTED, linespacing=1.55)
    fig.text(.706, .226, 'Dec 2024: 90% by 2822.\nThis point is outside the graph.',
             fontsize=10.5, color=GROUPS[1][2], linespacing=1.5)
    captions = [
        (.153, 'ESPAI asks about AI solving a long-standing open math problem on its own. Millennium problems are examples.'),
        (.122, 'LEAP asks about AI solving or substantially helping to solve a Millennium Prize Problem.'),
        (.084, 'Markers show reported values. Lines join those values as guides. ESPAI 2023 has one reported point.'),
        (.057, 'ESPAI averages fitted probability distributions. LEAP reports median respondent probabilities. The questions and methods differ.'),
        (.028, 'Sources: AI Impacts ESPAI 2023/2024; FRI LEAP wave 2. Source data saved 30 Sep 2026.'),
    ]
    for y, text in captions:
        fig.text(.075, y, text, fontsize=10.4 if y > .1 else 9.6, color=MUTED)
    fig.canvas.draw()
    bounds = fig.bbox
    renderer = fig.canvas.get_renderer()
    for artist in fig.texts + [legend]:
        box = artist.get_window_extent(renderer)
        assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1), (
            'Text extends beyond the figure', getattr(artist, 'get_text', lambda: 'Legend')())
    OUT.mkdir(exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'human_surveys_months.{ext}', dpi=180, facecolor='white')
    plt.close(fig)
    assert preserved == {name: sha(ROOT / name) for name in preserved}
    receipt = {
        'status': 'numerically_checked_visual_review_pending',
        'graphs': 1, 'axes': 1, 'survey_waves': 3, 'survey_group_series': 9,
        'reported_observations': 25, 'visible_observations': 24,
        'off_axis_observations': off_axis,
        'color_meaning': 'One distinct color per respondent group, in survey date order.',
        'colors': 'Red through orange, gold, green, teal, and blue.',
        'survey_dates': 'Months from actual survey start and end dates.',
        'series': series_receipts,
        'line_meaning': 'Straight joins between reported visible points only.',
        'extra_points_added': False,
        'source_sha256': sha(source), 'script_sha256': sha(__file__),
        'preserved_original_outputs_sha256': preserved,
        'outputs_sha256': {f'human_surveys_months.{ext}': sha(OUT / f'human_surveys_months.{ext}')
                           for ext in ['png', 'pdf', 'svg']},
    }
    (OUT / 'VALIDATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ['status', 'graphs', 'survey_waves',
                                           'survey_group_series', 'reported_observations',
                                           'visible_observations', 'color_meaning']}))


if __name__ == '__main__':
    main()
