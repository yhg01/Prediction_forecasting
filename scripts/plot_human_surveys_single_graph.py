#!/usr/bin/env python3
"""One timeline graph of the saved published human survey observations."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'runs/human-millennium-surveys-20260930'
OUT = ROOT / 'single-graph-v3'
OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/predictor-survey-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK = '#182A3C'
MUTED = '#526477'
GRID = '#DFE6EC'
COLORS = {2023: '#BD4650', 2024: '#8D5CA5', 2025: '#236C9F'}
GROUPS = [
    ('espai2023', 8.8, '2023  AI researchers'),
    ('espai2024', 7.8, '2024  AI researchers'),
    ('leap2025_expert', 5.7, '2025  All experts'),
    ('leap2025_public', 4.7, '2025  General public'),
    ('leap2025_superforecaster', 3.7, '2025  Superforecasters'),
    ('leap2025_computer_science', 2.0, '2025  Computer scientists'),
    ('leap2025_economics', 1.0, '2025  Economists'),
    ('leap2025_industry', 0.0, '2025  Industry'),
    ('leap2025_policy', -1.0, '2025  Policy'),
]
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def area(probability):
    return float(probability) * 1300


def main():
    source = ROOT / 'human_millennium_forecasts.csv'
    assert SHA(source) == json.loads((ROOT / 'VALIDATION.json').read_text())['outputs'][source.name]
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 25
    assert {r['series'] for r in rows} == {sid for sid, _, _ in GROUPS}
    assert all(0 <= float(r['probability']) <= 1 for r in rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(14.4, 8.8), facecolor='white')
    fig.subplots_adjust(left=.265, right=.945, top=.785, bottom=.185)
    fig.text(.055, .949, 'Human forecasts of AI solving open mathematics',
             fontsize=23, weight='bold', color=INK)
    fig.text(.055, .907, 'One row per survey group. Bubble area and labels show the reported probability.',
             fontsize=12, color=MUTED)
    legend = [Line2D([], [], marker='o', linestyle='None', markerfacecolor='#6C8093',
                     markeredgecolor='white', markersize=area(p) ** .5, label=f'{p:.0%}')
              for p in [.1, .5, .9]]
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.05, .877),
               ncol=3, frameon=False, handletextpad=1.0, columnspacing=2.5,
               fontsize=10.5, title='Reported probability', title_fontsize=10.5)
    fig.text(.55, .845, 'Earlier survey: red     Later survey: blue', fontsize=10.5, color=MUTED)
    ax.set(xlim=(2025, 2060), ylim=(-1.85, 9.7),
           xticks=[2027, 2030, 2040, 2050, 2060], yticks=[])
    ax.set_xlabel('Future deadline year', fontsize=12, labelpad=12, color=INK)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(axis='x', labelsize=10.5, colors=MUTED, length=0, pad=9)
    ax.grid(axis='x', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.axhspan(-1.65, 2.75, facecolor='#F3F6F8', zorder=0)
    for _, y, _ in GROUPS:
        ax.axhline(y, color=GRID, linewidth=.65, zorder=0)
    for y, label in [(9.55, 'ESPAI | AI solves a long-standing open problem on its own'),
                     (6.6, 'LEAP | AI solves or substantially helps solve a Millennium problem')]:
        ax.text(0, y, label, transform=ax.get_yaxis_transform(), fontsize=10.3,
                color=MUTED, weight='bold', va='center')
    ax.text(0, 2.9, 'Expert subgroups within the same 2025 survey wave',
            transform=ax.get_yaxis_transform(), fontsize=9.6, color=MUTED, va='center')
    plotted, outside = [], []
    for sid, y, label in GROUPS:
        group = [r for r in rows if r['series'] == sid]
        year = int(group[0]['survey_year'])
        ax.text(-.04, y, label, transform=ax.get_yaxis_transform(), ha='right',
                va='center', fontsize=11, color=INK,
                weight='bold' if sid in ['espai2023', 'espai2024', 'leap2025_expert'] else 'normal')
        for row in group:
            x, probability = float(row['deadline_year']), float(row['probability'])
            if x > 2060:
                assert (sid, x, probability) == ('espai2024', 2822, .9)
                ax.annotate('', xy=(2059.8, y), xytext=(2054.2, y),
                            arrowprops={'arrowstyle': '->', 'color': COLORS[year], 'lw': 1.5})
                ax.text(2059.8, y - .55, '90% by 2822 (off scale)', ha='right', va='center',
                        fontsize=10.2, color=COLORS[year], weight='bold')
                outside.append(row)
                continue
            artist = ax.scatter(x, y, s=area(probability), color=COLORS[year],
                                edgecolor='white', linewidth=1.2, alpha=.94, zorder=3)
            assert tuple(artist.get_offsets()[0]) == (x, y)
            label = f'{probability * 100:g}%'
            if sid == 'espai2023':
                label += ' by ~2050'
            elif sid == 'espai2024':
                label += f' by {int(x)}'
            if sid.startswith('espai'):
                offset = (0, area(probability) ** .5 / 2 + 4)
                horizontal, vertical = 'center', 'bottom'
            else:
                offset = (area(probability) ** .5 / 2 + 5, 0)
                horizontal, vertical = 'left', 'center'
            ax.annotate(label, (x, y), xytext=offset,
                        textcoords='offset points', fontsize=10.4, color=INK,
                        ha=horizontal, va=vertical, zorder=4)
            plotted.append(row)
    fig.text(.055, .095, 'ESPAI averages fitted probability distributions. LEAP reports median respondent probabilities.',
             fontsize=10, color=MUTED)
    fig.text(.055, .068, 'The shaded rows are part of the all-expert sample. The questions and methods differ between surveys.',
             fontsize=10, color=MUTED)
    fig.text(.055, .035, 'Sources: AI Impacts ESPAI 2023/2024; FRI LEAP wave 2. Data saved 30 Sep 2026.',
             fontsize=9, color=MUTED)
    assert len(plotted) == 24 and len(outside) == 1
    assert sorted(json.dumps(r, sort_keys=True) for r in plotted + outside) == sorted(
        json.dumps(r, sort_keys=True) for r in rows)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'human_surveys_one_graph.{ext}', dpi=180, facecolor='white')
    plt.close(fig)
    assert SHA(source) == json.loads((ROOT / 'VALIDATION.json').read_text())['outputs'][source.name]
    receipt = {'status': 'numerically_checked_visual_review_pending',
               'source_sha256': SHA(source), 'script_sha256': SHA(__file__),
               'graphs': 1, 'axes': 1, 'survey_waves': 3, 'survey_group_series': 9,
               'observations': 25, 'bubbles': 24, 'off_scale_annotations': outside,
               'bubble_area': '1300 * probability in points squared',
               'x': 'Exact reported future deadline year', 'y': 'Named survey respondent group',
               'interpolation': False, 'missing_values_invented': False,
               'outputs_sha256': {f'human_surveys_one_graph.{ext}': SHA(OUT / f'human_surveys_one_graph.{ext}')
                                  for ext in ['png', 'svg', 'pdf']}}
    (OUT / 'VALIDATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if 'sha256' not in k}))


if __name__ == '__main__':
    main()
