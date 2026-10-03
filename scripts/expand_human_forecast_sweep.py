#!/usr/bin/env python3
"""Render the expanded survey/crowd evidence from immutable public source files.

Markets are a separate evidence class, not additional human survey waves.
No fitting, monotonic repair, invented endpoint or historical resolution value.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from plot_human_millennium_surveys import observations, framing_supplement

ROOT = Path(__file__).resolve().parents[1] / 'runs/human-millennium-surveys-20260930'
SRC = ROOT / 'sources' / 'manifold'
DATES = ['2023-12-31', '2024-12-31', '2025-12-31', '2026-09-07']
# Deadlines are exact exclusive UTC boundaries; EOY 2028 is 2029-01-01.
MARKETS = [
    ('M01', '6vw71lj8bi', '2030-01-01', 'AI alone · before 2030', 'deadline_set'),
    ('M02', 'ipq49s5f07', '2035-01-01', 'AI alone · before 2035', 'deadline_set'),
    ('M03', 'lcy2g2svq6', '2040-01-01', 'AI alone · before 2040', 'deadline_set'),
    ('M04', 'q19px3g26d', '2050-01-01', 'AI alone · before 2050', 'deadline_set'),
    ('M05', 'd8MOJWh8V5QEfl97Gm8c', '2029-01-01', 'AI solves · end 2028', 'wording_underspecified'),
    ('M06', 'uOBecu3y5X2RseZXg3Ol', '2030-01-01', 'Substantial AI help · before 2030', 'criteria_clarified_after_cutoff'),
    ('M07', 'KmvP3Ggw5z7vFATu5urA', '2025-07-01', 'AI primary solver · June 2025', 'background_list_error'),
    ('M08', 'UsRnlu52sz', '2028-01-01', 'CMI confirms AI solution · end 2027', 'cmi_confirmation'),
    ('M09', 'QPZnPc5Rtd', '2032-01-01', 'AI primary solver · before 2032', 'peer_review_community_acceptance'),
    ('M10', 'lQl2h0pApI', '2030-01-01', 'No human assistance · before 2030', 'no_human_assistance'),
    ('M11', 'D6a8febc49', '2030-01-01', 'AI alone · before 2030 (clone M01)', 'clone_M01'),
    ('M12', 'D18865e22b', '2029-01-01', 'AI solves · end 2028 (clone M05)', 'clone_M05'),
    ('M13', 'PQA8qyC9ts', '2027-01-01', 'Verified AI-assisted solution · end 2026', 'wording_underspecified'),
]

def dt(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

def decimal_year(s):
    t = dt(s)
    return t.year + (t-dt(f'{t.year}-01-01')).total_seconds() / (dt(f'{t.year+1}-01-01')-dt(f'{t.year}-01-01')).total_seconds()

def save_csv(name, rows):
    with (ROOT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def load_history(mid):
    files = sorted(SRC.glob(mid+'_bets_*.json'), key=lambda p:int(p.stem.rsplit('_',1)[-1]))
    assert files and len(json.loads(files[-1].read_text())) < 1000, 'Incomplete pagination'
    records = [b for p in files for b in json.loads(p.read_text())]
    assert len(records) == len({b['id'] for b in records}), 'Duplicate historical trade'
    assert all(b['contractId'] == mid for b in records)
    return records

def immediate_execution(b):
    # A limit order's creation time may precede fills by months. Do not date its
    # eventual execution at order placement. Counterpart market orders supply
    # contemporaneous records. Include limit orders only with immediate fills.
    fills = b.get('fills', [])
    return (bool(fills) and abs(b.get('amount', 0)) > 0
            and not b.get('isRedemption', False)
            and isinstance(b.get('probAfter'), (int, float))
            and all(abs(f['timestamp']-b['createdTime']) <= 2000 for f in fills))

def market_rows():
    rows, registry = [], []
    for code, mid, deadline, label, flag in MARKETS:
        m = json.loads((SRC/f'{mid}.json').read_text())
        assert m['outcomeType'] == 'BINARY'
        history = load_history(mid)
        deadline_ms = dt(deadline).timestamp()*1000
        registry.append(dict(code=code, market_id=mid, question=m['question'].replace('\n',' '),
                             deadline_exclusive_utc=deadline, label=label, interpretation_flag=flag,
                             source_url=m['url'], history_records=len(history),
                             description_version='Retrieved 2026-09-30; historical edits not fully recoverable'))
        for date in DATES:
            cutoff = dt(date+'T23:59:59.999').timestamp()*1000
            if cutoff >= min(deadline_ms, m.get('resolutionTime', float('inf')), m.get('closeTime', float('inf'))):
                continue
            eligible = [b for b in history if immediate_execution(b)
                        and b['createdTime'] <= cutoff and all(f['timestamp'] <= cutoff for f in b['fills'])]
            if not eligible:
                continue
            b = max(eligible, key=lambda r:r['createdTime'])
            assert 0 <= b['probAfter'] <= 1
            stamp = datetime.fromtimestamp(b['createdTime']/1000, timezone.utc)
            rows.append(dict(series=f'{code}_{date}', code=code, market_id=mid,
                             snapshot_date=date, color_year=int(date[:4]),
                             deadline_exclusive_utc=deadline, deadline_year=decimal_year(deadline),
                             probability=b['probAfter'], trade_timestamp_utc=stamp.isoformat(),
                             stale_days=(cutoff-b['createdTime'])/86400000,
                             evidence_bet_id=b['id'], immediate_execution_accounts=len({v['userId'] for v in eligible}),
                             statistic='Last observed immediate-execution post-trade quote by cutoff; not a survey mean',
                             label=label, interpretation_flag=flag, source_url=m['url']))
    save_csv('crowd_forecast_snapshots.csv', rows)
    save_csv('crowd_market_registry.csv', registry)
    return rows, registry

CMAP = LinearSegmentedColormap.from_list('forecast_year', ['#c53240','#b35686','#725fbc','#2166ac'])
NORM = Normalize(2023, 2026)
def color(year): return CMAP(NORM(year))

def setup(ax, xmin=2025, xmax=2055):
    ax.set(xlim=(xmin,xmax), ylim=(0,1), xlabel='Future deadline year', ylabel='Probability by deadline')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.17); ax.set_axisbelow(True)

def savefig(fig, stem):
    for ext in ['png','pdf','svg']:
        fig.savefig(ROOT/f'{stem}.{ext}', dpi=180, facecolor='white')
    plt.close(fig)

def overview(surveys, markets):
    fig, axes = plt.subplots(1,2,figsize=(17,8.5))
    fig.subplots_adjust(left=.06,right=.98,bottom=.35,top=.79,wspace=.21)
    fig.text(.06,.945,'Forecasts of AI solving a major open mathematical problem',fontsize=23,weight='bold')
    fig.text(.06,.897,'Expanded evidence: 3 survey waves + public market histories · red = earlier, blue = later',fontsize=14,color='#4b5563')
    ax=axes[0]; setup(ax)
    ax.set_title('Published surveys · 9 respondent-group series',loc='left',fontsize=13,pad=14,weight='bold')
    groups=list(dict.fromkeys(r['series'] for r in surveys))
    styles=['None','--','-','--',':','-.',(0,(5,2,1,2)),(0,(1,1)),(0,(7,3))]
    marks=['D','s','o','^','v','P','X','*','h']
    for i,sid in enumerate(groups):
        rs=[r for r in surveys if r['series']==sid and r['deadline_year']<=2055]
        ax.plot([r['deadline_year'] for r in rs],[r['probability'] for r in rs],
                color=color(rs[0]['survey_year']),linestyle=styles[i],marker=marks[i],lw=2,ms=6,label=rs[0]['label'])
    ax.legend(loc='upper left',bbox_to_anchor=(-.025,-.16),ncol=2,frameon=False,fontsize=9.5,handlelength=2.6,columnspacing=1)
    ax=axes[1];setup(ax,2028,2052)
    ax.set_title('Crowd markets · same four deadline questions',loc='left',fontsize=13,pad=14,weight='bold')
    for date,style in zip(DATES[1:],['-','--','-.']):
        rs=sorted([r for r in markets if r['code'] in ['M01','M02','M03','M04'] and r['snapshot_date']==date],key=lambda r:r['deadline_year'])
        assert len(rs)==4
        ax.plot([r['deadline_year'] for r in rs],[r['probability'] for r in rs],color=color(int(date[:4])),marker='o',lw=2.5,ls=style,label=f'As of {date}')
    ax.set_xlabel('Deadline boundary · before 1 January of labeled year')
    ax.set_xticks([2030,2035,2040,2045,2050])
    ax.legend(loc='lower right',frameon=False,fontsize=10)
    fig.text(.55,.205,'Separate markets can disagree: the 2026 curve has a small reversal.\nQuotes are preserved; no monotonic correction is applied.\nMarket participants may include bots; these are not survey waves.',fontsize=10.2,color='#4b5563',linespacing=1.6)
    fig.text(.06,.086,'Survey definitions differ: ESPAI asks for autonomous solutions to long-standing open problems; LEAP allows substantial AI assistance.',fontsize=10.5,color='#374151')
    fig.text(.06,.058,'Markers are observed values. Joins are visual guides; single observations are not extended. The ESPAI 2024 90% date (2822) is outside the view.',fontsize=10,color='#6b7280')
    fig.text(.06,.031,'Sources: AI Impacts ESPAI 2023/2024; FRI LEAP 2025; Manifold public API. Market rules and participant composition may change. Retrieved 30 Sep 2026.',fontsize=9.6,color='#6b7280')
    savefig(fig,'human_forecast_sweep')

def sparse_figure(rows, registry):
    fig,axes=plt.subplots(3,3,figsize=(15,12))
    fig.subplots_adjust(left=.07,right=.98,top=.87,bottom=.14,hspace=.52,wspace=.28)
    fig.text(.07,.956,'Additional crowd forecasts with only one deadline',fontsize=23,weight='bold')
    fig.text(.07,.917,'9 market questions · points colored by historical snapshot year · every plotted point has a dated trade record',fontsize=12,color='#4b5563')
    for ax,reg in zip(axes.flat,registry[4:]):
        setup(ax,2025,2033)
        ax.set_title(reg['code']+' · '+reg['label'].replace(' · ','\n',1),loc='left',fontsize=10.5,pad=8)
        rs=[r for r in rows if r['code']==reg['code']]
        for r in rs:
            ax.scatter(r['deadline_year'],r['probability'],color=color(r['color_year']),
                       marker={2023:'o',2024:'s',2025:'^',2026:'D'}[r['color_year']],
                       s=58,zorder=3,edgecolor='white',linewidth=.7)
        ax.set_xticks([2025,2027,2029,2031,2033]);ax.set_yticks([0,.5,1]);ax.tick_params(labelsize=9)
        ax.set_xlabel('Deadline boundary',fontsize=9);ax.set_ylabel('Market probability',fontsize=9)
    handles=[Line2D([0],[0],marker=m,ls='None',color=color(int(d[:4])),label=d) for d,m in zip(DATES,['o','s','^','D'])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.076),ncol=4,frameon=False,fontsize=11)
    fig.text(.07,.057,'Dates: end 2027 → 1 Jan 2028; end 2028 → 1 Jan 2029. A lone deadline provides a point, not a probability curve.',fontsize=10,color='#4b5563')
    fig.text(.07,.034,'Quotes can be stale; exact trade dates, account counts and rule caveats are in the CSV. Cloned markets share resolution and may share traders.',fontsize=10,color='#4b5563')
    savefig(fig,'crowd_sparse_forecasts')

def framing_figure(surveys):
    rows=framing_supplement()
    fig,ax=plt.subplots(figsize=(11,6.3));fig.subplots_adjust(left=.10,right=.96,top=.79,bottom=.28)
    fig.text(.10,.935,'Human forecasts differ by question framing',fontsize=20,weight='bold')
    fig.text(.10,.873,'Same 2023 ESPAI wave · autonomous solution of a long-standing open math problem',fontsize=12,color='#4b5563')
    setup(ax,2023,2075)
    for framing,style,marker in [('fixed_probability','-','o'),('fixed_horizon','--','s')]:
        rs=[r for r in rows if r['framing']==framing]
        n=rs[0]['n_complete_monotone_triples']
        ax.plot([r['deadline_year'] for r in rs],[r['probability'] for r in rs],c=color(2023),ls=style,marker=marker,lw=2.4,label=f'{framing.replace("_"," ").capitalize()} · analyst medians · n={n}')
    ax.scatter(2050,.5,color=color(2023),marker='D',s=90,label='Published pooled fit · 50% ≈ 2050',zorder=4)
    ax.legend(loc='upper left',frameon=False,fontsize=10)
    fig.text(.10,.15,'The two analyst curves summarize separate framing groups using medians of complete, numeric, monotone response triples.',fontsize=10,color='#4b5563')
    fig.text(.10,.112,'The paper instead averages fitted individual distributions across framings. These summaries are not interchangeable.',fontsize=10,color='#4b5563')
    fig.text(.10,.065,'No extra survey wave is created by splitting a sample. Lines connect elicited coordinates only. Source: public cleaned ESPAI 2023 responses.',fontsize=9.5,color='#6b7280')
    savefig(fig,'survey_framing_sensitivity')

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('human_millennium_forecasts.*')}
    surveys=observations();rows,registry=market_rows()
    overview(surveys,rows);sparse_figure(rows,registry);framing_figure(surveys)
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('human_millennium_forecasts.*')}
    assert before==after, 'Existing survey artifacts must remain unchanged'
    nonmonotone=[]
    for date in DATES[1:]:
        rs=sorted([r for r in rows if r['code'] in ['M01','M02','M03','M04'] and r['snapshot_date']==date],key=lambda r:r['deadline_year'])
        for a,b in zip(rs,rs[1:]):
            if a['probability']>b['probability']:nonmonotone.append(dict(date=date,from_code=a['code'],to_code=b['code'],from_probability=a['probability'],to_probability=b['probability']))
    artifacts=[p for pattern in ['human_forecast_sweep.*','crowd_sparse_forecasts.*','survey_framing_sensitivity.*','crowd_forecast_snapshots.csv','crowd_market_registry.csv'] for p in ROOT.glob(pattern)]
    receipt=dict(status='numerically_validated_visual_review_pending',survey_waves=3,published_survey_group_series=9,
                 derived_framing_series=2,market_questions=len(registry),market_observations=len(rows),market_deadline_curves=3,
                 extra_single_deadline_observations=sum(r['code'] not in ['M01','M02','M03','M04'] for r in rows),
                 cutoffs=DATES,nonmonotone_market_segments=nonmonotone,
                 original_artifacts_preserved=before,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 evidence_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.glob('*') if p.is_file()},
                 output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    (ROOT/'EXPANSION_VALIDATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if 'sha256' not in k and k!='original_artifacts_preserved'},indent=2))

if __name__=='__main__': main()
