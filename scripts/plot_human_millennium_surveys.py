#!/usr/bin/env python3
"""Reproduce the human-survey comparison from saved primary-source evidence."""
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]/'runs/human-millennium-surveys-20260930'
SRC=ROOT/'sources'
URL23='https://arxiv.org/html/2401.02843v3'
URL24='https://aiimpacts.org/wp-content/uploads/2026/09/ESPAI2024.pdf'
URL25='https://leap.forecastingresearch.org/reports/wave2#millennium-prize'

def observations():
    rows=[]
    def add(series,label,year,start,end,event,stat,points,url,n=None,detail=''):
        for x,p,ni in [(x,p,n) if len(t)==2 else t for t in points for x,p in [t[:2]]]:
            rows.append(dict(series=series,label=label,survey_year=year,survey_start=start,survey_end=end,
                             deadline_year=x,probability=p,n_item=ni,event=event,statistic=stat,source_url=url,detail=detail))
    # Published rounded calendar years; no inferred 10%/90% points for 2023.
    assert '27 years' in (SRC/'espai2023.html').read_text()
    add('espai2023','2023 · AI researchers (ESPAI)',2023,'2023-10-11','2023-10-24',
        'Autonomously solve a long-standing open math problem; Millennium is one example',
        'Published mean of fitted individual gamma CDFs',[(2050,.5)],URL23,
        detail='Rounded published 27-year horizon + 2023; survey total 2778, item subset. Single point only.')
    txt=(SRC/'espai2024.txt').read_text()
    assert re.search(r'Solve unsolved math problem.*130\s+4 yr \(2028\)\s+29 yr \(2053\)\s+798 yr \(2822\)',txt)
    add('espai2024','2024 · AI researchers (ESPAI)',2024,'2024-12-09','2024-12-24',
        'Autonomously solve a long-standing open math problem; Millennium is one example',
        'Published mean mixture CDF quantiles',[(2028,.1),(2053,.5),(2822,.9)],URL24,130,
        'Table 7, printed p.45. Survey in 2024; paper in 2026. 90% year outside main plot, retained in data.')
    lines=(SRC/'leap_tables.txt').read_text().splitlines()
    assert lines[0]=='Resolution Date | Expert | Public | Superforecaster'
    group_map={'Expert':'All experts','Public':'General public','Superforecaster':'Superforecasters',
               'Computer Science':'Computer scientists','Economics':'Economists','Industry':'Industry','Policy':'Policy'}
    for header,first in [(lines[0].split(' | ')[1:],1),(lines[4].split(' | ')[1:],5)]:
        for k,group in enumerate(header):
            points=[]
            for line in lines[first:first+3]:
                cells=line.split(' | ');year=int(cells[0][-4:])
                match=re.fullmatch(r'([\d.]+) \(([^)]+)\) \(n = (\d+)\)',cells[k+1])
                assert match,cells[k+1]
                points.append((year,float(match[1])/100,int(match[3])))
            add('leap2025_'+group.lower().replace(' ','_'),'2025 · '+group_map[group],2025,
                '2025-08-18','2025-09-15','AI solves or substantially assists with a Clay Millennium Prize Problem',
                'Published median respondent probability',points,URL25,
                detail='LEAP wave 2. Expert subgroups overlap the all-expert curve; these are not independent survey waves.')
    for sid in {x['series'] for x in rows}:
        rs=sorted([x for x in rows if x['series']==sid],key=lambda r:r['deadline_year'])
        assert len({r['deadline_year'] for r in rs})==len(rs)
        assert all(0<=r['probability']<=1 for r in rs)
        assert all(a['probability']<=b['probability'] for a,b in zip(rs,rs[1:]))
        assert all(r['survey_year']==int(r['survey_start'][:4]) for r in rs)
    assert len(rows)==25 and len({r['series'] for r in rows})==9
    return rows

def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plot(rows):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    cmap=LinearSegmentedColormap.from_list('survey_date',['#c83e49','#825bb1','#2468bf'])
    norm=Normalize(2023,2025)
    fig,ax=plt.subplots(figsize=(15.6,8.7))
    fig.subplots_adjust(left=.08,right=.69,bottom=.25,top=.81)
    fig.text(.08,.945,'When did humans expect AI to help solve a major open math problem?',fontsize=23,weight='bold')
    fig.text(.08,.897,'9 reported group series · 3 survey waves · colored by year conducted',fontsize=15,color='#4b5563')
    groups=list(dict.fromkeys(r['series'] for r in rows))
    markers=['D','s','o','^','v','P','X','*','h']
    styles=['None','--','-','--',':','-.',(0,(5,2,1,2)),(0,(1,1)),(0,(7,3))]
    handles=[]
    for i,sid in enumerate(groups):
        rs=sorted([r for r in rows if r['series']==sid],key=lambda r:r['deadline_year'])
        visible=[r for r in rs if 2025<=r['deadline_year']<=2055]
        color=cmap(norm(rs[0]['survey_year']))
        h,=ax.plot([r['deadline_year'] for r in visible],[r['probability'] for r in visible],
                   color=color,linestyle=styles[i],marker=markers[i],markersize=7.5,
                   linewidth=2.3,alpha=1 if i<5 else .78,label=rs[0]['label'])
        handles.append(h)
    ax.set(xlim=(2025,2055),ylim=(0,1),xlabel='Future deadline year',ylabel='Cumulative probability by deadline')
    ax.set_xticks([2025,2030,2035,2040,2045,2050,2055]);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.17);ax.set_axisbelow(True)
    ax.annotate('2023: 50% ≈ 2050',xy=(2050,.5),xytext=(2040,.36),color=cmap(norm(2023)),
                arrowprops=dict(arrowstyle='-',color=cmap(norm(2023)),lw=1),fontsize=11)
    ax.annotate('2024: 50% = 2053',xy=(2053,.5),xytext=(2044,.61),color=cmap(norm(2024)),
                arrowprops=dict(arrowstyle='-',color=cmap(norm(2024)),lw=1),fontsize=11)
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.025,1.025),frameon=False,fontsize=11.7,
              handlelength=2.8,labelspacing=1.0,title='Conducted · respondent group',title_fontsize=12)
    fig.text(.718,.33,'LEAP’s four expert subgroups\nare also included in “All experts.”\nThey add detail, not survey waves.',fontsize=10.8,color='#4b5563',linespacing=1.5)
    fig.text(.08,.159,'Definitions differ: ESPAI asks about autonomous solutions to long-standing open problems (Millennium is an example).',fontsize=11,color='#374151')
    fig.text(.08,.131,'LEAP asks specifically about solving or substantially assisting with a Millennium Prize Problem.',fontsize=11,color='#374151')
    fig.text(.08,.091,'Markers are reported observations; straight joins are visual guides. A lone point is not extended into a curve.',fontsize=10.5,color='#6b7280')
    fig.text(.08,.067,'ESPAI uses mean fitted CDFs; LEAP uses median probabilities. The 2024 90% date (2822) is retained in the data, outside this view.',fontsize=10.5,color='#6b7280')
    fig.text(.08,.036,'Sources: AI Impacts ESPAI 2023 and 2024; Forecasting Research Institute LEAP wave 2 (Aug–Sep 2025). Search: 30 Sep 2026.',fontsize=10,color='#6b7280')
    for ext in ['png','pdf','svg']:
        fig.savefig(ROOT/f'human_millennium_forecasts.{ext}',dpi=180,facecolor='white')
    plt.close(fig)

def framing_supplement():
    with (SRC/'espai2023_responses.csv').open() as f: rows=list(csv.DictReader(f))
    records=[]
    for framing,suffixes in [('fixed_probability',['10percent','50percent','90percent']),('fixed_horizon',['10years','20years','50years'])]:
        triples=[]
        for row in rows:
            try:v=[float(row[f'task_MillenniumPrize_{s}_clean']) for s in suffixes]
            except ValueError:continue
            if not all(x>=0 for x in v) or v!=sorted(v) or (framing=='fixed_horizon' and max(v)>100):continue
            triples.append(v)
        for i,suffix in enumerate(suffixes):
            val=median([v[i] for v in triples])
            records.append({'survey_year':2023,'framing':framing,'n_complete_monotone_triples':len(triples),
                            'deadline_year':2023+val if framing=='fixed_probability' else 2023+[10,20,50][i],
                            'probability':[.1,.5,.9][i] if framing=='fixed_probability' else val/100,
                            'statistic':'Analyst-computed coordinatewise median of complete monotone cleaned triples; not the published pooled CDF'})
    save_csv(ROOT/'espai2023_framing_supplement.csv',records)
    return records

def main():
    rows=observations();save_csv(ROOT/'human_millennium_forecasts.csv',rows);plot(rows)
    extra=framing_supplement()
    outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('human_millennium_forecasts.*')}
    evidence={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.iterdir() if p.is_file() and p.name!='download_manifest.json'}
    receipt={'status':'validated','survey_waves':3,'group_series':9,'reported_points':25,
             'visible_points':24,'single_point_series':['espai2023'],'off_axis_points':[{'series':'espai2024','year':2822,'probability':.9}],
             'no_synthetic_start_or_extrapolation':True,'survey_year_color_range':[2023,2025],
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'evidence':evidence,'outputs':outputs,
             'supplemental_framing':extra,'trend_conclusion':'Comparable 2023→2024 headline dates move from about 2050 to 2053. These data do not establish increasing human optimism.'}
    (ROOT/'VALIDATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['evidence','outputs','supplemental_framing']}))

if __name__=='__main__': main()
