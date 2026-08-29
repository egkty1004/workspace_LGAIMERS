#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path('/home/2022113165/workspace_LGAIMERS/데이터/open/data/trackman_history.csv')
OUTPUT = Path('/tmp/aimers9-trackman-physics-eda-v1')
PHYS = ['rel_speed','zone_speed','spin_rate','induced_vert_break','horz_break','extension','rel_height','rel_side']
META = ['season','game_month','pitch_type_group','tagged_pitch_type','auto_pitch_type','pitcher_hand','batter_hand']
USECOLS = META + PHYS
QUANTILES = [0.001,0.01,0.05,0.25,0.5,0.75,0.95,0.99,0.999]

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def clean(series: pd.Series) -> tuple[pd.Series,np.ndarray]:
    x=pd.to_numeric(series,errors='coerce').astype('float64')
    a=x.to_numpy(); finite=np.isfinite(a)
    return x,finite

def core(series: pd.Series, include_quantiles: bool=True) -> dict:
    x,finite=clean(series); a=x.to_numpy(); v=a[finite]
    result={'total':int(len(a)),'finite':int(finite.sum()),'missing':int(x.isna().sum()),
            'missing_rate':float(x.isna().mean()),'posinf':int(np.isposinf(a).sum()),
            'neginf':int(np.isneginf(a).sum())}
    if len(v):
        result.update({'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean()),
                       'std':float(v.std(ddof=1)) if len(v)>1 else None})
        if include_quantiles:
            q=np.quantile(v,QUANTILES)
            result['quantiles']={str(p):float(z) for p,z in zip(QUANTILES,q)}
    return result

def season_core(frame: pd.DataFrame,col: str) -> dict:
    out={}
    for season,g in frame.groupby('season',sort=True):
        c=core(g[col]); q=c.pop('quantiles',{})
        c['q01']=q.get('0.01'); c['q50']=q.get('0.5'); c['q99']=q.get('0.99')
        out[str(int(season))]=c
    return out

def outliers(series: pd.Series) -> dict:
    x,finite=clean(series); v=x.to_numpy()[finite]
    q1,med,q3=np.quantile(v,[.25,.5,.75]); iqr=q3-q1
    lo=q1-3*iqr; hi=q3+3*iqr
    mad=float(np.median(np.abs(v-med)))
    rz=np.full(len(v),0.0) if mad==0 else 0.6744897501960817*np.abs(v-med)/mad
    values,counts=np.unique(v,return_counts=True)
    low=[{'value':float(z),'count':int(c)} for z,c in zip(values[:10],counts[:10])]
    high=[{'value':float(z),'count':int(c)} for z,c in zip(values[-10:][::-1],counts[-10:][::-1])]
    return {'iqr_rule':'outside [Q1-3*IQR,Q3+3*IQR]','iqr_low':float(lo),'iqr_high':float(hi),
            'iqr_extreme_count':int(((v<lo)|(v>hi)).sum()),'iqr_extreme_rate':float(((v<lo)|(v>hi)).mean()),
            'mad':mad,'modified_z_rule':'0.67448975*abs(x-median)/MAD > 3.5',
            'mad_extreme_count':int((rz>3.5).sum()),'mad_extreme_rate':float((rz>3.5).mean()),
            'zero_count':int((v==0).sum()),'lowest_values':low,'highest_values':high}

def family_stats(frame: pd.DataFrame) -> dict:
    out={}
    for (season,family),g in frame.groupby(['season','pitch_type_group'],sort=True,dropna=False):
        key=f'{int(season)}|{family}'
        entry={'season':int(season),'pitch_type_group':str(family),'rows':int(len(g)),'physics':{}}
        for col in PHYS:
            c=core(g[col]); q=c.pop('quantiles',{})
            entry['physics'][col]={'finite':c['finite'],'missing':c['missing'],'missing_rate':c['missing_rate'],
                                   'mean':c.get('mean'),'std':c.get('std'),'median':q.get('0.5'),
                                   'q01':q.get('0.01'),'q99':q.get('0.99')}
        out[key]=entry
    return out

def hand_stats(frame: pd.DataFrame,col: str) -> dict:
    out={}
    for hand,g in frame.groupby('pitcher_hand',sort=True,dropna=False):
        x,finite=clean(g[col]); v=x.to_numpy()[finite]
        out[str(hand)]={'rows':int(len(g)),'finite':int(len(v)),'mean':float(v.mean()),'std':float(v.std(ddof=1)),
                        'median':float(np.median(v)),'negative_fraction':float((v<0).mean()),
                        'positive_fraction':float((v>0).mean()),'abs_mean':float(np.abs(v).mean()),
                        'abs_median':float(np.median(np.abs(v)))}
    return out

def category_summary(frame: pd.DataFrame,col: str) -> dict:
    counts=frame[col].value_counts(dropna=False)
    return {'dtype':str(frame[col].dtype),'missing':int(frame[col].isna().sum()),'unique':int(frame[col].nunique(dropna=True)),
            'counts':{str(k):int(v) for k,v in counts.items()}}

def main() -> int:
    if OUTPUT.exists(): raise RuntimeError('output exists; refusing overwrite')
    header=pd.read_csv(SOURCE,nrows=0,encoding='utf-8-sig')
    if any(c not in header for c in USECOLS): raise RuntimeError('projection missing')
    frame=pd.read_csv(SOURCE,usecols=USECOLS,encoding='utf-8-sig')
    if list(frame.columns) != [c for c in header.columns if c in USECOLS]:
        raise RuntimeError('projection ordering drift')
    report={'contract':'trackman-physics-eda-v1-target-free','source':str(SOURCE),'source_sha256':sha(SOURCE),
            'projection':USECOLS,'rows':int(len(frame)),'target_access':False,'test_access':False,
            'public_access':False,'entity_access':False,'physics':{},'categories':{}}
    for col in PHYS:
        report['physics'][col]={'dtype':str(frame[col].dtype),'overall':core(frame[col]),
                                'by_season':season_core(frame,col),'outliers':outliers(frame[col])}
    for col in META: report['categories'][col]=category_summary(frame,col)
    report['tagged_auto_exact_agreement_fraction']=float((frame['tagged_pitch_type'].astype('string')==frame['auto_pitch_type'].astype('string')).mean())
    report['family_by_season']=family_stats(frame)
    report['handedness_sign']={c:hand_stats(frame,c) for c in ('horz_break','rel_side')}
    numeric=frame[PHYS].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    report['correlation']={'pearson':numeric.corr(method='pearson',min_periods=2).to_dict(),
                           'spearman':numeric.corr(method='spearman',min_periods=2).to_dict(),
                           'paired_counts':numeric.notna().astype('int64').T.dot(numeric.notna().astype('int64')).to_dict()}
    report['monthly']={}
    for col in PHYS:
        rows=[]
        for (season,month),g in frame.groupby(['season','game_month'],sort=True):
            x,finite=clean(g[col]); v=x.to_numpy()[finite]
            rows.append({'season':int(season),'month':int(month),'rows':int(len(g)),'finite':int(len(v)),
                         'mean':float(v.mean()) if len(v) else None,'median':float(np.median(v)) if len(v) else None})
        report['monthly'][col]=rows
    report['adjacent_season_shift']={}
    for col in PHYS:
        rows=[]; by=report['physics'][col]['by_season']
        for a,b in zip(range(2019,2024),range(2020,2025)):
            x,y=by[str(a)],by[str(b)]
            rows.append({'from':a,'to':b,'mean_delta':y['mean']-x['mean'],
                         'delta_over_prior_std':(y['mean']-x['mean'])/x['std'] if x['std'] else None,
                         'std_ratio':y['std']/x['std'] if x['std'] else None,
                         'missing_rate_delta':y['missing_rate']-x['missing_rate']})
        report['adjacent_season_shift'][col]=rows
    OUTPUT.mkdir(parents=True)
    json_path=OUTPUT/'trackman_physics_eda.json'
    json_path.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# TrackMan Physics EDA v1','',f'- rows: {len(frame)}',f'- source_sha256: `{report["source_sha256"]}`','- target/test/Public access: false','']
    lines.append('|field|missing %|mean|std|min|max|q01|q50|q99|')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for col in PHYS:
        c=report['physics'][col]['overall']; q=c['quantiles']
        lines.append(f'|{col}|{100*c["missing_rate"]:.4f}|{c["mean"]:.6g}|{c["std"]:.6g}|{c["min"]:.6g}|{c["max"]:.6g}|{q["0.01"]:.6g}|{q["0.5"]:.6g}|{q["0.99"]:.6g}|')
    md_path=OUTPUT/'trackman_physics_eda.md'; md_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'json':str(json_path),'markdown':str(md_path),'json_sha256':sha(json_path),'markdown_sha256':sha(md_path),'source_sha256':report['source_sha256']},sort_keys=True))
    return 0

if __name__=='__main__': raise SystemExit(main())
