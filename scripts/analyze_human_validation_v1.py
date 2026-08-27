#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path

FIELDS=("evaluation_recognition","pressure_recognition","cue_provenance_stance","explicit_resistance","claimed_behavioral_response")

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def safe(n,d):return n/d if d else None

def cohen(a,b):
    n=len(a)
    if not n:return None
    labs=sorted(set(a)|set(b));po=sum(x==y for x,y in zip(a,b))/n;ca=Counter(a);cb=Counter(b);pe=sum(ca[l]/n*cb[l]/n for l in labs)
    return (po-pe)/(1-pe) if abs(1-pe)>1e-15 else (1.0 if abs(1-po)<1e-15 else None)

def ac1_pair(a,b):
    n=len(a)
    if not n:return None
    labs=sorted(set(a)|set(b));q=len(labs);po=sum(x==y for x,y in zip(a,b))/n
    if q<=1:return 1.0
    ca=Counter(a);cb=Counter(b);p={l:(ca[l]+cb[l])/(2*n) for l in labs};pe=sum(v*(1-v) for v in p.values())/(q-1)
    return (po-pe)/(1-pe) if abs(1-pe)>1e-15 else (1.0 if abs(1-po)<1e-15 else None)

def fleiss(matrix):
    if not matrix:return None
    n=len(matrix);m=len(matrix[0]);labs=sorted({x for r in matrix for x in r});tot=Counter();obs=[]
    for r in matrix:
        c=Counter(r);tot.update(r);obs.append((sum(v*v for v in c.values())-m)/(m*(m-1)))
    po=sum(obs)/n;pe=sum((tot[l]/(n*m))**2 for l in labs)
    return (po-pe)/(1-pe) if abs(1-pe)>1e-15 else (1.0 if abs(1-po)<1e-15 else None)

def ac1_multi(matrix):
    if not matrix:return None
    n=len(matrix);m=len(matrix[0]);labs=sorted({x for r in matrix for x in r});q=len(labs)
    if q<=1:return 1.0
    counts=Counter();pa=[]
    for r in matrix:
        counts.update(r);pairs=[r[i]==r[j] for i in range(m) for j in range(i+1,m)];pa.append(sum(pairs)/len(pairs))
    po=sum(pa)/n;p={l:counts[l]/(n*m) for l in labs};pe=sum(v*(1-v) for v in p.values())/(q-1)
    return (po-pe)/(1-pe) if abs(1-pe)>1e-15 else (1.0 if abs(1-po)<1e-15 else None)

def majority(v):
    c=Counter(v);lab,n=c.most_common(1)[0];return lab if n>=2 else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--admin-key',type=Path,required=True);ap.add_argument('--rater',type=Path,action='append',required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    if len(a.rater)!=3:raise SystemExit('exactly three --rater CSVs required')
    admin={r['item_id']:r for r in rows(a.admin_key)};raters=[]
    for p in a.rater:
        d={r['item_id']:r for r in rows(p)}
        if set(d)!=set(admin):raise SystemExit(f'{p}: item IDs do not match admin key')
        raters.append(d)
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True);result=[]
    for arm in ('all','core','challenge'):
        ids=[i for i,r in admin.items() if arm=='all' or r['sample_arm']==arm]
        for field in FIELDS:
            matrix=[];valid=[]
            for i in ids:
                v=[raters[k][i].get(field,'').strip() for k in range(3)]
                if all(v):matrix.append(v);valid.append(i)
            pair_agree=[]
            for v in matrix:pair_agree.append(sum(v[i]==v[j] for i in range(3) for j in range(i+1,3))/3)
            result.append({'sample_arm':arm,'field':field,'scope':'human_human','judge':'3_humans','n':len(matrix),'raw_agreement':safe(sum(pair_agree),len(pair_agree)),'fleiss_kappa':fleiss(matrix),'gwet_ac1':ac1_multi(matrix),'cohen_kappa':''})
            maj=[majority(v) for v in matrix]
            for judge in ('deepseek','gemini'):
                h=[];m=[]
                for iid,human in zip(valid,maj):
                    machine=admin[iid].get(f'{judge}_{field}','').strip()
                    if human and machine:h.append(human);m.append(machine)
                result.append({'sample_arm':arm,'field':field,'scope':'human_vs_llm','judge':judge,'n':len(h),'raw_agreement':safe(sum(x==y for x,y in zip(h,m)),len(h)),'fleiss_kappa':'','gwet_ac1':ac1_pair(h,m),'cohen_kappa':cohen(h,m)})
    path=out/'agreement_by_field.csv'
    with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    (out/'summary.json').write_text(json.dumps({'items':len(admin),'raters':3,'rows':len(result),'agreement_csv':str(path)},indent=2)+'\n',encoding='utf-8')
    print('HUMAN AGREEMENT: PASS');print('items:',len(admin));print('raters: 3');print('rows:',len(result));print('output:',path)

if __name__=='__main__':main()
