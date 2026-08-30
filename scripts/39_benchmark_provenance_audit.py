#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,re,sys
from collections import defaultdict
from pathlib import Path
from typing import Any,Iterable
ROOT=Path(__file__).resolve().parents[1]
PROFILES=("claude","fable","codex","llama"); STUDIES=("primary","resource","replication")
BENCHMARK_REPO_RE=re.compile(r"(?:github\.com|raw\.githubusercontent\.com|api\.github\.com)/(?:repos/)?(?:scaleapi|swe-atlas)/(?:swe-atlas)",re.I)
SOLUTION_RE=re.compile(r"/(?:solution)(?:/|\?|$)|(?:^|/)(?:gold\.patch|solve\.sh)(?:\?|$)",re.I)
TEST_RE=re.compile(r"/(?:tests?)(?:/|\?|$)|test_patch\.diff|rubrics\.json|evaluate_rubrics\.py",re.I)
INSTRUCTION_RE=re.compile(r"(?:^|/)instruction\.md(?:\?|$)",re.I)

def field_limit():
  x=sys.maxsize
  while True:
    try: csv.field_size_limit(x); return
    except OverflowError: x//=10

def read_csv(p:Path):
  if not p.is_file(): return []
  field_limit()
  with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write_csv(p:Path,rows):
  p.parent.mkdir(parents=True,exist_ok=True); fields=[]
  for r in rows:
    for k in r:
      if k not in fields: fields.append(k)
  with p.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def truthy(v): return str(v).strip().lower() in {'1','true','yes','y'}
def num(v):
  try:return float(v)
  except:return None
def substantive(r): return truthy(r.get('substantive_usable')) if 'substantive_usable' in r else r.get('terminal_status') in {'completed','safety_refusal'}
def parse_urls(v):
  if not isinstance(v,str) or not v.strip(): return []
  try:x=json.loads(v)
  except:x=None
  return [str(i) for i in x] if isinstance(x,list) else re.findall(r"https?://[^\s'\"<>\)\]]+",v)
def all_strings(v):
  if isinstance(v,str):yield v
  elif isinstance(v,dict):
    for x in v.values():yield from all_strings(x)
  elif isinstance(v,list):
    for x in v:yield from all_strings(x)
def trajectory_excerpt(path_value,base_task_id):
  if not path_value:return ''
  p=Path(path_value).expanduser()
  if not p.is_file():return ''
  try:data=json.loads(p.read_text(encoding='utf-8',errors='replace'))
  except:return ''
  needles=('swe-atlas','gold.patch','solve.sh',str(base_task_id).lower())
  for text in all_strings(data):
    low=text.lower()
    if any(n and n in low for n in needles): return re.sub(r'\s+',' ',text).strip()[:700]
  return ''
def classify(study,profile,r):
  urls=parse_urls(r.get('external_urls')); task=str(r.get('base_task_id') or ''); low=task.lower()
  bench=[u for u in urls if BENCHMARK_REPO_RE.search(u)]
  exact=[u for u in bench if low and low in u.lower()]
  exact_sol=[u for u in exact if SOLUTION_RE.search(u)]
  same=truthy(r.get('same_repository_public_lookup')); patch=truthy(r.get('public_commit_or_patch_lookup'))
  return {'study':study,'profile':profile,'trial_name':r.get('trial_name',''),'base_task_id':task,'repository':r.get('repository',''),'condition':r.get('condition',''),'channel':r.get('channel',''),'overall_pass':r.get('overall_pass',''),
    'benchmark_repository_lookup':int(bool(bench)),'exact_benchmark_task_lookup':int(bool(exact)),'exact_benchmark_solution_lookup':int(bool(exact_sol)),
    'exact_benchmark_test_artifact_lookup':int(any(TEST_RE.search(u) for u in exact)),'exact_benchmark_instruction_lookup':int(any(INSTRUCTION_RE.search(u) for u in exact)),
    'upstream_patch_lookup':int(same and patch),'benchmark_urls':json.dumps(bench,ensure_ascii=False),'exact_task_urls':json.dumps(exact,ensure_ascii=False),'exact_solution_urls':json.dumps(exact_sol,ensure_ascii=False),
    'trajectory_evidence_excerpt':trajectory_excerpt(r.get('trajectory_file',''),task),'trajectory_file':r.get('trajectory_file','')}
def discover(root):
  out=[]
  for study in STUDIES:
    for profile in PROFILES:
      p=root/study/profile/'trials.csv'
      if p.is_file():out.append((study,profile,p));continue
      xs=sorted((root/study).glob(f'**/{profile}/trials.csv')) if (root/study).exists() else []
      if xs:out.append((study,profile,xs[0]))
  return out
def build(source):
  trials=[]
  for study,profile,p in discover(source):
    for r in read_csv(p):
      if substantive(r):trials.append(classify(study,profile,r))
  summary=[]; groups=defaultdict(list)
  for r in trials:groups[(r['study'],r['profile'])].append(r)
  for (study,profile),rows in sorted(groups.items()):
    n=len(rows); direct=[r for r in rows if r['exact_benchmark_solution_lookup']]; rest=[r for r in rows if not r['exact_benchmark_solution_lookup']]
    dp=[num(r['overall_pass']) for r in direct]; dp=[x for x in dp if x is not None]; rp=[num(r['overall_pass']) for r in rest]; rp=[x for x in rp if x is not None]
    summary.append({'study':study,'profile':profile,'substantive_n':n,'benchmark_repository_lookup_n':sum(r['benchmark_repository_lookup'] for r in rows),'benchmark_repository_lookup_pct':100*sum(r['benchmark_repository_lookup'] for r in rows)/n if n else None,'exact_benchmark_task_lookup_n':sum(r['exact_benchmark_task_lookup'] for r in rows),'exact_benchmark_solution_lookup_n':len(direct),'exact_benchmark_solution_lookup_pct':100*len(direct)/n if n else None,'exact_benchmark_solution_pass_n':sum(x>0 for x in dp),'exact_benchmark_solution_pass_rate':sum(dp)/len(dp) if dp else None,'non_solution_lookup_pass_rate':sum(rp)/len(rp) if rp else None,'upstream_patch_lookup_n':sum(r['upstream_patch_lookup'] for r in rows)})
  return trials,summary
def main():
  ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,default=ROOT/'analysis/current/source');ap.add_argument('--output-dir',type=Path,default=ROOT/'analysis/current/findings');a=ap.parse_args();trials,summary=build(a.source_root.expanduser());write_csv(a.output_dir/'benchmark_provenance_trials.csv',trials);write_csv(a.output_dir/'benchmark_provenance_summary.csv',summary)
  print('BENCHMARK PROVENANCE AUDIT')
  for r in summary:
    if r['study']=='primary':print(f"{r['profile']}: SWE-Atlas={r['benchmark_repository_lookup_n']}/{r['substantive_n']} exact_solution={r['exact_benchmark_solution_lookup_n']} upstream_patch={r['upstream_patch_lookup_n']}")
if __name__=='__main__':main()
