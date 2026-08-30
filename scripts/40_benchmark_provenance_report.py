#!/usr/bin/env python3
from __future__ import annotations
import csv,html,json,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
FINDINGS=ROOT/'analysis/current/findings'; REPORT=ROOT/'reports/iclr-current/index.html'
START='<!-- BENCHMARK_PROVENANCE_AUDIT:START -->'; END='<!-- BENCHMARK_PROVENANCE_AUDIT:END -->'
def field_limit():
  x=sys.maxsize
  while True:
    try:csv.field_size_limit(x);return
    except OverflowError:x//=10
def read_csv(p):
  if not p.is_file():return []
  field_limit()
  with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def integer(v):
  try:return int(float(v))
  except:return 0
def number(v):
  try:return float(v)
  except:return None
def esc(v):return html.escape('' if v is None else str(v))
def pct_fraction(v):
  x=number(v);return '—' if x is None else f'{100*x:.1f}%'
def pct_value(v):
  x=number(v);return '—' if x is None else f'{x:.1f}%'
def first_url(v):
  try:x=json.loads(v or '[]')
  except:return ''
  return str(x[0]) if isinstance(x,list) and x else ''
def section(summary,trials):
  primary=[r for r in summary if r.get('study')=='primary' and r.get('profile')!='llama']
  contaminated=[r for r in trials if r.get('study')=='primary' and integer(r.get('benchmark_repository_lookup'))]
  summary_rows=''.join('<tr>'+f"<td>{esc(r.get('profile'))}</td><td>{integer(r.get('substantive_n'))}</td><td>{integer(r.get('benchmark_repository_lookup_n'))} ({pct_value(r.get('benchmark_repository_lookup_pct'))})</td><td>{integer(r.get('exact_benchmark_task_lookup_n'))}</td><td><strong>{integer(r.get('exact_benchmark_solution_lookup_n'))}</strong></td><td>{integer(r.get('exact_benchmark_solution_pass_n'))}</td><td>{pct_fraction(r.get('exact_benchmark_solution_pass_rate'))}</td><td>{pct_fraction(r.get('non_solution_lookup_pass_rate'))}</td></tr>" for r in primary)
  detail=[]
  for r in contaminated:
    direct=integer(r.get('exact_benchmark_solution_lookup')); u=first_url(r.get('exact_solution_urls')) or first_url(r.get('exact_task_urls')) or first_url(r.get('benchmark_urls'))
    detail.append(f'<tr class="{"benchmark-direct" if direct else ""}"><td>{esc(r.get("profile"))}</td><td><code>{esc(r.get("trial_name"))}</code></td><td><code>{esc(r.get("base_task_id"))}</code></td><td>{esc(r.get("condition"))}</td><td>{esc(r.get("channel"))}</td><td>{"pass" if (number(r.get("overall_pass")) or 0)>0 else "fail"}</td><td>{"exact solution artifact" if direct else "benchmark artifact"}</td><td><details><summary>Evidence</summary><code class="url-evidence">{esc(u)}</code>'+(f'<pre>{esc(r.get("trajectory_evidence_excerpt"))}</pre>' if r.get('trajectory_evidence_excerpt') else '')+'</details></td></tr>')
  return f'''{START}<section id="benchmark-provenance"><h2>Benchmark provenance / answer-recovery audit</h2><div class="callout"><strong>Interpretation.</strong> Generic public GitHub use is not automatically benchmark cheating. Direct access to SWE-Atlas is benchmark contamination; retrieval of the exact task's <code>solution/</code>, <code>gold.patch</code>, or <code>solve.sh</code> is the strongest direct answer-recovery flag and should be excluded or sensitivity-tested for clean capability claims. Same-repository upstream commits/patches are tracked separately as provenance risk.</div><style>#benchmark-provenance .benchmark-direct{{background:#fff2f0}}#benchmark-provenance .url-evidence{{white-space:normal;word-break:break-all}}#benchmark-provenance pre{{white-space:pre-wrap;max-width:900px;font-size:11px}}</style><h3>Primary-study summary</h3><div class="table-scroll"><table><thead><tr><th>Model</th><th>Substantive n</th><th>Any SWE-Atlas access</th><th>Exact task</th><th>Exact solution artifact</th><th>Passes after exact solution access</th><th>Pass rate: exact-solution subset</th><th>Pass rate: remaining runs</th></tr></thead><tbody>{summary_rows}</tbody></table></div><p class="muted">The pass-rate comparison is descriptive: hard tasks can cause both more searching and lower success.</p><h3>Trajectories with SWE-Atlas access</h3><div class="table-scroll"><table><thead><tr><th>Model</th><th>Trajectory</th><th>Base task</th><th>Condition</th><th>Placement</th><th>Outcome</th><th>Audit class</th><th>Trajectory evidence</th></tr></thead><tbody>{''.join(detail) if detail else '<tr><td colspan="8">No SWE-Atlas access detected.</td></tr>'}</tbody></table></div></section>{END}'''
def main():
  if not REPORT.is_file():raise SystemExit(f'Missing report: {REPORT}')
  s=REPORT.read_text(); summary=read_csv(FINDINGS/'benchmark_provenance_summary.csv'); trials=read_csv(FINDINGS/'benchmark_provenance_trials.csv')
  if START in s and END in s:
    a,rest=s.split(START,1);_,b=rest.split(END,1);s=a+b
  payload=section(summary,trials)
  for anchor in ('<section id="replication">','<section id="claims">','<section><h2>Limitations and interpretation boundaries</h2>'):
    if anchor in s:s=s.replace(anchor,payload+'\n'+anchor,1);break
  else:s=s.replace('</main>',payload+'\n</main>',1)
  REPORT.write_text(s);print(REPORT)
if __name__=='__main__':main()
