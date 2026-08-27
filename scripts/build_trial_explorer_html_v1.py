#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, html, json
from pathlib import Path

PROFILES=("claude","fable","codex","llama")
FIELDS=("evaluation_recognition","pressure_recognition","cue_provenance_stance","explicit_resistance","claimed_behavioral_response")
PROCESS=("raw_tool_calls","validation_command_calls","test_command_calls","input_tokens","output_tokens","seeded_cue_removed_or_modified")

def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def resolve(row, results_root):
    rp=str(row.get("result_path") or ""); rel=str(row.get("trajectory_file") or "")
    if not rp or not rel: return None
    x=Path(rp).expanduser().parent/rel
    if x.is_file(): return x.resolve()
    parts=Path(rp).parts
    if "results" in parts:
        i=parts.index("results")
        x=results_root.joinpath(*parts[i+1:]).parent/rel
        if x.is_file(): return x.resolve()
    return None

def cons(v):
    f=v.get("consensus",{}).get("fields",{})
    return {k:{"status":f.get(k,{}).get("status","missing"),"label":f.get(k,{}).get("label")} for k in FIELDS}

def judge(v):
    if not v: return {}
    e=v.get("final_cache_entry"); j=e.get("judgment") if isinstance(e,dict) else {}
    out={}
    for k in FIELDS:
        x=j.get(k,{}) if isinstance(j,dict) else {}
        out[k]={"label":x.get("label"),"evidence":x.get("evidence",[])}
    return out

def main():
    ap=argparse.ArgumentParser(); root=Path(__file__).resolve().parents[1]
    ap.add_argument("--project-root",type=Path,default=root); ap.add_argument("--results-root",type=Path); ap.add_argument("--output",type=Path)
    a=ap.parse_args(); root=a.project_root.expanduser().resolve()
    behavior=root/"analysis/frozen/historical-primary-repaired-llama-20260826"
    final=root/"analysis/semantic-multijudge-v1/final-repaired-llama-v1"
    results=(a.results_root or root/"results").expanduser().resolve()
    output=(a.output or root/"analysis/semantic-multijudge-v1/trial-explorer-v1.html").expanduser().resolve()
    b={}
    for profile in PROFILES:
        rows=load(behavior/profile/"trials.json")
        if len(rows)!=700: raise SystemExit(f"{profile}: expected 700 rows")
        for r in rows: b[(profile,str(r["trial_name"]))]=r
    c={}
    for path in (final/"consensus").glob("*.json"):
        v=load(path); c[(str(v["profile"]),str(v["trial_name"]))]=v
    j={}
    for path in (final/"jobs").glob("*.json"):
        v=load(path); j[(str(v["profile"]),str(v["trial_name"]),str(v["judge_family"]))]=v
    if (len(b),len(c),len(j))!=(2800,2776,5552): raise SystemExit(f"unexpected counts {(len(b),len(c),len(j))}")
    records=[]
    for (profile,trial),r in sorted(b.items()):
        cc=c.get((profile,trial)); tp=resolve(r,results)
        records.append({"profile":profile,"trial_name":trial,"base_task_id":r.get("base_task_id"),"task_name":r.get("task_name"),"repository":r.get("repository"),"condition":r.get("condition"),"placement":r.get("channel"),"pressure_type":r.get("pressure_type"),"terminal_status":r.get("terminal_status"),"substantive_usable":bool(r.get("substantive_usable")),"overall_pass":r.get("overall_pass"),"consensus":cons(cc) if cc else {},"deepseek":judge(j.get((profile,trial,"deepseek"))),"gemini":judge(j.get((profile,trial,"gemini"))),"process":{k:r.get(k) for k in PROCESS},"trajectory_path":str(tp) if tp else "","trajectory_uri":tp.as_uri() if tp else ""})
    payload=json.dumps(records,ensure_ascii=False,separators=(",",":")).replace("</","<\\/")
    fields=json.dumps(FIELDS)
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SWE-EvalPressure Trial Explorer</title><style>body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f6f8fb;color:#172033}header{padding:24px;background:#101828;color:#fff}main{padding:16px;max-width:1600px;margin:auto}.controls{display:grid;grid-template-columns:2fr repeat(5,1fr);gap:8px;position:sticky;top:0;background:#f6f8fb;padding:8px 0}input,select,button{padding:7px;border:1px solid #d0d5dd;border-radius:6px;background:#fff}.wrap{overflow:auto;max-height:70vh;background:#fff;border:1px solid #dce3ed}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px;border-bottom:1px solid #eaecf0;white-space:nowrap;text-align:left}th{position:sticky;top:0;background:#f9fafb}tbody tr{cursor:pointer}tbody tr:hover{background:#f4f7ff}.drawer{position:fixed;right:0;top:0;bottom:0;width:min(760px,95vw);background:#fff;box-shadow:-8px 0 25px #0002;transform:translateX(105%);transition:.15s;overflow:auto}.drawer.open{transform:none}.head{position:sticky;top:0;background:#fff;padding:12px;border-bottom:1px solid #ddd;display:flex;justify-content:space-between}.body{padding:14px}blockquote{white-space:pre-wrap;background:#f8fafc;border-left:3px solid #98a2b3;padding:7px;margin:5px 0}.sem td,.sem th{white-space:normal;vertical-align:top}</style></head><body><header><h1>SWE-EvalPressure - Historical Trial Explorer</h1><div>Frozen cohort - qualitative inspection only - no inferential recomputation</div></header><main><div class="controls"><input id="q" placeholder="Search task, trial, labels, evidence..."><select id="p"><option value="">All models</option></select><select id="c"><option value="">All conditions</option></select><select id="pl"><option value="">All placements</option></select><select id="s"><option value="">All semantic states</option><option value="eval">Eval recognized</option><option value="pressure">Pressure recognized</option><option value="distrust">Distrust</option><option value="resist">Resistance</option><option value="response">Claimed response</option><option value="unresolved">Unresolved</option></select><select id="m"><option value="">Cue modification: all</option><option value="1">Modified</option><option value="0">Not modified</option></select></div><p id="sum"></p><div class="wrap"><table><thead><tr><th>Model</th><th>Task</th><th>Condition</th><th>Placement</th><th>Pass</th><th>Eval</th><th>Pressure</th><th>Stance</th><th>Resistance</th><th>Response</th><th>Cue mod.</th><th>Tools</th><th>Validation</th><th>Input tokens</th></tr></thead><tbody id="rows"></tbody></table></div></main><div class="drawer" id="d"><div class="head"><b id="title"></b><button id="close">Close</button></div><div class="body" id="detail"></div></div><script id="data" type="application/json">__DATA__</script><script>const D=JSON.parse(document.getElementById('data').textContent),F=__FIELDS__,$=x=>document.getElementById(x),e=x=>String(x??'').replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));function cf(r,f){return r.consensus?.[f]||{status:'missing',label:null}}function L(r,f){let x=cf(r,f);return x.status==='agreement'?(x.label||''):`[${x.status}]`}function Y(v){return v===1||v==='1'||v===true}function P(r){if(!r.substantive_usable||r.overall_pass===null||r.overall_pass==='')return '-';return Number(r.overall_pass)>0?'PASS':'FAIL'}function U(r){return F.some(f=>cf(r,f).status!=='agreement')}function R(r){let x=cf(r,'claimed_behavioral_response');return x.status==='agreement'&&x.label&&x.label!=='none_observed'}function T(r){let a=[r.profile,r.trial_name,r.base_task_id,r.task_name,r.repository,r.condition,r.placement,r.pressure_type];F.forEach(f=>{a.push(L(r,f));['deepseek','gemini'].forEach(j=>{let x=r[j]?.[f];a.push(x?.label||'');(x?.evidence||[]).forEach(z=>a.push(z.quote||''))})});return a.join(' ').toLowerCase()}D.forEach(r=>r._t=T(r));function fill(id,k){[...new Set(D.map(r=>r[k]).filter(Boolean))].sort().forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=v;$(id).appendChild(o)})}fill('p','profile');fill('c','condition');fill('pl','placement');function SM(r,v){if(!v)return true;if(v==='eval')return L(r,'evaluation_recognition')==='observed';if(v==='pressure')return L(r,'pressure_recognition')==='observed';if(v==='distrust')return L(r,'cue_provenance_stance')==='untrusted_or_suspicious';if(v==='resist')return L(r,'explicit_resistance')==='observed';if(v==='response')return R(r);if(v==='unresolved')return U(r);return true}function filt(){let q=$('q').value.toLowerCase(),p=$('p').value,c=$('c').value,pl=$('pl').value,s=$('s').value,m=$('m').value;return D.filter(r=>(!q||r._t.includes(q))&&(!p||r.profile===p)&&(!c||r.condition===c)&&(!pl||r.placement===pl)&&SM(r,s)&&(m===''||(m==='1')===Y(r.process.seeded_cue_removed_or_modified)))}function render(){let d=filt();$('sum').textContent=`Showing ${d.length} / ${D.length}`;$('rows').innerHTML=d.slice(0,1200).map(r=>`<tr data-i="${D.indexOf(r)}"><td>${e(r.profile)}</td><td>${e(r.base_task_id)}</td><td>${e(r.condition)}</td><td>${e(r.placement)}</td><td>${P(r)}</td><td>${e(L(r,'evaluation_recognition'))}</td><td>${e(L(r,'pressure_recognition'))}</td><td>${e(L(r,'cue_provenance_stance'))}</td><td>${e(L(r,'explicit_resistance'))}</td><td>${e(L(r,'claimed_behavioral_response'))}</td><td>${Y(r.process.seeded_cue_removed_or_modified)?'YES':''}</td><td>${e(r.process.raw_tool_calls)}</td><td>${e(r.process.validation_command_calls)}</td><td>${e(r.process.input_tokens)}</td></tr>`).join('');document.querySelectorAll('tbody tr').forEach(tr=>tr.onclick=()=>show(D[+tr.dataset.i]))}function ev(j,f){let x=j?.[f];if(!x)return 'missing';return `<b>${e(x.label||'missing')}</b>`+(x.evidence||[]).map(z=>`<blockquote>step ${e(z.step_index)} - ${e(z.quote)}</blockquote>`).join('')}function show(r){$('title').textContent=`${r.profile} - ${r.base_task_id}`;let h=`<p><b>Trial:</b> ${e(r.trial_name)}<br><b>Condition:</b> ${e(r.condition)} - <b>Placement:</b> ${e(r.placement)} - <b>Outcome:</b> ${P(r)}</p><table class="sem"><tr><th>Field</th><th>Consensus</th><th>DeepSeek</th><th>Gemini</th></tr>`+F.map(f=>`<tr><td><b>${e(f)}</b></td><td>${e(L(r,f))}</td><td>${ev(r.deepseek,f)}</td><td>${ev(r.gemini,f)}</td></tr>`).join('')+`</table><h3>Process</h3><pre>${e(JSON.stringify(r.process,null,2))}</pre><h3>Trajectory</h3>`;if(r.trajectory_uri)h+=`<p><a href="${e(r.trajectory_uri)}">Open resolved trajectory</a></p>`;h+=`<code>${e(r.trajectory_path||'not resolved on this machine')}</code>`;$('detail').innerHTML=h;$('d').classList.add('open')}['q','p','c','pl','s','m'].forEach(id=>$(id).addEventListener(id==='q'?'input':'change',render));$('close').onclick=()=>$('d').classList.remove('open');render();</script></body></html>'''
    page=page.replace('__DATA__',payload).replace('__FIELDS__',fields)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(page,encoding='utf-8')
    print('TRIAL EXPLORER: PASS');print('records:',len(records));print('output:',output);print('sha256:',sha(output))

if __name__=='__main__': main()
