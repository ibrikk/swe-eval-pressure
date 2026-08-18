#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re, tomllib
from pathlib import Path
from typing import Any

SUPPORTED = {'.py':'#','.yml':'#','.yaml':'#','.toml':'#','.am':'#','.sh':'#','.rb':'#',
             '.go':'//','.ts':'//','.tsx':'//','.js':'//','.jsx':'//','.c':'//','.cc':'//',
             '.cpp':'//','.h':'//','.hpp':'//','.cue':'//','.rs':'//','.java':'//'}
TEST_RE = re.compile(r'(^|/)(test|tests|spec|specs)(/|$)', re.I)

def patch_sections(text: str) -> list[tuple[str,str]]:
    out=[]
    for chunk in re.split(r'(?=^diff --git )', text, flags=re.M):
        m=re.match(r'diff --git a/(.+?) b/(.+?)\n', chunk)
        if m: out.append((m.group(2), chunk))
    return out

def source_target(task: Path) -> tuple[str,str,list[str]]:
    text=(task/'solution/gold.patch').read_text(encoding='utf-8', errors='replace')
    all_paths=[]; candidates=[]
    for path, section in patch_sections(text):
        if path not in all_paths: all_paths.append(path)
        ext=Path(path).suffix.lower()
        is_new='\nnew file mode ' in section[:500] or '\n--- /dev/null\n' in section[:1200]
        if ext in SUPPORTED and not TEST_RE.search(path) and not is_new:
            candidates.append(path)
    if not candidates:
        raise ValueError('no safe existing non-test source path in gold patch')
    path=candidates[0]
    return path, SUPPORTED[Path(path).suffix.lower()], all_paths

def base_image(task: Path, parsed: dict[str,Any]) -> str:
    env=parsed.get('environment',{})
    if isinstance(env,dict) and env.get('docker_image'): return str(env['docker_image'])
    text=(task/'environment/Dockerfile').read_text(encoding='utf-8')
    m=re.search(r'(?m)^\s*FROM\s+(\S+)', text)
    if not m: raise ValueError('missing base image')
    return m.group(1)

def repo_from_image(image: str) -> str:
    m=re.search(r'swe_atlas_RF_(.+?)_[0-9a-f]{24}_', image)
    return m.group(1).replace('_','/') if m else ''

def workspace_root(task: Path) -> str:
    text=(task/'tests/test.sh').read_text(encoding='utf-8',errors='replace')
    m=re.search(r'(?m)^WORKSPACE=\"([^\"]+)\"', text)
    if not m: raise ValueError('tests/test.sh does not declare WORKSPACE')
    value=m.group(1).strip()
    if not value.startswith('/'): raise ValueError(f'workspace must be absolute: {value}')
    return value

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',type=Path,required=True); args=ap.parse_args()
    root=args.project_root; task_root=root/'vendor/rf'; manifest_root=root/'manifests'; manifest_root.mkdir(exist_ok=True)
    records=[]; errors=[]
    for task in sorted(task_root.glob('task-*')):
        try:
            required=['task.toml','instruction.md','solution/gold.patch','tests/test.sh','environment/Dockerfile']
            missing=[x for x in required if not (task/x).exists()]
            if missing: raise ValueError(f'missing {missing}')
            parsed=tomllib.loads((task/'task.toml').read_text(encoding='utf-8'))
            target, prefix, patch_paths=source_target(task)
            image=base_image(task,parsed)
            instruction=(task/'instruction.md').read_text(encoding='utf-8',errors='replace').strip()
            records.append({
                'task_id':task.name,
                'category':parsed.get('metadata',{}).get('category',''),
                'difficulty':parsed.get('metadata',{}).get('difficulty',''),
                'tags':parsed.get('metadata',{}).get('tags',[]),
                'repository':repo_from_image(image),
                'base_image':image,
                'workspace_root':workspace_root(task),
                'source_target':target,
                'source_comment_prefix':prefix,
                'source_extension':Path(target).suffix.lower(),
                'gold_patch_files':patch_paths,
                'instruction_preview':instruction.splitlines()[0][:240] if instruction else '',
                'eligible':True,
            })
        except Exception as e:
            errors.append({'task_id':task.name,'error':f'{type(e).__name__}: {e}','eligible':False})
    payload={'schema_version':'1.0','task_count':len(records),'invalid_count':len(errors),'tasks':records,'invalid_tasks':errors}
    (manifest_root/'task_registry.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (manifest_root/'task_registry.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['task_id','repository','difficulty','workspace_root','source_target','source_extension','instruction_preview','eligible']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in records: w.writerow({k:r.get(k,'') for k in fields})
        for r in errors: w.writerow({k:r.get(k,'') for k in fields})
    print(f'Valid refactoring tasks: {len(records)}')
    print(f'Invalid tasks: {len(errors)}')
    print(f'Registry: {manifest_root/"task_registry.json"}')
    if errors:
        for e in errors: print(f"  {e['task_id']}: {e['error']}")

if __name__=='__main__': main()
