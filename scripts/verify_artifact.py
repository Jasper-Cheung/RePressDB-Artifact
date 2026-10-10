#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, math, sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
TEXT_SUFFIXES={'.md','.txt','.json','.csv','.py','.tex','.bib','.cff','.toml','.ps1','.bat','.cls','.bst'}
TEXT_NAMES={'.gitignore','LICENSE','LICENSE_RESULTS'}
REQ=[
 'results/frozen_summaries/multiseed_global_results.csv',
 'results/frozen_summaries/unified_ablation_summary.csv',
 'results/frozen_summaries/v170_assessment.json',
 'results/frozen_summaries/v170_policy_statistics_by_implementation.csv',
 'results/frozen_summaries/crash_fuzz_summary.csv',
 'environment/environment_report.json','environment/hardware_exact.json',
 'paper/make_cleaned_figures.py','paper/main.tex'
]
def gm(a):
    import numpy as np
    a=np.asarray(a,float); return float(np.exp(np.mean(np.log(a))))
def manifest_digest(path):
    data=path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES:
        data=data.replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()
def main():
    missing=[p for p in REQ if not (ROOT/p).exists()]
    if missing: raise SystemExit('missing required files: '+', '.join(missing))
    mg=pd.read_csv(ROOT/'results/frozen_summaries/multiseed_global_results.csv')
    piv=mg.pivot_table(index=['trace','chunk_size','trace_seed'],columns='policy',values='total_time_ns')
    ratio=gm(piv['repressdb_frozen']/piv['shape_ewma_be_frozen'])
    reps=int(mg[mg.policy=='repressdb_frozen'].migration_count.sum()); shapes=int(mg[mg.policy=='shape_ewma_be_frozen'].migration_count.sum())
    if abs(ratio-0.9617)>5e-4: raise SystemExit(f'unexpected replay ratio: {ratio}')
    if (reps,shapes)!=(185,361): raise SystemExit(f'unexpected migration totals: {(reps,shapes)}')
    ass=json.loads((ROOT/'results/frozen_summaries/v170_assessment.json').read_text(encoding='utf-8'))
    if int(ass['total_measured_queries']) != 16908066: raise SystemExit('unexpected measured-query count')
    hw=json.loads((ROOT/'environment/hardware_exact.json').read_text(encoding='utf-8'))
    if 'i7-14700HX' not in hw['processor']['Name']: raise SystemExit('hardware metadata mismatch')
    manifest=ROOT/'manifests/SHA256SUMS.txt'
    if manifest.exists():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.startswith('#'): continue
            h,rel=line.split('  ',1); p=ROOT/rel
            if not p.exists(): raise SystemExit(f'manifest file missing: {rel}')
            if manifest_digest(p)!=h: raise SystemExit(f'checksum mismatch: {rel}')
    print(f'replay ratio: {ratio:.4f}')
    print(f'migrations: RePressDB={reps}, Shape-EWMA-BE={shapes}')
    print(f'measured queries: {int(ass["total_measured_queries"]):,}')
    print('artifact verification: PASS')
if __name__=='__main__': main()

