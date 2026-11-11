"""Verify the provenance and denominators of the read-only FGCS-2.3 analysis."""
from pathlib import Path
import csv, hashlib, json

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    report=json.loads((HERE/'results/analysis_summary.json').read_text(encoding='utf-8'))
    assert report['analysis_sha256']==sha(HERE/'analyze_existing.py')
    for item in report['inputs']:
        assert sha(EXP/item['file'].replace('\\','/'))==item['sha256'],item['file']
    protocol=json.loads((EXP/'method_v22/results/protocol.json').read_text(encoding='utf-8'))
    for name,digest in protocol['sources'].items():
        path=EXP/name if name=='fgcs_benchmark.py' else EXP/'method_v22'/name
        assert sha(path)==digest,name
    def rows(name):
        with (HERE/'results'/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    d=rows('decisions.csv');e=rows('migration_episodes.csv');w=rows('stable_windows.csv');c=rows('amortization_scenarios.csv')
    assert (len(d),len(e),len(w),len(c))==(14524,66,14328,24)
    assert sum(r['proposal']=='True' for r in d)==67
    assert sum(r['committed']=='True' for r in d)==66
    assert sum(r['budget_deferred']=='True' for r in d)==1
    assert report['prediction_checks']==24
    assert sum(report['mechanism']['outcomes'].values())==14524
    assert sum(r['model_feasible'] for r in report['agreement'])==14318
    result={'passed':True,'input_hashes_checked':len(report['inputs']),'frozen_code_hashes_checked':4,
            'decisions':len(d),'episodes':len(e),'stable_windows':len(w),'instances':len(c),
            'new_performance_executions':0,'analysis_summary_sha256':sha(HERE/'results/analysis_summary.json'),
            'derived_file_hashes':{p.name:sha(p) for p in sorted((HERE/'results').glob('*.csv'))}}
    (HERE/'results/verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()
