"""Read-only verification of archived source, raw latency arrays and action logs."""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;R=ROOT/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
freeze=json.loads((R/'freeze.json').read_text());assert sha(ROOT/'fgcs_benchmark.py')==freeze['code_sha256']
for fn in ['query_calibration.csv','durable_calibration.csv']:
    assert sha(R/fn)==freeze[fn]
assert sha(R/'protocol.json')==freeze['protocol_sha256']
assert sha(R/'chunk_manifest.csv')==freeze['manifest_sha256']
sources=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
for s in sources:assert sha(ROOT/'source'/s['file'])==s['sha256']
df=pd.read_csv(R/'summary.csv');families=['R1_A_gradual_periodic','R1_B_mean_reverting','R1_C_noisy_bursts'];policies=['static','shape','work_function','legacy_repress','repressdb']
assert set(map(tuple,df[['size','group','family','policy']].itertuples(index=False,name=None)))==set(itertools.product([8192,16384],range(3),families,policies))
queries=0;migrations=0;payload=0
for row in df.itertuples(index=False):
    p=R/'cells'/f'{row.size}_{row.group}_{families.index(row.family)}_{row.policy}'
    s=json.loads((p/'summary.json').read_text());assert s['code_sha256']==freeze['code_sha256'] and s['correct'] and s['integrity']['pass']
    a=np.load(p/'latencies.npz')['service_ns'];assert len(a)==160000 and int(a.sum())==s['service_ns'] and np.all(a>0)
    assert np.isclose(np.percentile(a,99),s['p99_ns']) and np.isclose(np.percentile(a,99.9),s['p999_ns'])
    queries+=len(a)
    if s['migration_count']:
        m=pd.read_csv(p/'migrations.csv');assert len(m)==s['migration_count'];assert int(m.rewrite_bytes.sum())==s['rewrite_bytes'];assert int(m.migration_time_ns.sum())==s['migration_ns'];assert (m.groupby(m.event//1000).rewrite_bytes.sum()<=s['budget_limit_bytes']).all()
        migrations+=len(m);payload+=int(m.rewrite_bytes.sum())
    else:assert s['rewrite_bytes']==0 and s['migration_ns']==0
mf=pd.read_csv(R/'chunk_manifest.csv').set_index('chunk_id')
for size in [8192,16384]:
    z=np.load(R/f'values_{size}.npz')
    for cid,values in zip(z['chunk_ids'],z['values']):assert hashlib.sha256(values.tobytes()).hexdigest()==mf.loc[str(cid),'raw_sha256']
b=pd.read_csv(R/'cost_boundary/paired_costs.csv');assert len(b)==240 and b.groupby(['chunk_id','source','target']).size().eq(3).all()
old=pd.read_csv(ROOT/'archival_evidence/fault_injection_summary.csv');fuzz=pd.read_csv(ROOT/'archival_evidence/crash_fuzz_summary.csv');assert old.n_trials.sum()==600 and fuzz.trials.sum()==500 and (fuzz.trials==fuzz.passed).all()
eq=json.loads((ROOT/'implementation_equivalence.json').read_text());assert eq['pass'] and eq['events']==400000
report={'pass':True,'source_files_verified':len(sources),'workload_cells':len(df),'raw_latency_entries':queries,'logged_policy_migrations':migrations,'logged_policy_payload_bytes':payload,'paired_boundary_observations':len(b),'original_recovery_trials':int(old.n_trials.sum()+fuzz.trials.sum()),'benchmark_code_sha256':freeze['code_sha256'],'all_completed_cells_preserved':True}
(R/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
