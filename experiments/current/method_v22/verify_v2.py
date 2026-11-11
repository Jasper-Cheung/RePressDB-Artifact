"""Independent checks of the complete revised measurement archive."""
from pathlib import Path
import hashlib,itertools,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fgcs_benchmark as old
import benchmark_v2 as bench
import numpy as np
import pandas as pd

def run():
    root=bench.OUT;bench.check_freeze();protocol=json.loads((root/'protocol.json').read_text())
    assert bench.sha(bench.HERE.parent/'method_v2/results/calibration_summary.json')==protocol['calibration_origin']['summary_sha256']
    assert bench.sha(bench.HERE/'efficient_executor.py')==protocol['calibration_origin']['executor_sha256']
    original_sources=json.loads((old.HERE/'source_manifest.json').read_text(encoding='utf-8'))
    for s in original_sources:assert bench.sha(old.SRC/s['file'])==s['sha256']
    manifest=pd.read_csv(old.OUT/'chunk_manifest.csv').set_index('chunk_id')
    for size in (8192,16384):
        z=np.load(old.OUT/f'values_{size}.npz')
        for cid,values in zip(z['chunk_ids'],z['values']):assert hashlib.sha256(values.tobytes()).hexdigest()==manifest.loc[str(cid),'raw_sha256']
    paths=sorted((root/'cells').glob('*/summary.json'));assert len(paths)==144
    summaries=[json.loads(p.read_text()) for p in paths]
    expected={(s,g,bench.FAMILIES[f],p) for s,g,f,p in itertools.product([8192,16384],range(3),range(4),bench.POLICIES)}
    assert {(s['size'],s['group'],s['family'],s['policy']) for s in summaries}==expected
    calibration=json.loads((root/'calibration_summary.json').read_text())
    for name,digest in calibration['hashes'].items():assert bench.sha(root/name)==digest
    records=json.loads((root/'query_calibration.json').read_text());assert len(records)==8352
    for r in records:
        samples=np.asarray(r['samples']);assert len(samples)==64 and (samples>0).all()
        assert np.mean(samples)==r['mean_ns'] and np.median(samples)==r['median_ns']
        assert np.percentile(samples,95)==r['p95_ns']
    migration=pd.read_csv(root/'durable_calibration.csv');assert len(migration)==4320
    assert (migration.groupby(['chunk_id','source','target']).size()==3).all()
    cells=pd.DataFrame(summaries)
    for _,rows in cells.groupby(['size','group','family']):
        assert len(set(rows.seed))==1
        assert len({tuple(x) for x in rows.initial_codecs})==1
    values=0;actions=0;rewrite=0;decisions=0
    for path,s in zip(paths,summaries):
        assert s['sources']==protocol['sources'] and s['correct'] and s['integrity']['pass']
        z=np.load(path.parent/'latencies.npz');a=z['service_ns']
        assert len(a)==s['n_queries']==160000
        assert np.array_equal(a,z['query_ns']+z['control_ns']+z['migration_ns'])
        for array,field in [('service_ns','service_ns'),('query_ns','query_ns'),('control_ns','controller_ns'),('migration_ns','migration_ns')]:assert int(z[array].sum())==s[field]
        assert np.percentile(a,99)==s['p99_ns'] and np.percentile(a,99.9)==s['p999_ns'] and int(a.max())==s['max_ns']
        assert int((z['migration_ns']>0).sum())==s['migration_count']
        values+=len(a)
        if s['migration_count']:
            m=pd.read_csv(path.parent/'migrations.csv');assert len(m)==s['migration_count']
            assert (m.durable_transactions==3).all()
            assert int(m.rewrite_bytes.sum())==s['rewrite_bytes']
            assert int(m.migration_time_ns.sum())==s['migration_ns']
            assert (m.groupby(m.event//1000).rewrite_bytes.sum()<=s['budget_limit_bytes']).all()
            assert np.array_equal(z['migration_ns'][m.event.to_numpy(int)],m.migration_time_ns.to_numpy())
            actions+=len(m);rewrite+=int(m.rewrite_bytes.sum())
        else:assert s['rewrite_bytes']==0
        if s['policy']=='risk' and s['family']!='R2_D_persistent_shift':
            old_cell=bench.HERE.parent/'method_v21/results/cells'/path.parent.name
            old_summary=json.loads((old_cell/'summary.json').read_text())
            assert s['migration_count']==old_summary['migration_count'] and s['rewrite_bytes']==old_summary['rewrite_bytes']
            if s['migration_count']:
                previous=pd.read_csv(old_cell/'migrations.csv')
                columns=['event','chunk_id','source','target','rewrite_bytes']
                assert m[columns].equals(previous[columns]),path.parent.name
        if s['policy']=='risk':
            logs=json.loads((path.parent/'decisions.json').read_text())
            for _,rows in logs.items():
                for d in rows:
                    assert d['local_count']%256==0
                    if d['proposal'] is not None:
                        assert d['feasible'][d['proposal']]
                        if d['reason']!='risk_repair':assert d['credit']>=1.25*d['admission_price']
                    decisions+=1
    assert values==23040000
    stats=pd.read_csv(root/'paired_statistics.csv')
    for _,s in stats.iterrows():
        table=cells.pivot(index=['size','group','family'],columns='policy',values=s.metric)
        ratio=table.risk/table[s.denominator]
        assert np.isclose(np.exp(np.log(ratio).mean()),s.gm_ratio,rtol=1e-12)
        assert int((ratio<1).sum())==s.wins
    paired=pd.read_csv(bench.HERE/'validation/paired_migrations.csv');assert len(paired)==240
    assert (paired.original_commits==6).all() and (paired.efficient_commits==3).all()
    assert np.array_equal(paired.original_bytes,paired.efficient_bytes)
    faults=json.loads((bench.HERE/'validation/combined_fault_summary.json').read_text());assert faults['trials']==faults['passed']==492
    equivalence=json.loads((bench.HERE/'implementation_equivalence.json').read_text());assert equivalence['passed'] and equivalence['events']==2880000 and equivalence['cached_source_sha256']==bench.sha(bench.HERE/'risk_controller.py')
    output=dict(passed=True,cells=144,query_events=values,original_sources_verified=len(original_sources),query_calibration_records=8352,query_calibration_events=567936,
                durable_observations=4320,paired_executor_observations=240,policy_actions=actions,policy_rewrite_bytes=rewrite,
                risk_decisions_checked=decisions,fault_trials=492,old_cleanup_failure_reproduced=True,
                sources=protocol['sources'],calibration_hashes=calibration['hashes'])
    old.write(root/'verification.json',output);print(json.dumps(output,indent=2))
if __name__=='__main__':run()
