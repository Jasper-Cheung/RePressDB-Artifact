"""One post-hoc cost-matched control, using the unchanged frozen driver.

Only the migration-cost input supplied to Horizon-RePress changes. No function
on the timed request path is wrapped or edited. All primary records remain intact.
"""
from pathlib import Path
import argparse,datetime,hashlib,itertools,json,shutil,subprocess,sys
import numpy as np
import pandas as pd
import fgcs_benchmark as bench

HERE=Path(__file__).resolve().parent;PRIMARY=HERE/'results';OUT=HERE/'cost_matched_results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def setup():
    OUT.mkdir(exist_ok=True)
    for name in ['protocol.json','chunk_manifest.csv','values_8192.npz','values_16384.npz','query_calibration.csv','durable_calibration.csv','codec_sizes.csv']:
        shutil.copy2(PRIMARY/name,OUT/name)
    bench.write(OUT/'extension_protocol.json',{'stage':'post-hoc targeted cost-matched control','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'policy':'Horizon-RePress with full durable costs','changed_input':'the archived representation-rewrite cost table is replaced by medians of the same full durable calibration supplied to RePressDB','unchanged':'all horizon controller settings, request sequences, initial choices, active-view executor, write budget, timing implementation and processor affinity','rationale':'separate a cost-table change from an admission-rule change; this comparison was added after the primary 90-cell results were inspected','planned_cells':18,'queries_per_cell':160000,'source_driver_sha256':sha(HERE/'fgcs_benchmark.py'),'extension_sha256':sha(Path(__file__))})

def worker(size,group,family):
    import psutil
    psutil.Process().cpu_affinity([0]);bench.OUT=OUT
    normal_read=pd.read_csv
    dur=normal_read(PRIMARY/'durable_calibration.csv').groupby(['chunk_id','source','target']).ns.median()
    rows=[];codecs=sorted(set(dur.index.get_level_values('source')))
    for cid in sorted(set(dur.index.get_level_values('chunk_id'))):
        for a,b in itertools.product(codecs,codecs):rows.append({'chunk_id':cid,'current_codec':a,'target_codec':b,'median_migration_ns':0. if a==b else float(dur.loc[(cid,a,b)])})
    full_table=pd.DataFrame(rows)
    target=(bench.SRC/'results_jss_major_revision_v151/j4_calibration/fresh_full_calibration/measured_migration_costs.csv').resolve()
    def read_input(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).resolve()==target:return full_table.copy()
        return normal_read(path,*args,**kwargs)
    pd.read_csv=read_input
    try:bench.cell(size,group,family,'legacy_repress')
    finally:pd.read_csv=normal_read
    d=OUT/'cells'/f'{size}_{group}_{family}_legacy_repress';p=d/'summary.json';s=json.loads(p.read_text());s['policy']='horizon_full_cost';s['extension_sha256']=sha(Path(__file__));s['migration_input']='full durable transaction medians';bench.write(p,s)

def aggregate():
    paths=list((OUT/'cells').glob('*/summary.json'));assert len(paths)==18
    rows=[json.loads(p.read_text()) for p in paths];df=pd.DataFrame(rows);df.to_csv(OUT/'summary.csv',index=False)
    for p,s in zip(paths,rows):
        a=np.load(p.parent/'latencies.npz')['service_ns'];assert len(a)==160000 and int(a.sum())==s['service_ns'] and s['correct'] and s['integrity']['pass'];assert s['extension_sha256']==sha(Path(__file__))
        if s['migration_count']:
            m=pd.read_csv(p.parent/'migrations.csv');assert len(m)==s['migration_count'] and int(m.rewrite_bytes.sum())==s['rewrite_bytes'];assert (m.groupby(m.event//1000).rewrite_bytes.sum()<=s['budget_limit_bytes']).all()
    primary=pd.read_csv(PRIMARY/'summary.csv');idx=['size','group','family'];r=primary[primary.policy=='repressdb'].set_index(idx);h=df.set_index(idx).loc[r.index]
    summary={'stage':'post-hoc cost-matched control','cells':len(df),'verified_queries':int(df.n_queries.sum()),'horizon_full_cost_migrations':int(df.migration_count.sum()),'horizon_full_cost_rewrite_bytes':int(df.rewrite_bytes.sum()),'repressdb_migrations':int(r.migration_count.sum()),'repressdb_rewrite_bytes':int(r.rewrite_bytes.sum()),'service_ratio_repressdb_over_cost_matched':float(np.exp(np.log(r.service_ns/h.service_ns).mean())),'p99_ratio_repressdb_over_cost_matched':float(np.exp(np.log(r.p99_ns/h.p99_ns).mean())),'all_recorded_checks_pass':True,'note':'separate post-hoc executions; ratios are descriptive and not part of the primary pre-scheduled five-policy intervals'}
    bench.write(OUT/'comparison.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['run','worker','aggregate']);ap.add_argument('--size',type=int);ap.add_argument('--group',type=int);ap.add_argument('--family',type=int);a=ap.parse_args()
    if a.mode=='worker':worker(a.size,a.group,a.family)
    elif a.mode=='aggregate':aggregate()
    else:
        setup()
        for size,g,f in itertools.product([8192,16384],range(3),range(3)):subprocess.run([sys.executable,str(Path(__file__)),'worker','--size',str(size),'--group',str(g),'--family',str(f)],check=True)
        aggregate()
