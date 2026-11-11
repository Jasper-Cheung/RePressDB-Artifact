from pathlib import Path
import argparse,hashlib,itertools,json,os,shutil,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fgcs_benchmark as bench
import numpy as np
import pandas as pd
import psutil
from efficient_executor import EfficientMigration

HERE=Path(__file__).resolve().parent;OUT=HERE/'validation'
STAGES=['after_intent','after_encode','after_validate','after_data_publish',
        'after_publish','after_commit','after_post_validate','after_retire','after_unlink_previous','after_cleanup']
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def remove_store(path):
    assert OUT.resolve() in path.resolve().parents
    shutil.rmtree(path)
def initialize(path,m,j,chunk,codec):
    raw=m.b1.SingleActiveChunkStore(path,m.base,recover_on_open=False)
    raw.create_chunk(chunk.chunk_id,chunk.series_index,chunk.values,codec)
    return raw,j.ActiveViewCachedStore(raw,m.base)
def check(path,m,j,chunk,expected):
    raw=m.b1.SingleActiveChunkStore(path,m.base,recover_on_open=True)
    store=j.ActiveViewCachedStore(raw,m.base)
    assert raw._row(chunk.chunk_id)['active_codec']==expected
    assert np.array_equal(store.read_all(chunk.chunk_id),chunk.values)
    pools=m.base.build_queries(chunk.values,chunk.chunk_id,80731,4)
    for q in pools:
        actual,*_=store.execute_query(chunk.chunk_id,q)
        assert m.base.results_equal(actual,m.base.execute_on_array(chunk.values,q))
    assert raw.integrity_check()['pass']
    # A second recovery must be harmless.
    store.recover();assert raw.integrity_check()['pass'];store.close()
    return len(pools)
def crash_worker(size,source,target,stage,path):
    p,m,j,_,_,_=bench.context();chunk=bench.group_chunks(m,size,0)[0]
    raw,store=initialize(path,m,j,chunk,source)
    def hook(where,_):
        if where==stage:os._exit(73)
    EfficientMigration(store,m.b1).migrate_profiled(chunk.chunk_id,target,hook)
    raise AssertionError('Requested termination point was not reached')
def faults():
    OUT.mkdir(exist_ok=True);p,m,j,_,_,_=bench.context();codecs=tuple(sorted(m.base.representation_registry(3)))
    bench.write(OUT/'fault_protocol.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        stages=STAGES,controlled='all 20 directed pairs at each size and all ten stages',
        abrupt='whole-to-block and block-to-whole at each size and all ten stages',
        source_sha256=digest(HERE/'efficient_executor.py'),script_sha256=digest(Path(__file__))))
    rows=[];path=OUT/'fault_store'
    for size in (8192,16384):
        chunk=bench.group_chunks(m,size,0)[0]
        for source,target in itertools.permutations(codecs,2):
            for stage in STAGES:
                if path.exists():remove_store(path)
                raw,store=initialize(path,m,j,chunk,source)
                class Injected(Exception):pass
                def hook(where,_):
                    if where==stage:raise Injected(stage)
                try:EfficientMigration(store,m.b1).migrate_profiled(chunk.chunk_id,target,hook)
                except Injected:pass
                else:raise AssertionError('Missing fault hook')
                store.close();expected=source if STAGES.index(stage)<5 else target
                queries=check(path,m,j,chunk,expected)
                rows.append(dict(size=size,source=source,target=target,stage=stage,mode='exception',correct_queries=queries,passed=True))
        print('CONTROLLED',size,len(rows),flush=True)
    for size in (8192,16384):
        chunk=bench.group_chunks(m,size,0)[0]
        for source,target in [('whole_lz4','block_lz4_1024'),('block_lz4_1024','whole_lz4')]:
            for stage in STAGES:
                if path.exists():remove_store(path)
                r=subprocess.run([sys.executable,str(Path(__file__)),'crash','--size',str(size),'--source',source,'--target',target,'--stage',stage,'--path',str(path)],stdout=subprocess.DEVNULL)
                assert r.returncode==73,r.returncode
                queries=check(path,m,j,chunk,source if STAGES.index(stage)<5 else target)
                rows.append(dict(size=size,source=source,target=target,stage=stage,mode='abrupt_exit',correct_queries=queries,passed=True))
    # Checksum failure after publication must restore the retained original.
    for size in (8192,16384):
        chunk=bench.group_chunks(m,size,0)[0]
        for target in codecs:
            source='whole_lz4' if target!='whole_lz4' else 'block_lz4_1024'
            if path.exists():remove_store(path)
            raw,store=initialize(path,m,j,chunk,source)
            def corrupt(where,context):
                if where=='after_commit':
                    with open(context['final_data'],'r+b') as f:
                        f.truncate(1);f.flush();os.fsync(f.fileno())
                    raise RuntimeError('Injected corruption')
            try:EfficientMigration(store,m.b1).migrate_profiled(chunk.chunk_id,target,corrupt)
            except RuntimeError as e:assert 'Injected' in str(e)
            store.close();queries=check(path,m,j,chunk,source)
            rows.append(dict(size=size,source=source,target=target,stage='after_commit',mode='truncate_active',correct_queries=queries,passed=True))
    df=pd.DataFrame(rows);df.to_csv(OUT/'fault_trials.csv',index=False)
    result=dict(trials=len(df),passed=int(df.passed.sum()),by_mode=df.groupby('mode').size().to_dict(),correct_spot_queries=int(df.correct_queries.sum()),source_sha256=digest(HERE/'efficient_executor.py'))
    bench.write(OUT/'fault_summary.json',result);print(json.dumps(result,indent=2),flush=True)
    if path.exists():remove_store(path)
def paired():
    OUT.mkdir(exist_ok=True);p,m,j,_,_,_=bench.context();codecs=tuple(sorted(m.base.representation_registry(3)))
    chunks=[c for size in (8192,16384) for c in bench.group_chunks(m,size,0)[:2]]
    bench.write(OUT/'paired_protocol.json',dict(chunks=[c.chunk_id for c in chunks],directions=20,repeats=3,
        order='alternating original-first and efficient-first, independent stores',
        original_transactions=6,efficient_transactions=3,processor_affinity=[0],
        source_sha256=digest(HERE/'efficient_executor.py')))
    rows=[]
    for c in chunks:
        stores={};executors={}
        for kind in ('original','efficient'):
            raw,store=initialize(OUT/f'{kind}_store_{c.chunk_size}',m,j,c,codecs[0]);stores[kind]=store
            executors[kind]=EfficientMigration(store,m.b1) if kind=='efficient' else j.CachedInstrumentedMigration(p.InstrumentedSingleActiveChunkStore,store,m.b1)
        for pair,(source,target) in enumerate(itertools.permutations(codecs,2)):
            for repeat in range(3):
                for kind,store in stores.items():
                    if store.raw._row(c.chunk_id)['active_codec']!=source:executors[kind].migrate_profiled(c.chunk_id,source)
                order=('original','efficient') if (pair+repeat)%2==0 else ('efficient','original');row=dict(chunk_id=c.chunk_id,size=c.chunk_size,source=source,target=target,repeat=repeat,first=order[0])
                for kind in order:
                    # Count actual COMMIT statements, independently of the returned metadata.
                    commits=[];stores[kind].raw.connection.set_trace_callback(lambda sql:commits.append(sql) if sql.strip().upper()=='COMMIT' else None)
                    result=executors[kind].migrate_profiled(c.chunk_id,target)
                    stores[kind].raw.connection.set_trace_callback(None)
                    row[kind+'_ns']=result['migration_time_ns'];row[kind+'_commits']=len(commits);row[kind+'_bytes']=result['rewrite_bytes']
                    assert len(commits)==(6 if kind=='original' else 3)
                    assert np.array_equal(stores[kind].read_all(c.chunk_id),c.values)
                assert row['original_bytes']==row['efficient_bytes'];rows.append(row)
        for store in stores.values():assert store.raw.integrity_check()['pass'];store.close()
        print('PAIRED',c.chunk_id,len(rows),flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'paired_migrations.csv',index=False)
    pairs=df.groupby(['chunk_id','source','target'])[['original_ns','efficient_ns']].median();ratios=pairs.efficient_ns/pairs.original_ns
    summary=dict(observations=len(df),directed_pairs=len(pairs),original_median_ms=float(pairs.original_ns.median()/1e6),efficient_median_ms=float(pairs.efficient_ns.median()/1e6),median_paired_ratio=float(ratios.median()),ratio_p10=float(ratios.quantile(.1)),ratio_p90=float(ratios.quantile(.9)),all_correct=True)
    bench.write(OUT/'paired_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    psutil.Process().cpu_affinity([0]);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['faults','paired','crash']);ap.add_argument('--size',type=int);ap.add_argument('--source');ap.add_argument('--target');ap.add_argument('--stage');ap.add_argument('--path',type=Path);a=ap.parse_args()
    if a.mode=='crash':crash_worker(a.size,a.source,a.target,a.stage,a.path)
    else:globals()[a.mode]()
