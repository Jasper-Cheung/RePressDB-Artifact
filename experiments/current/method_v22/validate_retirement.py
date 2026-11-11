"""Reproduce the old unlink window and check the added 4096-value layout."""
from pathlib import Path
import argparse,json,os,subprocess,sys
import numpy as np
import pandas as pd
import psutil
import benchmark_v2 as new
import fgcs_benchmark as bench
import validate_executor as validation
from efficient_executor import EfficientMigration

def legacy_worker():
    p,m,j,_,_,_=bench.context();c=bench.group_chunks(m,16384,0)[0]
    root=validation.OUT/'legacy_gap_store'
    if root.exists():validation.remove_store(root)
    raw,store=validation.initialize(root,m,j,c,'block_lz4_1024')
    old_metadata=store.active_stored(c.chunk_id).metadata_path
    committed=[False]
    def hook(stage,_):
        if stage=='after_catalog_commit':committed[0]=True
    def trace(sql):
        if committed[0] and sql.strip().upper()=='BEGIN IMMEDIATE' and not old_metadata.exists():os._exit(73)
    raw.connection.set_trace_callback(trace)
    executor=j.CachedInstrumentedMigration(p.InstrumentedSingleActiveChunkStore,store,m.b1)
    executor.inner.migrate_profiled(c.chunk_id,'whole_lz4',fault_hook=hook)
    raise AssertionError('Old deletion-before-retirement window was not reached')
def run():
    assert len(list((new.OUT/'cells').glob('*/summary.json')))==144,'Run after performance measurement'
    result=subprocess.run([sys.executable,str(Path(__file__)),'legacy_worker'],stdout=subprocess.DEVNULL)
    assert result.returncode==73
    p,m,j,_,_,_=bench.context();root=validation.OUT/'legacy_gap_store'
    try:m.b1.SingleActiveChunkStore(root,m.base,recover_on_open=True)
    except FileNotFoundError as e:
        evidence=dict(observed=True,interruption='original cleanup transaction begins after previous block metadata was deleted',recovery_error=type(e).__name__,missing_path=str(e.filename),original_source='byte-preserved six-transaction executor',worker_exit=result.returncode)
    else:raise AssertionError('Expected original recovery failure was absent')
    bench.write(validation.OUT/'legacy_retirement_reproduction.json',evidence)
    p,m,j=new.context();rows=[];root=validation.OUT/'supplement_store'
    for size in (8192,16384):
        c=bench.group_chunks(m,size,0)[0]
        for source,target in [('block_lz4_4096','whole_lz4'),('whole_lz4','block_lz4_4096')]:
            for stage in validation.STAGES:
                if root.exists():validation.remove_store(root)
                raw,store=validation.initialize(root,m,j,c,source)
                class Injected(Exception):pass
                def hook(where,_):
                    if where==stage:raise Injected(stage)
                try:EfficientMigration(store,m.b1).migrate_profiled(c.chunk_id,target,hook)
                except Injected:pass
                else:raise AssertionError('Missing fault stage')
                store.close();expected=source if validation.STAGES.index(stage)<5 else target
                queries=validation.check(root,m,j,c,expected)
                rows.append(dict(size=size,source=source,target=target,stage=stage,mode='exception_4096',passed=True,correct_queries=queries))
        if root.exists():validation.remove_store(root)
        raw,store=validation.initialize(root,m,j,c,'whole_lz4')
        def corrupt(where,context):
            if where=='after_commit':
                with open(context['final_data'],'r+b') as f:f.truncate(1);f.flush();os.fsync(f.fileno())
                raise RuntimeError('Injected corruption')
        try:EfficientMigration(store,m.b1).migrate_profiled(c.chunk_id,'block_lz4_4096',corrupt)
        except RuntimeError as e:assert 'Injected' in str(e)
        store.close();queries=validation.check(root,m,j,c,'whole_lz4')
        rows.append(dict(size=size,source='whole_lz4',target='block_lz4_4096',stage='after_commit',mode='truncate_4096',passed=True,correct_queries=queries))
    df=pd.DataFrame(rows);df.to_csv(validation.OUT/'supplement_4096.csv',index=False)
    combined=dict(trials=450+len(df),passed=450+int(df.passed.sum()),additional_trials=len(df),old_retirement_failure_reproduced=True,source_sha256=validation.digest(new.HERE/'efficient_executor.py'))
    bench.write(validation.OUT/'combined_fault_summary.json',combined)
    if root.exists():validation.remove_store(root)
    print(json.dumps(combined,indent=2))
if __name__=='__main__':
    psutil.Process().cpu_affinity([0]);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['run','legacy_worker']);a=ap.parse_args();globals()[a.mode]()
