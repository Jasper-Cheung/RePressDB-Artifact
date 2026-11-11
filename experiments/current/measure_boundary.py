"""Paired cost-boundary diagnostic. Run after the policy campaign finishes."""
from pathlib import Path
import hashlib,itertools,json,time
import numpy as np
import pandas as pd
import psutil
from fgcs_benchmark import context,group_chunks,OUT,write

def run():
    psutil.Process().cpu_affinity([0]);root=OUT/'cost_boundary';root.mkdir(exist_ok=True)
    assert len(list((OUT/'cells').glob('*/summary.json')))==90, 'Do not overlap diagnostics with policy execution.'
    p,m,j,_,_,_=context();codecs=tuple(sorted(m.base.representation_registry(3)));reps=m.base.representation_registry(3)
    selected=[c for size in [8192,16384] for c in group_chunks(m,size,0)[:2]]
    write(root/'protocol.json',{'selection':'first two manifest-ordered chunks of building group 0 at each size; no performance-based selection','chunk_ids':[c.chunk_id for c in selected],'pairs':20,'repeats':3,'order':'alternating kernel-first and transaction-first','processor_affinity':[0],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'primary_kernel':'unchanged e1.perform_actual_migration; validates outside its measured interval'})
    records=[];raw=m.b1.SingleActiveChunkStore(root/'store',m.base,recover_on_open=False);store=j.ActiveViewCachedStore(raw,m.base);execute=j.CachedInstrumentedMigration(p.InstrumentedSingleActiveChunkStore,store,m.b1)
    for c in selected:
        raw.create_chunk(c.chunk_id,c.series_index,c.values,codecs[0]);candidates={}
        for r in codecs:
            stem=root/'candidates'/c.chunk_id/r;stem.parent.mkdir(parents=True,exist_ok=True);candidates[r]=reps[r].encode_to_files(c.values,stem)
        for pi,(source,target) in enumerate(itertools.permutations(codecs,2)):
            for rep in range(3):
                if raw._row(c.chunk_id)['active_codec']!=source:execute.migrate_profiled(c.chunk_id,source)
                order=['kernel','transaction'] if (pi+rep)%2==0 else ['transaction','kernel'];values={}
                for kind in order:
                    if kind=='kernel':values['kernel_ns'],values['kernel_bytes']=m.e1.perform_actual_migration(m.base,reps,candidates,c,source,target,root/'kernel_trials',f'{c.chunk_id}_{pi}_{rep}')
                    else:
                        r=execute.migrate_profiled(c.chunk_id,target);values['transaction_ns']=r['migration_time_ns'];values['transaction_bytes']=r['rewrite_bytes']
                records.append({'chunk_id':c.chunk_id,'size':c.chunk_size,'source':source,'target':target,'repeat':rep,'first':order[0],**values})
        assert np.array_equal(store.read_all(c.chunk_id),c.values)
    assert raw.integrity_check()['pass'];store.close();df=pd.DataFrame(records);df.to_csv(root/'paired_costs.csv',index=False)
    paired=df.groupby(['chunk_id','source','target'])[['kernel_ns','transaction_ns']].median();ratio=paired.transaction_ns/paired.kernel_ns
    summary={'pairs':len(paired),'observations':len(df),'kernel_median_ms':float(paired.kernel_ns.median()/1e6),'transaction_median_ms':float(paired.transaction_ns.median()/1e6),'median_paired_multiplier':float(ratio.median()),'multiplier_p10':float(ratio.quantile(.1)),'multiplier_p90':float(ratio.quantile(.9)),'all_correct':True}
    write(root/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
