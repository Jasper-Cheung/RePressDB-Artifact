"""Exact decision equivalence and alternating-order control timings."""
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fgcs_benchmark as original
import numpy as np
import pandas as pd
here=Path(__file__).resolve().parent;previous=here.parent/'method_v21'
parent=original.load(previous/'benchmark_v2.py','equivalence_input_v21')
old=original.load(previous/'risk_controller.py','risk_original_v21')
new=original.load(here/'risk_controller.py','risk_cached_v22')

def run():
    assert json.loads((previous/'results/verification.json').read_text())['passed']
    df=pd.read_csv(previous/'results/summary.csv');records=[];events=0;compared=0;setups=[]
    p,m,j=parent.context()
    for _,s in df[df.policy=='risk'].iterrows():
        size=int(s['size']);group=int(s.group);fi=original.FAMILIES.index(s.family);cs=parent.chunks(m,size,group)
        codecs,classes,pools,profile,samples,migration,sizes,iw,keys=parent.inputs(m,cs)
        path=previous/'results/cells'/f'{size}_{group}_{fi}_risk'
        stored=json.loads((path/'summary.json').read_text());initial=[codecs.index(r) for r in stored['initial_codecs']]
        oldc=[];newc=[]
        for ci,a in enumerate(initial):
            t=time.perf_counter_ns();oldc.append(old.RiskController(a,profile[ci],samples[ci],migration[ci],keys[ci],size));before=time.perf_counter_ns()-t
            t=time.perf_counter_ns();newc.append(new.RiskController(a,profile[ci],samples[ci],migration[ci],keys[ci],size));after=time.perf_counter_ns()-t
            setups.append(dict(size=size,group=group,family=s.family,chunk_id=cs[ci].chunk_id,original_ns=before,cached_ns=after,lookup_bytes=int(newc[-1].reference_cdf.nbytes+newc[-1].exceedance.nbytes)))
        seed=int(s.seed);rng=np.random.default_rng(seed);prob=p.expand_categories_to_query_classes(p.seeded_category_probabilities(s.family,160000,seed),classes)
        weights=np.arange(1,9,dtype=float)**-.8;weights/=weights.sum();positions=rng.choice(8,160000,p=weights);qpos=(rng.random(160000)[:,None]>np.cumsum(prob,axis=1)).sum(axis=1).clip(0,len(classes)-1);choices=rng.integers(0,2**16,160000)
        actions=pd.read_csv(path/'migrations.csv') if (path/'migrations.csv').exists() else pd.DataFrame(columns=['event','target'])
        actions={int(r.event):codecs.index(r.target) for r in actions.itertuples(index=False)}
        for i in range(160000):
            ci=int(positions[i]);c=cs[ci];pool=pools[c.chunk_id][classes[int(qpos[i])]];query=pool[int(choices[i])%len(pool)]
            a,b=oldc[ci],newc[ci];boundary=(a.count+1)%256==0
            if boundary:
                times={};values={};order=[('original',a),('cached',b)] if compared%2==0 else [('cached',b),('original',a)]
                for name,ctl in order:
                    t=time.perf_counter_ns();values[name]=ctl.observe(query);times[name]=time.perf_counter_ns()-t
                assert values['original']==values['cached'];assert np.array_equal(a.credit,b.credit)
                if a.count>=768:assert a.decisions[-1]==b.decisions[-1],(size,group,fi,ci,a.count)
                records.append(dict(size=size,group=group,family=s.family,chunk_id=c.chunk_id,local_count=a.count,**times));compared+=1
            else:assert a.observe(query)==b.observe(query)
            if i in actions:
                assert a.decisions[-1]['proposal']==actions[i]
                a.committed_switch(actions[i]);b.committed_switch(actions[i])
            events+=1
        saved=json.loads((path/'decisions.json').read_text())
        for c,ctl in zip(cs,newc):assert ctl.decisions==saved[c.chunk_id]
        print('EQUIVALENT',size,group,fi,events,flush=True)
    timing=pd.DataFrame(records);timing.to_csv(here/'decision_microtimings.csv',index=False);pd.DataFrame(setups).to_csv(here/'lookup_setup.csv',index=False)
    result=dict(passed=True,events=events,decision_boundaries=compared,
        original_decision_median_ns=float(timing.original.median()),cached_decision_median_ns=float(timing.cached.median()),median_paired_time_ratio=float((timing.cached/timing.original).median()),
        original_source_sha256=original.sha(previous/'risk_controller.py'),cached_source_sha256=original.sha(here/'risk_controller.py'),
        note='alternating component timings; exact request, decision, credit, and recorded-action equivalence; lookup initialization recorded separately')
    original.write(here/'implementation_equivalence.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([0]);run()
