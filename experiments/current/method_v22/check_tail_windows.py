"""Descriptive validation of empirical tail checks on stable-layout windows.

Use the same requests from separately executed Static and Risk-RePress cells.
Overlapping windows and separate timings make these agreement counts descriptive.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import benchmark_v2 as b

def run():
    assert json.loads((b.OUT/'verification.json').read_text())['passed']
    summaries=pd.read_csv(b.OUT/'summary.csv');rows=[]
    for _,s in summaries[summaries.policy=='risk'].iterrows():
        fi=b.FAMILIES.index(s.family);folder=b.OUT/'cells'/f'{s["size"]}_{s.group}_{fi}_risk'
        risk=np.load(folder/'latencies.npz')['query_ns'];static=np.load(b.OUT/'cells'/f'{s["size"]}_{s.group}_{fi}_static'/'latencies.npz')['query_ns']
        weights=np.arange(1,9,dtype=float)**-.8;weights/=weights.sum()
        positions=np.random.default_rng(int(s.seed)).choice(8,160000,p=weights)
        # Chunk order comes from the preserved NPZ manifest, as in group_chunks.
        mf=pd.read_csv(b.bench.OUT/'chunk_manifest.csv');groups=json.loads((b.bench.OUT/'protocol.json').read_text())['groups'][int(s.group)]
        wanted=set(mf[(mf.chunk_size==s['size'])&mf.building_id.isin(groups)].chunk_id)
        z=np.load(b.bench.OUT/f'values_{s["size"]}.npz');ids=[str(c) for c in z['chunk_ids'] if str(c) in wanted]
        assert len(ids)==8
        decisions=json.loads((folder/'decisions.json').read_text());migrations=pd.read_csv(folder/'migrations.csv') if (folder/'migrations.csv').exists() else pd.DataFrame(columns=['chunk_id','event'])
        # Initial labels match the six-codec sorted registry used in execution.
        codecs=sorted(['whole_lz4','whole_zstd','block_lz4_64','block_lz4_256','block_lz4_1024','block_lz4_4096'])
        initial=json.loads((folder/'summary.json').read_text())['initial_codecs']
        for ci,cid in enumerate(ids):
            route=np.flatnonzero(positions==ci);rq=risk[route];sq=static[route]
            switches=np.searchsorted(route,migrations[migrations.chunk_id==cid].event.to_numpy(int))+1
            for d in decisions[cid]:
                n=d['local_count'];past=switches[switches<=n];last=int(past[-1]) if len(past) else 0
                if n-last<768:continue
                ratios=[]
                for width in (256,768):ratios.append(float(np.percentile(rq[n-width:n],99)/np.percentile(sq[n-width:n],99)))
                rows.append(dict(size=int(s['size']),group=int(s.group),family=s.family,chunk_id=cid,local_count=n,
                    layout=codecs[d['active']],role='reference' if codecs[d['active']]==initial[ci] else 'adapted',
                    model_feasible=bool(d['feasible'][d['active']]),max_window_p99_ratio=max(ratios),observed_within_tolerance=max(ratios)<=1.10))
    df=pd.DataFrame(rows);df.to_csv(b.OUT/'stable_tail_windows.csv',index=False)
    summary={}
    for role,part in df.groupby('role'):
        predicted=part[part.model_feasible]
        summary[role]=dict(windows=len(part),model_feasible_windows=len(predicted),observed_within_tolerance=int(part.observed_within_tolerance.sum()),
                           model_feasible_and_within_tolerance=int(predicted.observed_within_tolerance.sum()),
                           median_p99_ratio=float(part.max_window_p99_ratio.median()))
    output=dict(scope='same-request separately timed query-path windows; stable layout for 768 local requests; overlapping descriptive observations',roles=summary)
    b.bench.write(b.OUT/'tail_window_validation.json',output);print(json.dumps(output,indent=2))
if __name__=='__main__':run()
