"""Fresh, balanced comparison of the revised method on a common executor.

Keep the earlier FGCS-1.2 campaign immutable. Every policy here receives the
same operation/length profiles, six layouts, and three-transaction executor.
"""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,itertools,json,platform,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fgcs_benchmark as bench
import numpy as np
import pandas as pd
import psutil
from efficient_executor import EfficientMigration
from risk_controller import RiskController,feature_key

HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
POLICIES=['static','shape','horizon','work_function','credit','risk']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sources():
    return {p.name:sha(p) for p in [Path(__file__),HERE/'efficient_executor.py',HERE/'risk_controller.py',HERE.parent/'fgcs_benchmark.py']}
def context():
    p,m,j,*_=bench.context();original=m.base.representation_registry
    def registry(*args,**kwargs):
        result=original(*args,**kwargs);result['block_lz4_4096']=m.base.BlockLZ4(4096)
        return result
    m.base.representation_registry=registry;m.b1.CODECS=tuple(sorted(registry(3)))
    return p,m,j
def chunks(m,size,group):return bench.group_chunks(m,size,group)
def check_freeze():assert json.loads((OUT/'protocol.json').read_text())['sources']==sources(),'Frozen source changed'
def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Preserve an existing protocol'
    primary=json.loads((bench.OUT/'protocol.json').read_text())
    protocol=dict(version='FGCS-2.0',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),sources=sources(),
        groups=primary['groups'],sizes=[8192,16384],families=bench.FAMILIES,seeds=[2026110101,2026110102,2026110103],
        events=160000,policies=POLICIES,layouts=['whole_lz4','whole_zstd','block_lz4_64','block_lz4_256','block_lz4_1024','block_lz4_4096'],
        query_samples_per_stratum=64,query_warmups=4,migration_repeats=3,
        profile='query operation and length bins; all policies receive identical refined mean costs and directed durable costs',
        tail_reference='initial layout under the current short and long feature mixtures',tail_factor=1.10,tail_exceedance=.01,
        block_size=256,windows=[768,256],confirmations=3,risk_repair_confirmations=2,minimum_gain=.08,credit_factor=1.25,
        budget='max target payload per 1000 global events; all policies',
        execution='one worker pinned to logical processor 0; two cyclic orders across 18 workload instances',
        data_role='same 24 previously studied buildings; new query realizations; no untouched-corpus designation',
        design_role='method designed after the FGCS-1.2 results; parameters frozen before this comparison')
    bench.write(OUT/'protocol.json',protocol)
    bench.write(OUT/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,affinity=psutil.Process().cpu_affinity()))
    print('FROZEN',json.dumps(sources()),flush=True)
def make_pools(m,cs):return m.a2.build_query_pools(m.base,m.e1,m.core,cs)
def strata(pools,n):
    groups=defaultdict(list)
    for queries in pools.values():
        for query in queries:groups[feature_key(query,n)].append(query)
    return dict(sorted(groups.items()))
def calibrate():
    check_freeze();p,m,j=context();codecs=tuple(sorted(m.base.representation_registry(3)))
    allrows=[];migration=[];payload=[];wall=0
    for size,group in itertools.product([8192,16384],range(3)):
        marker=OUT/f'calibration_{size}_{group}.json'
        if marker.exists():
            data=json.loads(marker.read_text());allrows+=data['query'];migration+=data['migration'];payload+=data['payload'];wall+=data['elapsed_ns'];continue
        start=time.perf_counter_ns();cs=chunks(m,size,group);pools=make_pools(m,cs)
        raw=m.b1.SingleActiveChunkStore(OUT/f'calibration_store_{size}_{group}',m.base,recover_on_open=False)
        store=j.ActiveViewCachedStore(raw,m.base);execute=EfficientMigration(store,m.b1)
        local=[];lm=[];lp=[]
        for c in cs:raw.create_chunk(c.chunk_id,c.series_index,c.values,codec=codecs[0])
        for ci,c in enumerate(cs):
            features=strata(pools[c.chunk_id],size)
            # Rotate layout order and each complete stratum's query order.
            order=codecs[ci%len(codecs):]+codecs[:ci%len(codecs)]
            for codec in order:
                if raw._row(c.chunk_id)['active_codec']!=codec:execute.migrate_profiled(c.chunk_id,codec)
                lp.append(dict(chunk_id=c.chunk_id,codec=codec,bytes=store.active_stored(c.chunk_id).stored_size_bytes))
                rng=np.random.default_rng(2026110100+ci+size)
                for fi,(key,queries) in enumerate(features.items()):
                    indices=[]
                    while len(indices)<68:indices.extend(rng.permutation(len(queries)).tolist())
                    times=[]
                    for sample,k in enumerate(indices[:68]):
                        query=queries[k];t=time.perf_counter_ns();actual,*_=store.execute_query(c.chunk_id,query);elapsed=time.perf_counter_ns()-t
                        assert m.base.results_equal(actual,m.base.execute_on_array(c.values,query))
                        if sample>=4:times.append(elapsed)
                    local.append(dict(chunk_id=c.chunk_id,codec=codec,operation=key[0],length_bin=key[1],feature=fi,pool_count=len(queries),mean_ns=float(np.mean(times)),median_ns=float(np.median(times)),p95_ns=float(np.percentile(times,95)),samples=times))
            for source,target in itertools.permutations(codecs,2):
                for repeat in range(3):
                    if raw._row(c.chunk_id)['active_codec']!=source:execute.migrate_profiled(c.chunk_id,source)
                    result=execute.migrate_profiled(c.chunk_id,target)
                    lm.append(dict(chunk_id=c.chunk_id,source=source,target=target,repeat=repeat,ns=result['migration_time_ns'],bytes=result['rewrite_bytes'],**result['stage_times_ns']))
            assert np.array_equal(store.read_all(c.chunk_id),c.values)
        assert raw.integrity_check()['pass'];store.close();elapsed=time.perf_counter_ns()-start
        data=dict(query=local,migration=lm,payload=lp,elapsed_ns=elapsed);bench.write(marker,data)
        allrows+=local;migration+=lm;payload+=lp;wall+=elapsed
        print('CALIBRATED',size,group,len(local),flush=True)
    bench.write(OUT/'query_calibration.json',allrows)
    pd.DataFrame([{k:v for k,v in r.items() if k!='samples'} for r in allrows]).to_csv(OUT/'query_calibration.csv',index=False)
    pd.DataFrame(migration).to_csv(OUT/'durable_calibration.csv',index=False);pd.DataFrame(payload).to_csv(OUT/'codec_sizes.csv',index=False)
    bench.write(OUT/'calibration_summary.json',dict(query_records=len(allrows),query_events=len(allrows)*68,durable_observations=len(migration),wall_ns=wall,
        hashes={p.name:sha(p) for p in [OUT/'query_calibration.json',OUT/'durable_calibration.csv',OUT/'codec_sizes.csv']}))
    print('CALIBRATION COMPLETE',len(allrows),len(migration),flush=True)
def inputs(m,cs):
    codecs=tuple(sorted(m.base.representation_registry(3)));classes=tuple(m.core.QUERY_CLASSES)
    data=json.loads((OUT/'query_calibration.json').read_text());mapping={(r['chunk_id'],r['codec'],r['operation'],r['length_bin']):r for r in data}
    migration=pd.read_csv(OUT/'durable_calibration.csv').groupby(['chunk_id','source','target']).ns.median()
    sizes=pd.read_csv(OUT/'codec_sizes.csv').set_index(['chunk_id','codec']).bytes
    pools=make_pools(m,cs);profiles=[];samples=[];costs=[];weights=[];keys=[]
    p=bench.load(bench.SRC/'RUN_THIS_ONLY_REPRESSDB_JSS_MAJOR_REVISION_EXPERIMENTS_V1_5_1.py','initial_weight_parent_v2')
    iw=p.initial_weights_from_base(m,classes)
    for c in cs:
        features=strata(pools[c.chunk_id],c.chunk_size);fk=list(features);keys.append(fk)
        profiles.append([[mapping[(c.chunk_id,r,*key)]['mean_ns'] for key in fk] for r in codecs])
        samples.append([[mapping[(c.chunk_id,r,*key)]['samples'] for key in fk] for r in codecs])
        costs.append([[0 if a==b else migration.loc[c.chunk_id,a,b] for b in codecs] for a in codecs])
        fw=np.zeros(len(fk));ki={key:i for i,key in enumerate(fk)}
        for qc,weight in zip(classes,iw):
            for query in pools[c.chunk_id][qc]:fw[ki[feature_key(query,c.chunk_size)]]+=weight/len(pools[c.chunk_id][qc])
        assert np.isclose(fw.sum(),1);weights.append(fw)
    return codecs,classes,pools,np.asarray(profiles),np.asarray(samples),np.asarray(costs),sizes,weights,keys
class WorkFunction(bench.WorkFunctionController):
    def __init__(self,active,ncodecs,nfeatures):
        super().__init__(active,ncodecs);self.hist=np.zeros(nfeatures)
def cell(size,group,family,policy):
    check_freeze();destination=OUT/'cells'/f'{size}_{group}_{family}_{policy}'
    if (destination/'summary.json').exists():return
    destination.mkdir(parents=True,exist_ok=True)
    p,m,j=context();cs=chunks(m,size,group)
    codecs,classes,pools,profile,samples,migration,sizes,iw,keys=inputs(m,cs)
    protocol=json.loads((OUT/'protocol.json').read_text());n=protocol['events'];seed=protocol['seeds'][group]+100*family+size
    rng=np.random.default_rng(seed);prob=p.expand_categories_to_query_classes(p.seeded_category_probabilities(bench.FAMILIES[family],n,seed),classes)
    route_weights=np.arange(1,9,dtype=float)**-.8;route_weights/=route_weights.sum()
    positions=rng.choice(8,n,p=route_weights);qpos=(rng.random(n)[:,None]>np.cumsum(prob,axis=1)).sum(axis=1).clip(0,len(classes)-1);choices=rng.integers(0,2**16,n)
    initial=[int(np.argmin(x@w)) for x,w in zip(profile,iw)]
    root=destination/'store'
    if root.exists():
        import shutil
        assert OUT.resolve() in root.resolve().parents
        shutil.rmtree(root)
    raw=m.b1.SingleActiveChunkStore(root,m.base,recover_on_open=False)
    for c,r in zip(cs,initial):raw.create_chunk(c.chunk_id,c.series_index,c.values,codec=codecs[r])
    store=j.ActiveViewCachedStore(raw,m.base);execute=EfficientMigration(store,m.b1)
    cfg=json.loads((bench.SRC/'results_s1_shape_ewma_be_v130/shape_ewma_be_frozen_config.json').read_text());controllers=[]
    for ci,a in enumerate(initial):
        ctl=None if policy=='static' else RiskController(a,profile[ci],samples[ci],migration[ci],keys[ci],size) if policy=='risk' else bench.CreditController(a,len(codecs)) if policy=='credit' else WorkFunction(a,len(codecs),len(keys[ci])) if policy=='work_function' else p.ShapeEWMAController(a,iw[ci],cfg) if policy=='shape' else p.RepressController(a,p.RepressVariant(name='full_repressdb'))
        controllers.append(ctl)
    for c in cs:store.execute_query(c.chunk_id,pools[c.chunk_id][classes[0]][0])
    active=initial.copy();keymaps=[{k:i for i,k in enumerate(ks)} for ks in keys]
    query_ns=np.zeros(n,dtype=np.int64);control_ns=np.zeros(n,dtype=np.int64);action_ns=np.zeros(n,dtype=np.int64)
    feature_ids=np.zeros(n,dtype=np.int8);logs=[];truth_ns=0;predicted=0.;interval=-1;used=0;denied=0
    limit=max(int(sizes.loc[(c.chunk_id,r)]) for c in cs for r in codecs);start=time.perf_counter_ns()
    for i in range(n):
        ci=int(positions[i]);c=cs[ci];pool=pools[c.chunk_id][classes[int(qpos[i])]];query=pool[int(choices[i])%len(pool)]
        t=time.perf_counter_ns();actual,*_=store.execute_query(c.chunk_id,query);query_ns[i]=time.perf_counter_ns()-t
        t=time.perf_counter_ns();assert m.base.results_equal(actual,m.base.execute_on_array(c.values,query));truth_ns+=time.perf_counter_ns()-t
        feature=keymaps[ci][feature_key(query,size)];feature_ids[i]=feature;predicted+=profile[ci,active[ci],feature]
        ctl=controllers[ci];proposal=None
        if ctl is not None:
            t=time.perf_counter_ns()
            if policy=='risk':proposal=ctl.observe(query)
            else:
                q=keymaps[ci][feature_key(query,size)]
                if policy in ('credit','work_function'):proposal=ctl.step(q,profile[ci],migration[ci])
                else:proposal,_=ctl.observe_and_decide(q,profile[ci],migration[ci],codecs,i,c.chunk_id)
            control_ns[i]=time.perf_counter_ns()-t
        if i//1000!=interval:interval=i//1000;used=0
        if proposal is not None:
            rb=int(sizes.loc[c.chunk_id,codecs[proposal]])
            if used+rb>limit:denied+=1;continue
            before=active[ci];result=execute.migrate_profiled(c.chunk_id,codecs[proposal]);used+=result['rewrite_bytes'];action_ns[i]=result['migration_time_ns']
            t=time.perf_counter_ns();ctl.committed_switch(proposal);control_ns[i]+=time.perf_counter_ns()-t;active[ci]=proposal;predicted+=migration[ci,before,proposal]
            logs.append(dict(event=i,source=codecs[before],target=codecs[proposal],reason=ctl.proposal_reason if policy=='risk' else policy,**result))
    elapsed=time.perf_counter_ns()-start;integrity=raw.integrity_check();assert integrity['pass']
    for c in cs:assert np.array_equal(store.read_all(c.chunk_id),c.values)
    store.close();lat=query_ns+control_ns+action_ns
    np.savez_compressed(destination/'latencies.npz',service_ns=lat,query_ns=query_ns,control_ns=control_ns,migration_ns=action_ns,feature_ids=feature_ids)
    if logs:pd.DataFrame(logs).to_csv(destination/'migrations.csv',index=False)
    if policy=='risk':bench.write(destination/'decisions.json',{c.chunk_id:ctl.decisions for c,ctl in zip(cs,controllers)})
    summary=dict(size=size,group=group,family=bench.FAMILIES[family],policy=policy,seed=seed,n_queries=n,service_ns=int(lat.sum()),query_ns=int(query_ns.sum()),controller_ns=int(control_ns.sum()),migration_ns=int(action_ns.sum()),p99_ns=float(np.percentile(lat,99)),p999_ns=float(np.percentile(lat,99.9)),max_ns=int(lat.max()),migration_count=len(logs),rewrite_bytes=sum(r['rewrite_bytes'] for r in logs),budget_denied=denied,budget_limit_bytes=limit,prediction_ns=predicted,elapsed_ns=elapsed,truth_ns=truth_ns,correct=True,integrity=integrity,sources=sources(),initial_codecs=[codecs[a] for a in initial])
    bench.write(destination/'summary.json',summary);print('CELL',size,group,family,policy,round(summary['service_ns']/1e9,3),'p99',summary['p99_ns'],'migrations',len(logs),flush=True)
def campaign():
    check_freeze()
    for index,(size,group,family) in enumerate(itertools.product([8192,16384],range(3),range(3))):
        policies=POLICIES[index%6:]+POLICIES[:index%6]
        for policy in policies:subprocess.run([sys.executable,str(Path(__file__)),'cell','--size',str(size),'--group',str(group),'--family',str(family),'--policy',policy],check=True)
def aggregate():
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'cells').glob('*/summary.json'))]
    assert len(rows)==108;df=pd.DataFrame(rows);df.to_csv(OUT/'summary.csv',index=False)
    rng=np.random.default_rng(20261101);stats=[]
    for metric in ['service_ns','p99_ns','p999_ns']:
        pivot=df.pivot(index=['size','group','family'],columns='policy',values=metric)
        for den in [p for p in POLICIES if p!='risk']:
            ratio=pivot.risk/pivot[den];log=np.log(ratio.to_numpy());boot=np.exp(log[rng.integers(0,18,(20000,18))].mean(axis=1))
            stats.append(dict(metric=metric,denominator=den,gm_ratio=float(np.exp(log.mean())),ci_low=float(np.percentile(boot,2.5)),ci_high=float(np.percentile(boot,97.5)),wins=int((ratio<1).sum())))
    pd.DataFrame(stats).to_csv(OUT/'paired_statistics.csv',index=False);print(pd.DataFrame(stats).to_string(index=False))
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');psutil.Process().cpu_affinity([0]);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','calibrate','cell','campaign','aggregate']);ap.add_argument('--size',type=int,default=8192);ap.add_argument('--group',type=int,default=0);ap.add_argument('--family',type=int,default=0);ap.add_argument('--policy',default='risk',choices=POLICIES);a=ap.parse_args()
    if a.mode=='cell':cell(a.size,a.group,a.family,a.policy)
    else:globals()[a.mode]()
