"""Reproducible FGCS revision campaign. Original sources are unchanged in source/."""
from pathlib import Path
from collections import deque
import argparse, dataclasses, hashlib, importlib.util, itertools, json, os, platform, shutil, subprocess, sys, time
import numpy as np
import pandas as pd
import psutil

HERE=Path(__file__).resolve().parent
SRC=HERE/'source'
OUT=HERE/'results'
POLICIES=['static','shape','work_function','legacy_repress','repressdb']
FAMILIES=['R1_A_gradual_periodic','R1_B_mean_reverting','R1_C_noisy_bursts']

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def context():
    p=load(SRC/'RUN_THIS_ONLY_REPRESSDB_JSS_MAJOR_REVISION_EXPERIMENTS_V1_5_1.py','revision_parent')
    paths=p.resolve_project_paths(SRC);m=p.load_all_modules(paths,'fgcs')
    j=load(SRC/'RUN_THIS_ONLY_REPRESSDB_J1R_ACTIVE_VIEW_CACHE_CONFIRM_V1_6_2.py','active_view')
    chunks,manifest,meta=p.rebuild_r1_chunks_from_frozen_calibration(paths,m)
    return p,m,j,chunks,manifest,meta

class CreditController:
    """Causal accumulated cost advantage; no forecast horizon or future duration."""
    def __init__(self,active,ncodecs):
        self.active_position=active;self.events=deque(maxlen=300);self.count=0
        self.credit=np.zeros(ncodecs);self.last_switch=-10**9;self.confirm_target=None;self.confirm=0
    def step(self,q,profile,migration):
        self.count+=1;self.events.append(q)
        # Reflected cumulative advantage: observations favoring the active layout repay credit.
        self.credit=np.maximum(0.,self.credit+profile[self.active_position,q]-profile[:,q])
        if self.count<200 or self.count%100:return None
        a=np.asarray(self.events,dtype=int)
        lw=np.bincount(a,minlength=profile.shape[1])/len(a)
        sw=np.bincount(a[-100:],minlength=profile.shape[1])/len(a[-100:])
        lc=profile@lw;sc=profile@sw;t=int(np.argmin(lc));cur=self.active_position
        if t==cur or t!=int(np.argmin(sc)) or min((lc[cur]-lc[t])/lc[cur],(sc[cur]-sc[t])/sc[cur])<.08:
            self.confirm_target=None;self.confirm=0;return None
        self.confirm=self.confirm+1 if self.confirm_target==t else 1;self.confirm_target=t
        if self.count-self.last_switch<300 or self.confirm<3:return None
        return t if self.credit[t]>=1.25*migration[cur,t] else None
    def committed_switch(self,t):
        self.active_position=t;self.credit.fill(0.);self.last_switch=self.count;self.confirm=0;self.confirm_target=None

class WorkFunctionController:
    """Five-state work-function recurrence at common 100-query decision boundaries.

    Directed measured transitions and the shared budget invalidate metric-MTS
    guarantees; this implementation is an empirical baseline, not OReO itself.
    """
    def __init__(self,active,ncodecs):
        self.active_position=active;self.count=0;self.hist=np.zeros(8);self.work=None;self.initial=active
    def step(self,q,profile,migration):
        self.count+=1;self.hist[q]+=1
        if self.count%100:return None
        if self.work is None:self.work=migration[self.initial].copy()
        self.work=(self.work[:,None]+migration).min(axis=0)+profile@self.hist
        self.work-=self.work.min();self.hist.fill(0.)
        if self.count<200:return None
        candidate=int(np.argmin(self.work+migration[self.active_position]))
        return candidate if candidate!=self.active_position else None
    def committed_switch(self,t):self.active_position=t

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    p,m,j,chunks,manifest,meta=context()
    prior=pd.read_csv(SRC/'results_jss_major_revision_v151/_worker_assets/r2_selected_buildings.csv')
    # All eight previously executed buildings form a diagnostic set; remaining buildings are confirmation data.
    prior_ids=set(prior.head(8).building_id)
    builds=manifest[manifest.chunk_size==8192].sort_values('building_rank')
    confirm=[x for x in builds.building_id if x not in prior_ids]
    groups=[confirm[i:i+8] for i in range(0,24,8)]
    codecs=tuple(sorted(m.base.representation_registry(3)));classes=tuple(m.core.QUERY_CLASSES)
    protocol={'version':'FGCS-1.1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'policies':POLICIES,'groups':groups,'prior_diagnostic_buildings':sorted(prior_ids),'query_families':FAMILIES,
      'sizes':[8192,16384],'events_per_cell':160000,'calibration_query_samples':32,'calibration_migration_repeats':3,
      'seeds':[2026103101,2026103102,2026103103],
      'processor_affinity':[0],
      'pilot_note':'One unpinned static timing pilot and the first calibration are preserved separately in results_diagnostic_unpinned. Before comparative execution, all final calibration and worker processes are pinned to logical processor 0 to reduce scheduling variation. No policy parameter was selected using that pilot.',
      'control_settings':{'windows':[300,100],'decision_interval':100,'confirmations':3,'minimum_gain':.08,'cooldown':300,'credit_factor':1.25},
      'costs':'fresh query path including metadata dispatch; full durable migration transaction',
      'baseline_inputs':'Shape and work-function receive the same fresh query and full-transaction cost tables; legacy receives fresh query and codec-only migration costs.',
      'statistical_unit':'building-group / query-family / chunk-size; 18 independent query realizations',
      'prior_data_note':'These buildings occurred in earlier replay; they were absent from the old eight-building end-to-end campaign. This is a new seeded confirmation campaign, not an untouched corpus.',
      'code_sha256':sha(Path(__file__))}
    write(OUT/'protocol.json',protocol)
    selected=manifest[manifest.building_id.isin(confirm)].copy();selected.to_csv(OUT/'chunk_manifest.csv',index=False)
    for size in [8192,16384]:
        cs=[c for c in chunks if c.chunk_size==size and meta[c.chunk_id]['building_id'] in confirm]
        np.savez(OUT/f'values_{size}.npz',chunk_ids=np.array([c.chunk_id for c in cs]),values=np.stack([c.values for c in cs]),series=np.array([c.series_index for c in cs]))
    write(OUT/'environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'processor_affinity':psutil.Process().cpu_affinity(),'cpu_logical':psutil.cpu_count(),'cpu_physical':psutil.cpu_count(logical=False),'memory_total':psutil.virtual_memory().total})
    # Freeze before collecting performance measurements.
    write(OUT/'freeze.json',{'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(Path(__file__)),'manifest_sha256':sha(OUT/'chunk_manifest.csv')})
    print('PREPARED',len(confirm),'buildings',flush=True)

def group_chunks(m,size,group):
    proto=json.loads((OUT/'protocol.json').read_text());wanted=set(proto['groups'][group]);mf=pd.read_csv(OUT/'chunk_manifest.csv');ids=set(mf[(mf.chunk_size==size)&mf.building_id.isin(wanted)].chunk_id)
    z=np.load(OUT/f'values_{size}.npz');return [m.base.ChunkUnit(chunk_id=str(cid),series_index=int(z['series'][i]),chunk_start=0,chunk_size=size,values=z['values'][i].copy()) for i,cid in enumerate(z['chunk_ids']) if str(cid) in ids]

def calibrate():
    p,m,j,_,_,_=context();codecs=tuple(sorted(m.base.representation_registry(3)));classes=tuple(m.core.QUERY_CLASSES)
    records=[];migration_rows=[];codec_rows=[];start=time.perf_counter_ns()
    for size in [8192,16384]:
      for g in range(3):
        marker=OUT/f'calibration_{size}_{g}.json'
        if marker.exists():continue
        chunks=group_chunks(m,size,g);pools=m.a2.build_query_pools(m.base,m.e1,m.core,chunks)
        root=OUT/f'calibration_store_{size}_{g}';raw=m.b1.SingleActiveChunkStore(root,m.base,recover_on_open=False);store=j.ActiveViewCachedStore(raw,m.base)
        executor=j.CachedInstrumentedMigration(p.InstrumentedSingleActiveChunkStore,store,m.b1)
        localq=[];localm=[];localc=[]
        for c in chunks:raw.create_chunk(c.chunk_id,c.series_index,c.values,codec=codecs[0])
        for c in chunks:
          for rep in codecs:
            if raw._row(c.chunk_id)['active_codec']!=rep:executor.migrate_profiled(c.chunk_id,rep)
            stored=store.active_stored(c.chunk_id)
            localc.append({'chunk_id':c.chunk_id,'codec':rep,'bytes':stored.stored_size_bytes})
            for qc in classes:
              pool=pools[c.chunk_id][qc];times=[]
              for i in range(36):
                query=pool[i%len(pool)];s=time.perf_counter_ns();actual,_,_,_=store.execute_query(c.chunk_id,query);elapsed=time.perf_counter_ns()-s
                assert m.base.results_equal(actual,m.base.execute_on_array(c.values,query))
                if i>=4:times.append(elapsed)
              localq.append({'chunk_id':c.chunk_id,'codec':rep,'query_class':qc,'median_ns':float(np.median(times)),'p95_ns':float(np.percentile(times,95))})
          for source,target in itertools.permutations(codecs,2):
            for repeat in range(3):
              if raw._row(c.chunk_id)['active_codec']!=source:executor.migrate_profiled(c.chunk_id,source)
              r=executor.migrate_profiled(c.chunk_id,target)
              localm.append({'chunk_id':c.chunk_id,'source':source,'target':target,'repeat':repeat,'ns':r['migration_time_ns'],'bytes':r['rewrite_bytes'],**r['stage_times_ns']})
          assert np.array_equal(store.read_all(c.chunk_id),c.values)
        assert raw.integrity_check()['pass'];store.close()
        write(marker,{'query':localq,'migration':localm,'codecs':localc})
        print('CALIBRATED',size,g,flush=True)
    for path in sorted(OUT.glob('calibration_[0-9]*_*.json')):
        d=json.loads(path.read_text());records+=d['query'];migration_rows+=d['migration'];codec_rows+=d['codecs']
    pd.DataFrame(records).to_csv(OUT/'query_calibration.csv',index=False);pd.DataFrame(migration_rows).to_csv(OUT/'durable_calibration.csv',index=False);pd.DataFrame(codec_rows).to_csv(OUT/'codec_sizes.csv',index=False)
    write(OUT/'calibration_wall.json',{'elapsed_ns':time.perf_counter_ns()-start,'query_records':len(records),'migration_records':len(migration_rows),'includes_target_validation':True})

def cell(size,group,family,policy):
    name=f'{size}_{group}_{family}_{policy}';dest=OUT/'cells'/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():return
    p,m,j,_,_,_=context();chunks=group_chunks(m,size,group);codecs=tuple(sorted(m.base.representation_registry(3)));classes=tuple(m.core.QUERY_CLASSES)
    proto=json.loads((OUT/'protocol.json').read_text());n=proto['events_per_cell'];seed=proto['seeds'][group]+family*100+size
    rng=np.random.default_rng(seed)
    prob=p.expand_categories_to_query_classes(p.seeded_category_probabilities(FAMILIES[family],n,seed),classes)
    # Continuous query mixtures are unchanged; a fresh seeded Zipf-like chunk access process supplies event routing.
    weights=np.arange(1,9,dtype=float)**-.8;weights/=weights.sum();positions=rng.choice(8,n,p=weights)
    qpos=(rng.random(n)[:,None]>np.cumsum(prob,axis=1)).sum(axis=1).clip(0,len(classes)-1)
    choices=rng.integers(0,2**16,n);pools=m.a2.build_query_pools(m.base,m.e1,m.core,chunks)
    qframe=pd.read_csv(OUT/'query_calibration.csv');mc=pd.read_csv(OUT/'durable_calibration.csv').groupby(['chunk_id','source','target']).ns.median();sizes=pd.read_csv(OUT/'codec_sizes.csv').set_index(['chunk_id','codec']).bytes
    old=pd.read_csv(SRC/'results_jss_major_revision_v151/j4_calibration/fresh_full_calibration/measured_migration_costs.csv').set_index(['chunk_id','current_codec','target_codec']).median_migration_ns
    prof=np.array([[[qframe[(qframe.chunk_id==c.chunk_id)&(qframe.codec==r)&(qframe.query_class==q)].median_ns.iloc[0] for q in classes] for r in codecs] for c in chunks])
    durable=np.array([[[0 if a==b else mc.loc[(c.chunk_id,a,b)] for b in codecs] for a in codecs] for c in chunks])
    codec_cost=np.array([[[float(old.loc[(c.chunk_id,a,b)]) for b in codecs] for a in codecs] for c in chunks])
    # Fixed initial scan-heavy choice is calibrated without using the future mixture.
    iw=p.initial_weights_from_base(m,classes);initial=[int(np.argmin(x@iw)) for x in prof]
    root=dest/'store'
    if root.exists():
        assert HERE.resolve() in root.resolve().parents
        shutil.rmtree(root)
    raw=m.b1.SingleActiveChunkStore(root,m.base,recover_on_open=False)
    for c,r in zip(chunks,initial):raw.create_chunk(c.chunk_id,c.series_index,c.values,codecs[r])
    store=j.ActiveViewCachedStore(raw,m.base);executor=j.CachedInstrumentedMigration(p.InstrumentedSingleActiveChunkStore,store,m.b1)
    cfg=json.loads((SRC/'results_s1_shape_ewma_be_v130/shape_ewma_be_frozen_config.json').read_text())
    controllers=[None if policy=='static' else p.ShapeEWMAController(r,iw,cfg) if policy=='shape' else WorkFunctionController(r,len(codecs)) if policy=='work_function' else p.RepressController(r,p.RepressVariant(name='full_repressdb')) if policy=='legacy_repress' else CreditController(r,len(codecs)) for r in initial]
    for c in chunks:store.execute_query(c.chunk_id,pools[c.chunk_id][classes[0]][0])
    lat=np.zeros(n,dtype=np.int64);controller_ns=0;migration_ns=0;truth_ns=0;mig=[];active=initial.copy();pred_query=0.;pred_mig=0.;interval=-1;used=0;denied=0
    limit=max(int(sizes.loc[(c.chunk_id,r)]) for c in chunks for r in codecs)
    start=time.perf_counter_ns()
    for i in range(n):
      ci=int(positions[i]);q=int(qpos[i]);c=chunks[ci];pool=pools[c.chunk_id][classes[q]];query=pool[int(choices[i])%len(pool)]
      s=time.perf_counter_ns();actual,_,_,_=store.execute_query(c.chunk_id,query);query_ns=time.perf_counter_ns()-s;pred_query+=prof[ci,active[ci],q]
      s=time.perf_counter_ns();expected=m.base.execute_on_array(c.values,query);assert m.base.results_equal(actual,expected);truth_ns+=time.perf_counter_ns()-s
      ctl=controllers[ci];proposal=None;s=time.perf_counter_ns()
      if ctl is not None:
        if policy in ['repressdb','work_function']:proposal=ctl.step(q,prof[ci],durable[ci])
        else:proposal,_=ctl.observe_and_decide(q,prof[ci],codec_cost[ci] if policy=='legacy_repress' else durable[ci],codecs,i,c.chunk_id)
      ct=time.perf_counter_ns()-s if ctl is not None else 0;controller_ns+=ct;lat[i]=query_ns+ct
      if i//1000!=interval:interval=i//1000;used=0
      if proposal is not None:
        rb=int(sizes.loc[(c.chunk_id,codecs[proposal])])
        if used+rb>limit:denied+=1;continue
        before=active[ci];r=executor.migrate_profiled(c.chunk_id,codecs[proposal]);used+=r['rewrite_bytes'];migration_ns+=r['migration_time_ns'];lat[i]+=r['migration_time_ns'];pred_mig+=durable[ci,before,proposal]
        active[ci]=proposal;ctl.committed_switch(proposal)
        mig.append({'event':i,'chunk_id':c.chunk_id,'source':codecs[before],'target':codecs[proposal],'estimated_ns':durable[ci,before,proposal],**r})
    elapsed=time.perf_counter_ns()-start
    integrity=raw.integrity_check();assert integrity['pass']
    for c in chunks:assert np.array_equal(store.read_all(c.chunk_id),c.values)
    store.close();np.savez_compressed(dest/'latencies.npz',service_ns=lat)
    if mig:pd.DataFrame(mig).to_csv(dest/'migrations.csv',index=False)
    write(dest/'summary.json',{'size':size,'group':group,'family':FAMILIES[family],'policy':policy,'seed':seed,'n_queries':n,'service_ns':int(lat.sum()),'p99_ns':float(np.percentile(lat,99)),'p999_ns':float(np.percentile(lat,99.9)),'max_ns':int(lat.max()),'controller_ns':controller_ns,'migration_ns':migration_ns,'migration_count':len(mig),'rewrite_bytes':sum(x['rewrite_bytes'] for x in mig),'budget_denied':denied,'budget_limit_bytes':limit,'prediction_ns':pred_query+pred_mig,'elapsed_ns':elapsed,'truth_ns':truth_ns,'correct':True,'integrity':integrity,'code_sha256':sha(Path(__file__))})
    print('CELL',name,'s',round(lat.sum()/1e9,3),'migrations',len(mig),flush=True)

def campaign():
    # One performance worker at a time, cyclic policy order. No simultaneous calibration or plots.
    for index,(size,g,f) in enumerate(itertools.product([8192,16384],range(3),range(3))):
      for policy in POLICIES[index%len(POLICIES):]+POLICIES[:index%len(POLICIES)]:
        subprocess.run([sys.executable,str(Path(__file__)),'cell','--size',str(size),'--group',str(g),'--family',str(f),'--policy',policy],check=True)

def aggregate():
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'cells').glob('*/summary.json'))];df=pd.DataFrame(rows);df.to_csv(OUT/'summary.csv',index=False)
    stats=[];rng=np.random.default_rng(20261031)
    for metric in ['service_ns','p99_ns','p999_ns','rewrite_bytes','migration_count']:
      pivot=df.pivot(index=['size','group','family'],columns='policy',values=metric)
      for den in ['static','shape','work_function','legacy_repress']:
        a=pivot['repressdb'].to_numpy(float);b=pivot[den].to_numpy(float)
        if np.all(a>0) and np.all(b>0):
          log=np.log(a/b);boot=np.exp(log[rng.integers(0,len(log),(20000,len(log)))].mean(axis=1));gm=float(np.exp(log.mean()));lo,hi=np.percentile(boot,[2.5,97.5])
        else:gm=lo=hi=None
        stats.append({'metric':metric,'denominator':den,'n':len(a),'gm_ratio':gm,'ci_low':lo,'ci_high':hi,'median_ratio':float(np.median(a/b)) if np.all(b>0) else None,'wins':int((a<b).sum()),'numerator_total':float(a.sum()),'denominator_total':float(b.sum())})
    pd.DataFrame(stats).to_csv(OUT/'paired_statistics.csv',index=False)
    print(pd.DataFrame(stats).to_string(index=False))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');psutil.Process().cpu_affinity([0]);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','calibrate','cell','campaign','aggregate']);ap.add_argument('--size',type=int,default=8192);ap.add_argument('--group',type=int,default=0);ap.add_argument('--family',type=int,default=0);ap.add_argument('--policy',default='repressdb');a=ap.parse_args()
    if a.mode=='cell':cell(a.size,a.group,a.family,a.policy)
    else:globals()[a.mode]()
