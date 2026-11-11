"""Read-only reanalysis of the frozen FGCS-2.2 campaign for narrative revision.

No codec execution, migration, policy tuning, or new performance trial occurs.
Logged gates, calibrated post-action opportunity, and stable-window agreement
are derived with explicit denominators and input hashes.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;FINAL=EXP/'method_v22/results';OUT=HERE/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def run():
    OUT.mkdir(exist_ok=True);inputs=[]
    def read(p):inputs.append(p);return json.loads(p.read_text(encoding='utf-8'))
    verification=read(FINAL/'verification.json');assert verification['passed'] and verification['cells']==144
    protocol=read(FINAL/'protocol.json')
    for name,digest in protocol['sources'].items():
        path=EXP/name if name=='fgcs_benchmark.py' else FINAL.parent/name
        assert sha(path)==digest
    calibration=read(FINAL/'query_calibration.json');codecs=sorted({r['codec'] for r in calibration});assert len(codecs)==6
    profiles={}
    for r in calibration:
        profiles.setdefault(r['chunk_id'],np.full((6,29),np.nan))[codecs.index(r['codec']),r['feature']]=r['mean_ns']
    assert all(np.isfinite(a).all() for a in profiles.values())
    inputs.extend([FINAL/'summary.csv',FINAL/'durable_calibration.csv',EXP/'results/chunk_manifest.csv'])
    summary=pd.read_csv(FINAL/'summary.csv');migrations=pd.read_csv(FINAL/'durable_calibration.csv').groupby(['chunk_id','source','target']).ns.median()
    manifest=pd.read_csv(EXP/'results/chunk_manifest.csv');truth={size:np.load(EXP/'results'/f'values_{size}.npz') for size in (8192,16384)}
    inputs.extend(EXP/'results'/f'values_{size}.npz' for size in (8192,16384))
    decisions=[];episodes=[];windows=[];case_rows=[];proposals=0;accepted=0;denied=0;prediction_checks=0
    for _,case in summary[summary.policy=='risk'].iterrows():
        size=int(case['size']);group=int(case.group);family=protocol['families'].index(case.family)
        folder=FINAL/'cells'/f'{size}_{group}_{family}_risk';static_folder=FINAL/'cells'/f'{size}_{group}_{family}_static'
        stored=read(folder/'summary.json');static=read(static_folder/'summary.json');logs=read(folder/'decisions.json')
        inputs.extend([folder/'latencies.npz',static_folder/'latencies.npz'])
        risk_npz=np.load(folder/'latencies.npz');static_npz=np.load(static_folder/'latencies.npz')
        risk_q=risk_npz['query_ns'];risk_service=risk_npz['service_ns'];features=risk_npz['feature_ids']
        static_q=static_npz['query_ns'];static_service=static_npz['service_ns']
        if (folder/'migrations.csv').exists():
            inputs.append(folder/'migrations.csv');actions=pd.read_csv(folder/'migrations.csv')
        else:actions=pd.DataFrame(columns=['event','chunk_id','source','target','reason','migration_time_ns','rewrite_bytes'])
        weights=np.arange(1,9,dtype=float)**-.8;weights/=weights.sum();route=np.random.default_rng(int(case.seed)).choice(8,160000,p=weights)
        wanted=set(manifest[(manifest.chunk_size==size)&manifest.building_id.isin(protocol['groups'][group])].chunk_id)
        ids=[str(cid) for cid in truth[size]['chunk_ids'] if str(cid) in wanted];assert len(ids)==8
        predicted_queries=0.;predicted_actions=0.;case_proposals=0;case_denied=0
        for ci,cid in enumerate(ids):
            times=np.flatnonzero(route==ci);local_f=features[times];mu=profiles[cid]
            a0=codecs.index(stored['initial_codecs'][ci]);local_actions=actions[actions.chunk_id==cid].sort_values('event').reset_index(drop=True)
            action_map={int(np.searchsorted(times,int(a.event)))+1:a for a in local_actions.itertuples(index=False)}
            last_switch=-10**9;confirm_target=None;confirm=0
            for d in logs[cid]:
                n=int(d['local_count']);feasible=np.asarray(d['feasible'],bool);target=int(d['target']);cur=int(d['active'])
                lw=np.bincount(local_f[n-768:n],minlength=29)/768
                sw=np.bincount(local_f[n-256:n],minlength=29)/256
                lc=mu@lw;sc=mu@sw;unconstrained=int(np.argmin(lc))
                expected_target=a0 if not feasible[cur] else int(np.argmin(np.where(feasible,lc,np.inf)))
                assert target==expected_target,(folder.name,cid,n)
                if d['reason'] in ('candidate','risk_repair'):
                    confirm=confirm+1 if confirm_target==target else 1;confirm_target=target
                    required=2 if d['reason']=='risk_repair' else 3
                    if confirm<required:outcome='waiting_confirmation'
                    elif d['reason']!='risk_repair' and n-last_switch<768:outcome='waiting_cooldown'
                    elif d['reason']!='risk_repair' and d['credit']<1.25*d['admission_price']:outcome='waiting_credit'
                    else:outcome='proposed_repair' if d['reason']=='risk_repair' else 'proposed_normal'
                else:
                    confirm_target=None;confirm=0;outcome=d['reason']
                is_proposal=outcome.startswith('proposed_');assert is_proposal==(d['proposal'] is not None)
                price_eligible=(d['reason']=='candidate' and confirm>=3 and n-last_switch>=768)
                return_price_binding=bool(price_eligible and d['credit']>=1.25*d['migration_cost'] and d['credit']<1.25*d['admission_price'])
                committed=n in action_map
                assert not committed or (is_proposal and codecs[target]==action_map[n].target)
                row=dict(size=size,group=group,family=case.family,chunk_id=cid,local_count=n,active=codecs[cur],target=codecs[target],logged_reason=d['reason'],outcome=outcome,
                         feasible_layouts=int(feasible.sum()),any_infeasible=not bool(feasible.all()),unconstrained_target=codecs[unconstrained],unconstrained_target_filtered=not bool(feasible[unconstrained]),
                         price_eligible=bool(price_eligible),return_price_binding=return_price_binding,proposal=is_proposal,committed=committed,budget_deferred=bool(is_proposal and not committed))
                decisions.append(row)
                if is_proposal:
                    proposals+=1;case_proposals+=1
                    if committed:accepted+=1
                    else:denied+=1;case_denied+=1
                # Stable-window comparison excludes any migration within 768 requests.
                if n-last_switch>=768 and not committed:
                    qr=[];er=[]
                    for width in (256,768):
                        selected=times[n-width:n]
                        qr.append(float(np.percentile(risk_q[selected],99)/np.percentile(static_q[selected],99)))
                        er.append(float(np.percentile(risk_service[selected],99)/np.percentile(static_service[selected],99)))
                    windows.append(dict(size=size,group=group,family=case.family,chunk_id=cid,local_count=n,role='reference' if cur==a0 else 'adapted',model_feasible=bool(feasible[cur]),query_ratio=max(qr),event_ratio=max(er),query_within=max(qr)<=1.1,event_within=max(er)<=1.1))
                if committed:last_switch=n;confirm_target=None;confirm=0
            cursor=0;active=a0
            for e,a in enumerate(local_actions.itertuples(index=False)):
                pos=int(np.searchsorted(times,int(a.event)));source=codecs.index(a.source);target=codecs.index(a.target)
                assert active==source
                predicted_queries+=float(mu[active,local_f[cursor:pos+1]].sum());active=target;cursor=pos+1
                m_forward=float(migrations.loc[cid,a.source,a.target]);m_return=0. if target==a0 else float(migrations.loc[cid,a.target,codecs[a0]])
                predicted_actions+=m_forward
                terminal=e+1==len(local_actions)
                end=len(times) if terminal else int(np.searchsorted(times,int(local_actions.iloc[e+1].event)))+1
                gain=mu[source,local_f[pos+1:end]]-mu[target,local_f[pos+1:end]]
                cumulative=np.cumsum(gain);net=float(cumulative[-1]) if len(cumulative) else 0.
                crossing=np.flatnonzero(cumulative>=float(a.migration_time_ns))
                roundtrip_actual=None;roundtrip_covered=None
                if not terminal and source==a0:
                    nxt=local_actions.iloc[e+1]
                    if nxt.reason=='risk_repair' and nxt.target==codecs[a0]:
                        roundtrip_actual=float(a.migration_time_ns)+float(nxt.migration_time_ns)
                        roundtrip_covered=net>=roundtrip_actual
                episodes.append(dict(size=size,group=group,family=case.family,stratum='persistent' if family==3 else 'continuous',chunk_id=cid,event=int(a.event),source=a.source,target=a.target,reason=a.reason,
                    observed_queries=end-pos-1,closed_by_next_migration=not terminal,right_censored=terminal,modeled_saving_ns=net,executed_forward_ns=int(a.migration_time_ns),modeled_forward_ns=m_forward,modeled_return_ns=m_return,
                    modeled_forward_return_ns=m_forward+m_return,end_covers_executed_forward=net>=float(a.migration_time_ns),end_covers_modeled_forward_return=net>=m_forward+m_return,
                    first_forward_crossing_queries=int(crossing[0]+1) if len(crossing) else None,saving_forward_ratio=net/float(a.migration_time_ns),actual_roundtrip_ns=roundtrip_actual,model_covers_actual_roundtrip=roundtrip_covered))
            predicted_queries+=float(mu[active,local_f[cursor:]].sum())
        assert np.isclose(predicted_queries+predicted_actions,stored['prediction_ns'],rtol=0,atol=1e-3),(folder.name,predicted_queries+predicted_actions,stored['prediction_ns'])
        prediction_checks+=1;assert case_denied==stored['budget_denied']
        marker=EXP/'method_v2/results'/f'calibration_{size}_{group}.json';cal_group=read(marker)['elapsed_ns']
        delta=(float(static['service_ns'])-float(stored['service_ns']))/160000
        additional_setup=max(0,int(stored['controller_setup_ns'])-int(static['controller_setup_ns']))
        case_rows.append(dict(size=size,group=group,family=case.family,stratum='persistent' if family==3 else 'continuous',proposals=case_proposals,budget_deferrals=case_denied,
                              service_static_ns=int(static['service_ns']),service_risk_ns=int(stored['service_ns']),net_service_saving_per_query_ns=delta,group_calibration_ns=cal_group,additional_setup_ns=additional_setup,
                              conditional_break_even_queries=(cal_group+additional_setup)/delta if delta>0 else None))
    assert accepted==66 and denied==1 and proposals==67
    D=pd.DataFrame(decisions);E=pd.DataFrame(episodes);W=pd.DataFrame(windows);C=pd.DataFrame(case_rows)
    for name,frame in [('decisions',D),('migration_episodes',E),('stable_windows',W),('amortization_scenarios',C)]:frame.to_csv(OUT/(name+'.csv'),index=False)
    # Confirm the query-window cohort exactly matches the earlier descriptive audit.
    inputs.append(FINAL/'stable_tail_windows.csv');old=pd.read_csv(FINAL/'stable_tail_windows.csv')
    idx=['size','group','family','chunk_id','local_count'];merged=W.merge(old,on=idx,validate='one_to_one',suffixes=('_new','_old'));assert len(merged)==len(W)==len(old)
    assert np.allclose(merged.query_ratio,merged.max_window_p99_ratio,rtol=1e-12,atol=0)
    assert (merged.model_feasible_new==merged.model_feasible_old).all()
    mechanism=dict(decisions=len(D),outcomes=D.outcome.value_counts().to_dict(),any_infeasible=int(D.any_infeasible.sum()),unconstrained_target_filtered=int(D.unconstrained_target_filtered.sum()),
        price_eligible=int(D.price_eligible.sum()),return_price_binding=int(D.return_price_binding.sum()),proposals=proposals,committed=accepted,budget_deferred=denied,
        proposals_by_reason=D[D.proposal].logged_reason.value_counts().to_dict(),actions_by_reason=E.reason.value_counts().to_dict())
    payback=[]
    for stratum,part in E.groupby(['stratum','reason']):
        payback.append(dict(stratum=stratum[0],reason=stratum[1],episodes=len(part),closed=int(part.closed_by_next_migration.sum()),right_censored=int(part.right_censored.sum()),
            covered_forward=int(part.end_covers_executed_forward.sum()),covered_reserved=int(part.end_covers_modeled_forward_return.sum()),median_saving_forward_ratio=float(part.saving_forward_ratio.median()),
            total_modeled_saving_ns=float(part.modeled_saving_ns.sum()),total_executed_forward_ns=int(part.executed_forward_ns.sum())))
    agreement=[]
    for role,part in W.groupby('role'):
        yes=part[part.model_feasible]
        agreement.append(dict(role=role,stable_windows=len(part),model_feasible=len(yes),query_within=int(yes.query_within.sum()),event_within=int(yes.event_within.sum())))
    P=C[C.stratum=='persistent'];assert (P.net_service_saving_per_query_ns>0).all()
    report=dict(analysis='existing records only; no new performance execution',mechanism=mechanism,payback=payback,agreement=agreement,
        amortization=dict(scope='conditional fixed-rate reuse scenario; full measured group commissioning assigned to adaptation and zero to Static; measured service savings per query held fixed',
            persistent_cases=len(P),min_queries=float(P.conditional_break_even_queries.min()),max_queries=float(P.conditional_break_even_queries.max()),median_queries=float(P.conditional_break_even_queries.median()),
            continuous_positive_estimates=int((C[C.stratum=='continuous'].net_service_saving_per_query_ns>0).sum())),
        measured_roundtrips=int(E.actual_roundtrip_ns.notna().sum()),modeled_roundtrips_covered=int(E.model_covers_actual_roundtrip.eq(True).sum()),
        prediction_checks=prediction_checks,rewrites_reduction_percent=100*(1-summary[summary.policy=='risk'].rewrite_bytes.sum()/summary[summary.policy=='credit'].rewrite_bytes.sum()),
        inputs=[dict(file=str(p.relative_to(EXP)),sha256=sha(p)) for p in sorted(set(inputs))],analysis_sha256=sha(Path(__file__)))
    write(OUT/'analysis_summary.json',report);print(json.dumps({k:v for k,v in report.items() if k!='inputs'},indent=2))
if __name__=='__main__':run()
