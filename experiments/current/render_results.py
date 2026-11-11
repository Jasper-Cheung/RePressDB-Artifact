"""Generate manuscript tables/figures only from a complete, audited campaign."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];RESULT=ROOT/'experiments/results';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
names={'static':'Static','shape':'Shape-EWMA-BE','work_function':'Work-function','legacy_repress':'Horizon-RePress','repressdb':'RePressDB'}
order=list(names);colors={'static':'#7F8C8D','shape':'#D68910','work_function':'#7D3C98','legacy_repress':'#2874A6','repressdb':'#148F77'}
df=pd.read_csv(RESULT/'summary.csv');assert len(df)==90 and df.correct.all()
assert (df.groupby(['size','group','family']).size()==5).all()
freeze=json.loads((RESULT/'freeze.json').read_text())
assert freeze['code_sha256']==hashlib.sha256((ROOT/'experiments/fgcs_benchmark.py').read_bytes()).hexdigest()
for name in ['query_calibration.csv','durable_calibration.csv']:
    assert freeze[name]==hashlib.sha256((RESULT/name).read_bytes()).hexdigest()
assert df.n_queries.sum()==14400000
violations=[]
for p in (RESULT/'cells').glob('*/summary.json'):
    d=json.loads(p.read_text());assert d['code_sha256']==freeze['code_sha256'] and d['correct'] and d['integrity']['pass']
    if (p.parent/'migrations.csv').exists():
        m=pd.read_csv(p.parent/'migrations.csv');spent=m.groupby(m.event//1000).rewrite_bytes.sum()
        if (spent>d['budget_limit_bytes']).any():violations.append(p.parent.name)
assert not violations
def gm(a):return float(np.exp(np.log(np.asarray(a,float)).mean()))
def ci(a):
    a=np.log(np.asarray(a,float));rng=np.random.default_rng(20261031);r=np.exp(a[rng.integers(0,len(a),(20000,len(a)))].mean(axis=1));return np.percentile(r,[2.5,97.5])
index=['size','group','family'];base=df[df.policy=='static'].set_index(index)
rows=[]
for pol in order:
    cur=df[df.policy==pol].set_index(index).loc[base.index]
    ratio=cur.service_ns/base.service_ns;tail=cur.p99_ns/base.p99_ns;l,h=ci(ratio)
    rows.append({'policy':pol,'service_ratio':gm(ratio),'service_low':l,'service_high':h,'p99_ratio':gm(tail),'migrations':int(cur.migration_count.sum()),'rewrite_mib':float(cur.rewrite_bytes.sum()/2**20),'controller_fraction':float(cur.controller_ns.sum()/cur.service_ns.sum()),'wins':int((ratio<1).sum())})
tot=pd.DataFrame(rows);tot.to_csv(RESULT/'manuscript_policy_table.csv',index=False)
tex=[r'\begin{table*}[!t]',r'\centering',r'\caption{New end-to-end policy comparison over 18 paired workload instances. Service and event-p99 ratios are normalized to Static; lower values favor the policy. Rewrite bytes count target payloads and representation metadata.}',r'\label{tab:newresults}',r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',r'Policy & Service ratio & 95\% interval & Event p99 ratio & Migrations & Rewrite (MiB) & Control (\%) \\',r'\midrule']
for r in rows:
    interval='--' if r['policy']=='static' else f"[{r['service_low']:.3f}, {r['service_high']:.3f}]"
    tex.append(f"{names[r['policy']]} & {r['service_ratio']:.3f} & {interval} & {r['p99_ratio']:.3f} & {r['migrations']} & {r['rewrite_mib']:.3f} & {100*r['controller_fraction']:.2f}"+r' \\')
tex +=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(ROOT/'sections/table_new_results.tex').write_text('\n'.join(tex),encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
stats=pd.read_csv(RESULT/'paired_statistics.csv')
fig,axs=plt.subplots(1,2,figsize=(7.2,2.6),gridspec_kw={'width_ratios':[1,1.1]})
for i,den in enumerate(['static','shape','work_function','legacy_repress']):
    r=stats[(stats.metric=='service_ns')&(stats.denominator==den)].iloc[0]
    axs[0].errorbar(r.gm_ratio,i,xerr=[[r.gm_ratio-r.ci_low],[r.ci_high-r.gm_ratio]],fmt='o',color=colors[den],capsize=3)
axs[0].axvline(1,color='.4',lw=.8,ls='--');axs[0].set_yticks(range(4),[names[p] for p in ['static','shape','work_function','legacy_repress']]);axs[0].invert_yaxis();axs[0].set_xlabel('RePressDB / comparison service time')
for r in rows:
    axs[1].scatter(r['service_ratio'],r['rewrite_mib'],s=50,color=colors[r['policy']],label=names[r['policy']])
axs[1].axvline(1,color='.4',lw=.8,ls='--');axs[1].set_xlabel('Service time / Static');axs[1].set_ylabel('Total payload rewrite (MiB)');axs[1].legend(fontsize=7,frameon=False,loc='best')
fig.tight_layout();fig.savefig(FIG/'policy_tradeoff.pdf',bbox_inches='tight');fig.savefig(FIG/'policy_tradeoff.png',dpi=180,bbox_inches='tight');plt.close(fig)

dur=pd.read_csv(RESULT/'durable_calibration.csv')
stages=['intent_catalog_ns','decode_source_ns','encode_and_log_ns','validate_and_log_ns','publish_files_ns','publish_catalog_state_ns','catalog_commit_ns','post_commit_validate_ns','cleanup_ns']
labels=['Intent','Decode','Encode + log','Validate + log','Publish files','Publish state','Commit','Post-validate','Cleanup']
means=dur[stages].mean()/1e6
fig,axs=plt.subplots(1,2,figsize=(7.2,2.7));axs[0].barh(labels,means,color=['#566573','#2874A6','#5DADE2','#85C1E9','#AEB6BF','#5D6D7E','#148F77','#73C6B6','#BFC9CA']);axs[0].invert_yaxis();axs[0].set_xlabel('Mean stage time (ms)')
for pol in order:
    cur=df[df.policy==pol];axs[1].scatter(cur.prediction_ns/1e9,cur.service_ns/1e9,s=20,alpha=.7,color=colors[pol],label=names[pol])
lower=min(df.prediction_ns.min(),df.service_ns.min())/1e9;upper=max(df.prediction_ns.max(),df.service_ns.max())/1e9
axs[1].plot([lower,upper],[lower,upper],ls='--',lw=.8,color='.4');axs[1].set_xlabel('Calibrated query + migration cost (s)');axs[1].set_ylabel('Measured complete service time (s)');axs[1].legend(fontsize=6.5,frameon=False)
fig.tight_layout();fig.savefig(FIG/'cost_alignment.pdf',bbox_inches='tight');fig.savefig(FIG/'cost_alignment.png',dpi=180,bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(7.2,2.5));cur=df[df.policy=='repressdb'].set_index(index)
for ax,den in zip(axs,['static','legacy_repress']):
    ref=df[df.policy==den].set_index(index);allrat=[]
    for ix in cur.index:
        size,g,fam=ix;fi=['R1_A_gradual_periodic','R1_B_mean_reverting','R1_C_noisy_bursts'].index(fam)
        a=np.load(RESULT/'cells'/f'{size}_{g}_{fi}_repressdb'/'latencies.npz')['service_ns'];b=np.load(RESULT/'cells'/f'{size}_{g}_{fi}_{den}'/'latencies.npz')['service_ns']
        allrat.append(np.cumsum(a,dtype=float)[999::1000]/np.cumsum(b,dtype=float)[999::1000])
    allrat=np.asarray(allrat);x=np.arange(1,161)*1000
    ax.plot(x, np.exp(np.log(allrat).mean(axis=0)),color=colors['repressdb']);lo,hi=np.percentile(allrat,[25,75],axis=0);ax.fill_between(x,lo,hi,color=colors['repressdb'],alpha=.15)
    ax.axhline(1,color='.4',ls='--',lw=.8);ax.set_xlabel('Queries completed');ax.set_ylabel('Cumulative service ratio');ax.set_title('RePressDB / '+names[den],fontsize=9);ax.ticklabel_format(axis='x',style='sci',scilimits=(3,3))
fig.tight_layout();fig.savefig(FIG/'service_trajectories.pdf',bbox_inches='tight');fig.savefig(FIG/'service_trajectories.png',dpi=180,bbox_inches='tight');plt.close(fig)

audit={'pass':True,'cells':len(df),'verified_queries':int(df.n_queries.sum()),'budget_violations':violations,'all_catalogs_valid':True,'source_hash':freeze['code_sha256'],'query_cost_records':len(pd.read_csv(RESULT/'query_calibration.csv')),'calibration_query_executions':len(pd.read_csv(RESULT/'query_calibration.csv'))*36,'calibration_migration_observations':len(dur),'duration_median_ms':float(dur.ns.median()/1e6),'duration_p95_ms':float(dur.ns.quantile(.95)/1e6)}
(RESULT/'final_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2));print(tot.to_string(index=False))
