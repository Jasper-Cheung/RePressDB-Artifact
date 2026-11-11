"""Render final figures and tables from the verified FGCS-2.1 archive."""
from pathlib import Path
import json,sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
ROOT=HERE.parents[2]/'FGCS_revision_v2';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
LABEL={'static':'Static','shape':'Shape-EWMA-BE','horizon':'Horizon-RePress','work_function':'Work-function','credit':'Credit-RePress','risk':'Risk-RePress'}
ORDER=list(LABEL);COLORS=['#8b9698','#e39c39','#477ba2','#8c63a6','#64a4b7','#118c7b']
def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight');fig.savefig(FIG/f'{name}.png',dpi=220,bbox_inches='tight');plt.close(fig)
def run():
    assert json.loads((OUT/'verification.json').read_text())['passed']
    df=pd.read_csv(OUT/'summary.csv');assert len(df)==144
    stats=pd.read_csv(OUT/'paired_statistics.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5})
    pairs=pd.read_csv(HERE/'validation/paired_migrations.csv').groupby(['chunk_id','size','source','target'])[['original_ns','efficient_ns']].median().reset_index()
    ratios=pairs.efficient_ns/pairs.original_ns
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.65),layout='constrained')
    for size,color in [(8192,'#477ba2'),(16384,'#118c7b')]:
        part=pairs[pairs['size']==size];axes[0].scatter(part.original_ns/1e6,part.efficient_ns/1e6,s=17,color=color,alpha=.75,label=f'{size//1024}K')
    lower=min(pairs.original_ns.min(),pairs.efficient_ns.min())/1e6*.95;upper=max(pairs.original_ns.max(),pairs.efficient_ns.max())/1e6*1.03
    axes[0].plot([lower,upper],[lower,upper],ls='--',lw=.9,color='#666666');axes[0].set(xlabel='Six-transaction replacement (ms)',ylabel='Three-transaction replacement (ms)');axes[0].legend(frameon=False)
    ordered=np.sort(ratios);axes[1].step(ordered,np.arange(1,len(ordered)+1)/len(ordered),where='post',color='#118c7b',lw=1.7)
    axes[1].axvline(np.median(ordered),ls='--',lw=.9,color='#666666');axes[1].set(xlabel='Paired replacement-time ratio',ylabel='Fraction of directed pairs',ylim=(0,1.02));axes[1].text(.04,.95,f'Median {np.median(ordered):.3f}\nCatalog commits: 6 → 3',transform=axes[1].transAxes,va='top')
    save(fig,'executor_cost')
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.6),layout='constrained')
    den=[p for p in ORDER if p!='risk']
    for ax,metric,xlabel in zip(axes,['service_ns','p99_ns'],['Risk-RePress / comparison service time','Risk-RePress / comparison event p99']):
        rows=stats[stats.metric==metric].set_index('denominator').loc[den]
        for i,(name,r) in enumerate(rows.iterrows()):
            ax.errorbar(r.gm_ratio,i,xerr=[[r.gm_ratio-r.ci_low],[r.ci_high-r.gm_ratio]],fmt='o',ms=4.5,capsize=2,color=COLORS[ORDER.index(name)])
        ax.axvline(1,ls='--',lw=.8,color='#666666');ax.set(yticks=np.arange(5),yticklabels=[LABEL[x] for x in den],xlabel=xlabel);ax.invert_yaxis()
    save(fig,'policy_comparison')
    series={};index=['size','group','family']
    for policy in ('risk','static','credit'):
        sub=df[df.policy==policy];items={}
        for _,r in sub.iterrows():
            family=['R1_A_gradual_periodic','R1_B_mean_reverting','R1_C_noisy_bursts','R2_D_persistent_shift'].index(r.family)
            path=OUT/'cells'/f'{r["size"]}_{r.group}_{family}_{policy}'/'latencies.npz'
            a=np.load(path)['service_ns'];items[(r['size'],r.group,r.family)]=np.cumsum(a)[999::1000]
        series[policy]=items
    fig,axes=plt.subplots(2,2,figsize=(7.15,4.35),layout='constrained')
    for row,stratum in enumerate(['Continuous mixtures','Persistent shift']):
        selected=[k for k in sorted(series['risk']) if (k[2]=='R2_D_persistent_shift')==(row==1)]
        for ax,den in zip(axes[row],['static','credit']):
            a=np.asarray([series['risk'][k]/series[den][k] for k in selected]);x=np.arange(1000,160001,1000)/1000
            ax.plot(x,np.exp(np.log(a).mean(axis=0)),color='#118c7b',lw=1.5)
            ax.fill_between(x,*np.percentile(a,[25,75],axis=0),color='#118c7b',alpha=.15,lw=0)
            ax.axhline(1,ls='--',lw=.8,color='#666666');ax.set(xlabel='Queries completed (thousands)',ylabel='Cumulative service ratio',title=stratum+': Risk / '+LABEL[den])
    save(fig,'service_trajectories')
    service=df.pivot(index=index,columns='policy',values='service_ns');tail=df.pivot(index=index,columns='policy',values='p99_ns')
    rng=np.random.default_rng(202611019);policy_summary={};rows=[]
    for policy in ORDER:
        part=df[df.policy==policy];ratio=service[policy]/service.static;log=np.log(ratio.to_numpy());boot=np.exp(log[rng.integers(0,24,(20000,24))].mean(axis=1));lo,hi=np.percentile(boot,[2.5,97.5])
        if policy=='risk':
            matched=stats[(stats.metric=='service_ns')&(stats.denominator=='static')].iloc[0];lo,hi=matched.ci_low,matched.ci_high
        record=dict(service_ratio=float(np.exp(log.mean())),ci=[float(lo),float(hi)],p99_ratio=float(np.exp(np.log(tail[policy]/tail.static).mean())),migrations=int(part.migration_count.sum()),rewrite_mib=float(part.rewrite_bytes.sum()/2**20),controller_percent=float(100*part.controller_ns.sum()/part.service_ns.sum()))
        policy_summary[policy]=record
        ci='--' if policy=='static' else f'[{lo:.3f}, {hi:.3f}]'
        rows.append(f'{LABEL[policy]} & {record["service_ratio"]:.3f} & {ci} & {record["p99_ratio"]:.3f} & {record["migrations"]} & {record["rewrite_mib"]:.3f} & {record["controller_percent"]:.2f} \\\\')
    table=r'''\begin{table*}[!t]
\centering\small
\caption{Common-executor comparison over all 24 instances. Service and event-p99 ratios are geometric means relative to Static; intervals are paired-bootstrap 95\% intervals. Rewrites cover target payloads and representation metadata.}
\label{tab:newresults}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Policy & Service ratio & 95\% interval & Event p99 ratio & Migrations & Rewrite (MiB) & Control (\%) \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table*}
'''
    (ROOT/'sections/table_new_results.tex').write_text(table,encoding='utf-8')
    pivot=df.pivot(index=index,columns='policy',values='service_ns');worst=(pivot.risk/pivot.static).idxmax();case=df.set_index(index).loc[worst].set_index('policy').to_dict('index')
    stratified=pd.read_csv(OUT/'stratified_statistics.csv')
    sr=[]
    for label,key,n in [('Continuous','continuous',18),('Persistent','persistent',6)]:
        cells=[]
        for metric,den in [('service_ns','static'),('p99_ns','static'),('service_ns','credit'),('p99_ns','credit')]:
            item=stratified[(stratified.stratum==key)&(stratified.metric==metric)&(stratified.denominator==den)].iloc[0]
            cells.append(f'{item.gm_ratio:.3f} [{item.ci_low:.3f}, {item.ci_high:.3f}]')
        sr.append(label+f' ({n}) & '+' & '.join(cells)+r' \\')
    (ROOT/'sections/table_strata.tex').write_text(r'''\begin{table*}[!t]
\centering\small
\caption{Risk-RePress by workload stratum. Ratios are geometric means with paired-bootstrap 95\% intervals. The number of instances is shown in parentheses.}
\label{tab:strata}
\begin{tabular}{@{}lrrrr@{}}\toprule
Stratum & Service / Static & Event p99 / Static & Service / Credit & Event p99 / Credit \\
\midrule
'''+ '\n'.join(sr)+r'''
\bottomrule\end{tabular}
\end{table*}
''',encoding='utf-8')
    setup=pd.read_csv(HERE/'lookup_setup.csv')
    lookup=setup.groupby(['size','group','family']).lookup_bytes.sum()
    summary=dict(policies=policy_summary,paired=stats.to_dict('records'),stratified=stratified.to_dict('records'),
        controller_setup=dict(median_ms=float(df[df.policy=='risk'].controller_setup_ns.median()/1e6),max_ms=float(df[df.policy=='risk'].controller_setup_ns.max()/1e6),lookup_bytes_median=float(lookup.median()),lookup_bytes_max=int(lookup.max())),
        equivalence=json.loads((HERE/'implementation_equivalence.json').read_text()),
        worst_static_instance=dict(size=int(worst[0]),group=int(worst[1]),family=worst[2],policies=case),executor=json.loads((HERE/'validation/paired_summary.json').read_text()),faults=json.loads((HERE/'validation/combined_fault_summary.json').read_text()))
    (OUT/'manuscript_results.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    (HERE/'environment_plotting.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),indent=2),encoding='utf-8')
    print(json.dumps(policy_summary,indent=2))
if __name__=='__main__':run()
