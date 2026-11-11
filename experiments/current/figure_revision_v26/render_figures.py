"""Revise Figs. 2--4 using the unchanged final FGCS-2.2 measurements.

Figure 1 and all scientific statistics are preserved. No benchmark is run.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
FINAL=HERE.parent/'method_v22'
ROOT=BASE/'paper/current'
FIG=ROOT/'figures'
OUT=HERE/'results'
LABEL={'static':'Static','shape':'Shape-EWMA-BE','horizon':'Horizon-RePress',
       'work_function':'Work-function','credit':'Credit-RePress','risk':'Risk-RePress'}
ORDER=list(LABEL)
COLORS=['#8b9698','#e39c39','#477ba2','#8c63a6','#64a4b7','#118c7b']
GREEN='#118c7b';SHIFT='#b76a20'
FAMILIES=['R1_A_gradual_periodic','R1_B_mean_reverting','R1_C_noisy_bursts','R2_D_persistent_shift']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)
def run():
    FIG.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    paths=[FINAL/'results/verification.json',FINAL/'results/summary.csv',
           FINAL/'results/paired_statistics.csv',FINAL/'results/stratified_statistics.csv',
           FINAL/'validation/paired_migrations.csv',FINAL/'results/protocol.json']
    assert json.loads(paths[0].read_text())['passed']
    protocol=json.loads(paths[-1].read_text())
    for name,digest in protocol['sources'].items():
        path=FINAL.parent/name if name=='fgcs_benchmark.py' else FINAL/name
        assert sha(path)==digest;paths.append(path)
    df=pd.read_csv(FINAL/'results/summary.csv');assert len(df)==144
    stats=pd.read_csv(FINAL/'results/paired_statistics.csv')
    strata=pd.read_csv(FINAL/'results/stratified_statistics.csv')
    for font_file in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts')/font_file))
    assert Path(font_manager.findfont('Times New Roman',fallback_to_default=False)).name.lower()=='times.ttf'
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
        'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5})

    # Figure 2: preserve the 80 paired medians and ratio ECDF.
    pairs=pd.read_csv(FINAL/'validation/paired_migrations.csv').groupby(
        ['chunk_id','size','source','target'])[['original_ns','efficient_ns']].median().reset_index()
    assert len(pairs)==80
    ratios=pairs.efficient_ns/pairs.original_ns
    pairs.assign(paired_ratio=ratios).to_csv(OUT/'executor_points.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.8),layout='constrained')
    for size,color in [(8192,'#477ba2'),(16384,GREEN)]:
        part=pairs[pairs['size']==size]
        axes[0].scatter(part.original_ns/1e6,part.efficient_ns/1e6,s=17,color=color,
                        alpha=.75,label=f'{size//1024}K values')
    lower=min(pairs.original_ns.min(),pairs.efficient_ns.min())/1e6*.95
    upper=max(pairs.original_ns.max(),pairs.efficient_ns.max())/1e6*1.03
    axes[0].plot([lower,upper],[lower,upper],ls='--',lw=.9,color='#666666',label='Equal time')
    axes[0].set(xlabel='Six-transaction replacement (ms)',ylabel='Three-transaction replacement (ms)',
                title='(a) Replacement time')
    axes[0].legend(frameon=False,fontsize=7.5)
    ordered=np.sort(ratios)
    axes[1].step(ordered,np.arange(1,len(ordered)+1)/len(ordered),where='post',color=GREEN,lw=1.7)
    axes[1].axvline(np.median(ordered),ls='--',lw=.9,color='#666666')
    axes[1].set(xlabel='Replacement-time ratio (three / six)',ylabel='Fraction of directed pairs',
                ylim=(0,1.02),title='(b) Time ratios')
    axes[1].text(.04,.95,f'Median {np.median(ordered):.3f}\nCatalog commits: 6 to 3',
                 transform=axes[1].transAxes,va='top')
    save(fig,'executor_cost')

    # Figure 3: same point estimates and paired-bootstrap intervals.
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.8),layout='constrained')

    den=[p for p in ORDER if p!='risk'];comparison=[]
    for ax,metric,title,xlabel in zip(axes,['service_ns','p99_ns'],
        ['(a) Service time, n=24','(b) Event p99, n=24'],
        ['Service-time ratio (Risk / baseline)','Event-p99 ratio (Risk / baseline)']):
        rows=stats[stats.metric==metric].set_index('denominator').loc[den]
        pivot=df.pivot(index=['size','group','family'],columns='policy',values=metric)
        for i,(name,r) in enumerate(rows.iterrows()):
            actual=float(np.exp(np.log(pivot.risk/pivot[name]).mean()))
            assert np.isclose(actual,r.gm_ratio,rtol=1e-12)
            comparison.append(dict(metric=metric,denominator=name,ratio=actual,ci_low=r.ci_low,ci_high=r.ci_high))
            ax.errorbar(r.gm_ratio,i,xerr=[[r.gm_ratio-r.ci_low],[r.ci_high-r.gm_ratio]],
                        fmt='o',ms=4.5,capsize=2,color=COLORS[ORDER.index(name)])
        ax.axvline(1,ls='--',lw=.8,color='#666666')
        ax.set(yticks=np.arange(5),yticklabels=[LABEL[x] for x in den],xlabel=xlabel,title=title)
        ax.invert_yaxis()
    pd.DataFrame(comparison).to_csv(OUT/'policy_points.csv',index=False)
    save(fig,'policy_comparison')

    # Figure 4: preserve per-instance cumulative ratios, GM, and pointwise IQR.
    series={}
    for policy in ('risk','static','credit'):
        items={}
        for _,r in df[df.policy==policy].iterrows():
            path=FINAL/'results/cells'/f'{r["size"]}_{r.group}_{FAMILIES.index(r.family)}_{policy}/latencies.npz'
            paths.append(path);a=np.load(path)['service_ns']
            assert len(a)==160000 and int(a.sum())==int(r.service_ns)
            items[(r['size'],r.group,r.family)]=np.cumsum(a)[999::1000]
        series[policy]=items
    fig,axes=plt.subplots(2,2,figsize=(7.15,4.75),layout='constrained',sharey='col')
    fig.legend(handles=[Line2D([0],[0],color=GREEN,lw=1.5,label='Geometric mean'),
        Patch(facecolor=GREEN,alpha=.15,label='Across-instance IQR (25th-75th)'),
        Line2D([0],[0],color=SHIFT,lw=1.1,ls=':',label='Workload shift (20k)')],
        loc='outside upper center',ncol=3,frameon=False,fontsize=7)
    output=[];bounds=[[],[]];endpoints=[];x=np.arange(1000,160001,1000)/1000
    for row,stratum in enumerate(['continuous','persistent']):
        selected=[k for k in sorted(series['risk']) if (k[2]==FAMILIES[-1])==(row==1)]
        n=len(selected);assert n==(18 if row==0 else 6)
        for col,den in enumerate(['static','credit']):
            ax=axes[row,col]
            a=np.asarray([series['risk'][k]/series[den][k] for k in selected])
            gm=np.exp(np.log(a).mean(axis=0));lo,hi=np.percentile(a,[25,75],axis=0)
            expected=strata[(strata.stratum==stratum)&(strata.metric=='service_ns')&(strata.denominator==den)].iloc[0]
            assert np.isclose(gm[-1],expected.gm_ratio,rtol=1e-12)
            endpoints.append(dict(stratum=stratum,denominator=den,n=n,ratio=float(gm[-1])))
            output.extend(dict(stratum=stratum,denominator=den,queries=int(t*1000),n=n,
                               geometric_mean=float(v),q25=float(l),q75=float(h)) for t,v,l,h in zip(x,gm,lo,hi))
            bounds[col].extend([lo.min(),hi.max(),gm.min(),gm.max()])
            ax.plot(x,gm,color=GREEN,lw=1.5)
            ax.fill_between(x,lo,hi,color=GREEN,alpha=.15,lw=0)
            ax.axhline(1,ls='--',lw=.8,color='#666666')
            title=f'({"abcd"[row*2+col]}) {"Continuous" if row==0 else "Persistent"}, n={n}: Risk / {"Static" if den=="static" else "Credit"}'
            ax.set(xlabel='Queries completed (thousands)',ylabel='Cumulative service ratio',title=title,xlim=(0,160))
            ax.set_xticks(np.arange(0,161,20))
            if row==1:
                ax.axvline(20,color=SHIFT,ls=':',lw=1.1)
                ax.text(23,.96,'20k shift',transform=ax.get_xaxis_transform(),color=SHIFT,
                        fontsize=7,va='top',bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
    limits=[]
    for col in (0,1):
        low=min(bounds[col]);high=max(bounds[col]);padding=.04*(high-low)
        limits.append([float(low-padding),float(high+padding)])
        axes[0,col].set_ylim(low-padding,high+padding)
        axes[0,col].yaxis.set_major_locator(MaxNLocator(nbins=5))
    pd.DataFrame(output).to_csv(OUT/'trajectory_points.csv',index=False)
    save(fig,'service_trajectories')
    report={'passed':True,'new_performance_executions':0,'paired_observations':240,'directed_pairs':80,
        'executor_median_ratio':float(np.median(ratios)),'policy_points':len(comparison),
        'event_arrays_checked':72,'trajectory_points':len(output),'trajectory_endpoints':endpoints,
        'figure4_y_limits_by_column':limits,'figure1_method_sha256':sha(ROOT/'sections/method_v2.tex'),
        'inputs':[dict(file=p.relative_to(BASE).as_posix(),sha256=sha(p)) for p in sorted(set(paths))],
        'figure_hashes':{p.name:sha(p) for p in sorted(FIG.glob('*.pdf'))},'render_source_sha256':sha(Path(__file__))}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'},indent=2))
if __name__=='__main__':run()
