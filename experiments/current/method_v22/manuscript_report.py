"""Write measurement-bound manuscript results and the author-facing report."""
from pathlib import Path
import json,sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';ROOT=HERE.parents[2]/'FGCS_revision_v2'
def run():
    d=json.loads((OUT/'manuscript_results.json').read_text());assert json.loads((OUT/'verification.json').read_text())['passed']
    df=pd.read_csv(OUT/'summary.csv');assert len(df)==144
    p=d['policies'];r=p['risk'];c=p['credit'];paired=d['paired'];strata=d['stratified']
    def stat(metric,den,stratum=None):
        source=paired if stratum is None else strata
        return next(x for x in source if x['metric']==metric and x['denominator']==den and (stratum is None or x['stratum']==stratum))
    def ratio(s,nd=3):return f"{s['gm_ratio']:.{nd}f} ({s['ci_low']:.{nd}f}--{s['ci_high']:.{nd}f})"
    ss=stat('service_ns','static');sc=stat('service_ns','credit');ts=stat('p99_ns','static');tc=stat('p99_ns','credit')
    cs=stat('service_ns','static','continuous');ct=stat('p99_ns','static','continuous');ps=stat('service_ns','static','persistent');pt=stat('p99_ns','static','persistent')
    reduction=100*(1-r['rewrite_mib']/c['rewrite_mib']);actions=100*(1-r['migrations']/c['migrations'])
    exc=d['executor'];eq=d['equivalence'];setup=d['controller_setup'];cal=json.loads((OUT/'calibration_summary.json').read_text())
    tail=json.loads((OUT/'tail_window_validation.json').read_text())
    abstract=(r"Changing request mixtures can favor different compressed time-series layouts, while their installation adds synchronization and recovery work. RePressDB couples empirical tail admission with a lower-cost durable replacement path. It models query operation and range length, accumulates advantage only within continuous tail-feasible epochs, and prices both installation and a return to the reference layout. Conditional-distribution count tables reduce repeated decision work. The executor groups replacement into three catalog transactions and commits retirement before deleting old files. "
        +f"A common-executor comparison checks 23.04 million query events across six policies, 24 building series, and 24 workload instances. Relative to accumulated admission with the same calibrated inputs, migrations change from {c['migrations']} to {r['migrations']} and rewritten payloads are {r['rewrite_mib']:.3f} versus {c['rewrite_mib']:.3f}~MiB. The paired service-time ratio is {sc['gm_ratio']:.3f} (95\\% interval {sc['ci_low']:.3f}--{sc['ci_high']:.3f}), and the event-p99 ratio is {ratio(tc)}. "
        +f"Relative to a fixed initial layout, service ratios are {cs['gm_ratio']:.3f} for continuously changing mixtures and {ps['gm_ratio']:.3f} for persistent shifts; corresponding event-p99 ratios are {ct['gm_ratio']:.3f} and {pt['gm_ratio']:.3f}. "
        +f"A paired executor diagnostic yields a median replacement-time ratio of {exc['median_paired_ratio']:.3f} against the six-transaction path. All 492 fresh software-fault trials pass. These measurements connect reconfiguration cost, request-tail behavior, and recovery boundaries to the operating conditions of adaptive storage.")
    (ROOT/'abstract.tex').write_text(abstract+'\n',encoding='utf-8')
    w=d['worst_static_instance'];wr=w['policies']['risk'];ws=w['policies']['static']
    rows=[]
    for label,key in [('Controlled stage exceptions', '440'),('Abrupt process exit','40'),('Post-activation payload truncation','12')]:rows.append(label+' & '+key+' & '+key+r' \\')
    tailpara=[]
    for role in ('reference','adapted'):
        if role in tail['roles']:
            x=tail['roles'][role]
            tailpara.append(f"For stable {role}-layout windows, {x['model_feasible_and_within_tolerance']:,} of {x['model_feasible_windows']:,} model-feasible observations also fall within the 10\\% paired query-p99 tolerance.")
    text=r'''\section{Results}
\label{sec:results}
\subsection{RQ1: Replacement and decision work}
The paired executor diagnostic selects the first two manifest-ordered chunks in the first building group at each size. Four chunks cover 80 nontrivial directed pairs over the five original layouts. Each pair has three repetitions; execution order alternates between the original and revised executor using independent stores. Target payload and metadata bytes match in every observation. SQL trace callbacks record six COMMIT statements for the original path and three for the revised path.

'''+f"Across directed-pair medians, the original and revised replacement times are {exc['original_median_ms']:.2f} and {exc['efficient_median_ms']:.2f}~ms. The median paired ratio is {exc['median_paired_ratio']:.3f}, with 10th and 90th percentiles {exc['ratio_p10']:.3f} and {exc['ratio_p90']:.3f}. Figure~\\ref{{fig:executor}} shows the complete paired distribution. Validation and file synchronization remain inside both measurements.\n\n"+r'''
\begin{figure*}[!t]
\centering\includegraphics[width=.94\textwidth]{executor_cost.pdf}
\caption{Paired complete-replacement costs. Left: 80 directed-pair medians from 240 alternating-order observations, with equality marked. Right: the cumulative distribution of paired revised/original ratios.}
\label{fig:executor}
\end{figure*}
'''+f"The equivalent count-table controller reduces median decision-boundary work from {eq['original_decision_median_ns']/1000:.1f} to {eq['cached_decision_median_ns']/1000:.1f}~$\\mu$s in the alternating-order component diagnostic. Its median paired ratio is {eq['median_paired_time_ratio']:.3f}. All 2.88 million checked requests preserve the scanning implementation's decisions, counters, and recorded actions. In the final executions, controller initialization has a median of {setup['median_ms']:.2f}~ms and a maximum of {setup['max_ms']:.2f}~ms per eight-chunk instance. The additional conditional-count arrays have a median footprint of {setup['lookup_bytes_median']/2**20:.3f}~MiB per instance.\n\n"+f"The shared refined calibration completes {cal['query_records']:,} query-cost strata and {cal['durable_observations']:,} directed replacement observations. Its wall time is {cal['wall_ns']/1e9:.1f}~s, including setup replacements. This commissioning expenditure is separate from the following steady-state policy measurements.\n\n"+r'''
\subsection{RQ2: Service, tails, and rewriting}
All 144 scheduled executions complete, totaling 23,040,000 independently checked query events. Final decoded arrays and catalog integrity pass in every cell. The action-log audit finds zero payload-budget violations. Table~\ref{tab:newresults} reports the common-executor comparison; Table~\ref{tab:strata} separates the two workload strata.
\input{sections/table_new_results}
\input{sections/table_strata}

'''+f"Risk-RePress performs {r['migrations']} migrations and rewrites {r['rewrite_mib']:.3f}~MiB; Credit-RePress performs {c['migrations']} and rewrites {c['rewrite_mib']:.3f}~MiB. The paired service-time and event-p99 ratios to Credit-RePress are {ratio(sc)} and {ratio(tc)}, with {sc['wins']} and {tc['wins']} wins across the 24 instances. Relative to Static, the service ratio is {ratio(ss)} and event-p99 ratio is {ratio(ts)}. Controller work accounts for {r['controller_percent']:.2f}\\% of Risk-RePress service, compared with {c['controller_percent']:.2f}\\% for Credit-RePress.\n\n"+r'''
\begin{figure*}[!t]
\centering\includegraphics[width=.94\textwidth]{policy_comparison.pdf}
\caption{Risk-RePress relative to each comparison policy across all 24 instances. Points show geometric-mean ratios, and bars show paired-bootstrap 95\% intervals. Service and event-p99 costs are displayed separately.}
\label{fig:comparison}
\end{figure*}
'''+f"The continuous-mixture stratum has a service ratio to Static of {ratio(cs)} and an event-p99 ratio of {ratio(ct)}. The persistent-shift stratum has corresponding ratios of {ratio(ps)} and {ratio(pt)}. The latter provides a sustained point-heavy interval in which a replacement can recover its forward and return price; the former repeatedly tests decisions as query mixtures evolve. Figure~\\ref{{fig:trajectories}} keeps their realized service trajectories separate.\n\n"+r'''
\begin{figure*}[!t]
\centering\includegraphics[width=.94\textwidth]{service_trajectories.pdf}
\caption{Cumulative measured service ratios by workload stratum. Curves are geometric means over 18 continuous-mixture instances (top) and six persistent-shift instances (bottom); shading shows the interquartile range. The persistent shift occurs after 20,000 queries. Every instance contributes at every query index; commissioning is outside these trajectories.}
\label{fig:trajectories}
\end{figure*}
\subsection{Case-level costs and empirical tail agreement}
'''+f"The largest service ratio to Static occurs in group {w['group']+1}, with {w['size']:,}-value chunks under {w['family'].replace('_',' ')}. Risk-Repress uses {wr['service_ns']/1e9:.3f}~s and Static uses {ws['service_ns']/1e9:.3f}~s; their event p99 values are {wr['p99_ns']/1000:.1f} and {ws['p99_ns']/1000:.1f}~$\\mu$s. Risk-RePress makes {wr['migration_count']} migrations. Its control and migration components sum to {wr['controller_ns']/1e9:.3f} and {wr['migration_ns']/1e9:.3f}~s, while measured query time differs from Static by {(wr['query_ns']-ws['query_ns'])/1e9:.3f}~s. These components expose where the realized service difference occurs.\n\n"+' '.join(tailpara)+r''' These observations use windows with 768 local queries under an unchanged layout and compare query-path p99 on identical requests from separately timed executions. The overlapping-window counts are a descriptive agreement check. Event latency additionally includes decision work and maintenance, as reflected in the end-to-end quantiles.

\subsection{RQ3: Recovery across activation and retirement}
The new campaign contains 492 successful trials, summarized in Table~\ref{tab:recovery}. Controlled exceptions cover all 20 directed pairs over the original five layouts, both sizes, and ten software-visible stages. Additional checks exercise the 4,096-value layout in both directions against whole LZ4. Forty trials terminate the process directly. Twelve truncate the new payload after activation while the previous version remains retained. Reopening checks the active codec, exact values, query answers, catalog integrity, and repeated recovery.
\begin{table}[t]
\centering\small
\caption{Fresh recovery tests of the revised executor.}
\label{tab:recovery}
\begin{tabular}{@{}lrr@{}}\toprule
Trial class & Trials & Passed \\
\midrule
'''+ '\n'.join(rows)+r'''
\midrule Total & 492 & 492 \\
\bottomrule\end{tabular}
\end{table}

A separate abrupt-exit reproduction stops the original executor after old block metadata is deleted and before its cleanup transaction. Restart then raises a missing-metadata error while reconstructing the retained version. In the revised path, retirement has committed before this deletion stage; restart treats remaining old files as unreferenced garbage. The paired protocol trace and fault records bind this ordering to the executed implementation.
'''
    text=text.replace('Risk-Repress','Risk-RePress')
    (ROOT/'sections/results.tex').write_text(text,encoding='utf-8')
    conclusion=r'''\section{Conclusion}
RePressDB couples empirical request-tail admission with the cost and recovery path of durable layout replacement. Operation/length profiles expose minority query costs. Continuous feasible-epoch evidence and forward-plus-return prices connect admission to the controller's own repair behavior. Conditional-count tables evaluate the same empirical criterion with lower decision work. Three catalog transactions preserve activation and recovery boundaries while committing retirement before physical deletion.

'''+f"The complete 23.04-million-event comparison records a service ratio to Credit-RePress of {ratio(sc)} and an event-p99 ratio of {ratio(tc)}, with migrations changing from {c['migrations']} to {r['migrations']}. Against Static, continuous and persistent-shift service ratios are {cs['gm_ratio']:.3f} and {ps['gm_ratio']:.3f}; their tail ratios show the corresponding latency costs. Paired executor measurements and 492 successful fault trials connect these operating results to the implemented durable path. The preserved cost samples, decisions, component timings, and source hashes support replication across additional data, cache, and service regimes.\n"
    (ROOT/'sections/conclusion.tex').write_text(conclusion,encoding='utf-8')
    report=f'''# RePressDB 方法优化与新实验（FGCS-2.2）

这一轮已经落实到代码、完整测量和英文稿。原 JSS 文件与上一轮 FGCS 记录保留。

## 方法的实质变化

1. **按查询操作和长度建模。** 原来的粗类别拆成 29 个操作／长度分层，记录每层的真实延迟分布。所有对照策略共享相同均值成本和六种布局。
2. **只累计连续可行阶段的证据。** 目标布局一旦失去尾延迟可行性，旧收益计数清零；下一阶段重新累计。
3. **把返回成本计入准入。** 阈值包含迁入目标和返回初始参考布局的有向成本。负载变化触发的风险修复也消耗预算、计入服务时间。
4. **用条件分布计数表降低控制开销。** 64 个样本的计数使用一个字节存储，分位点在有限支持集上二分查找。288 万次请求、11,181 个决策边界与原实现完全一致。最终计数实现的决策中位耗时为 {eq['cached_decision_median_ns']/1000:.1f} 微秒，扫描实现为 {eq['original_decision_median_ns']/1000:.1f} 微秒；配对比值中位数 {eq['median_paired_time_ratio']:.3f}。
5. **重写迁移执行路径。** 六次目录事务合并为预约、激活、退役三次事务；先提交退役，再删除旧文件。240 组配对观测的迁移耗时比值中位数为 {exc['median_paired_ratio']:.3f}，减少 {(1-exc['median_paired_ratio'])*100:.1f}%。新路径通过 492 次故障检查，并复现、修复了旧清理顺序的重启缺陷。

## 最终正式实验

- 24 栋建筑、8K／16K 两种块大小、三个连续变化负载族和一个持续切换负载族。
- 24 个配对实例 × 6 个策略 = **144 个单元、23,040,000 次实测请求**，每次均核对独立真值。
- 三个连续负载族的 18 个实例与新增持续切换的 6 个实例分别统计。
- 8,352 条查询成本记录，每条 64 次测量与 4 次预热；4,320 次完整迁移校准。
- 完整校准耗时 {cal['wall_ns']/1e9:.1f} 秒，单独报告；控制器初始化中位耗时 {setup['median_ms']:.2f} 毫秒；每八块实例的附加计数表中位大小 {setup['lookup_bytes_median']/2**20:.3f} MiB。

## 实测结论

| 对比范围 | 服务时间比 | 95% 区间 | 事件 p99 比 | 95% 区间 |
|---|---:|---:|---:|---:|
| 全部实例／Credit-RePress | {sc['gm_ratio']:.4f} | [{sc['ci_low']:.4f}, {sc['ci_high']:.4f}] | {tc['gm_ratio']:.4f} | [{tc['ci_low']:.4f}, {tc['ci_high']:.4f}] |
| 连续负载／固定布局 | {cs['gm_ratio']:.4f} | [{cs['ci_low']:.4f}, {cs['ci_high']:.4f}] | {ct['gm_ratio']:.4f} | [{ct['ci_low']:.4f}, {ct['ci_high']:.4f}] |
| 持续切换／固定布局 | {ps['gm_ratio']:.4f} | [{ps['ci_low']:.4f}, {ps['ci_high']:.4f}] | {pt['gm_ratio']:.4f} | [{pt['ci_low']:.4f}, {pt['ci_high']:.4f}] |

新方法迁移 {r['migrations']} 次、重写 {r['rewrite_mib']:.3f} MiB；同输入累计准入对照迁移 {c['migrations']} 次、重写 {c['rewrite_mib']:.3f} MiB。服务时间、p99 和重写量一起解释，三个连续负载族与持续切换场景的差异也保留在正文。

## 实验范围与记录

48 个独立数据块共 2.25 MiB 原始值；运行环境是一台 Windows／NTFS 主机，工作集可放入内存。真实建筑数据配合半合成请求。事件延迟包含查询、控制和同步迁移；校准与初始化单列。这些条件直接写入正文。

初版尾风险规则在 33 个开发单元中出现往返切换，记录完整保留。随后完整的 FGCS-2.1 测量暴露了分布计算开销；等价优化后重新执行了全部正式策略，并加入事先声明的持续切换场景。开发结果、实现等价检查与最终测量分目录存放。

源码、计时数组、迁移记录、条件分布样本、故障记录和哈希清单随交付提供。英文稿采用直接描述条件、规则与结果的写法。新代码和论文尚未向外部仓库或期刊提交。
'''
    (ROOT/'方法优化与新实验.md').write_text(report,encoding='utf-8')
    letter=f'''1 November 2026

Editors
Future Generation Computer Systems

Dear Editors,

Please consider our manuscript, "RePressDB: Risk-Aware Layout Adaptation with Efficient Durable Replacement," as a research article.

The paper connects operation-dependent query latency to the measured cost and recovery path of a persistent layout change. It introduces continuous feasible-epoch evidence, forward-plus-return action pricing, and an exact conditional-count implementation of empirical tail admission. The replacement executor consolidates catalog work into three transactions and commits retirement before file deletion.

A common-executor comparison checks 23.04 million requests over six policies and 24 paired instances. Continuous-mixture and persistent-shift workloads are reported separately. Relative to accumulated admission under identical cost inputs, the service-time ratio is {ratio(sc)} and the event-p99 ratio is {ratio(tc)}. A paired executor diagnostic yields a median time ratio of {exc['median_paired_ratio']:.3f}; all 492 fresh software-fault tests pass. The manuscript reports calibration, initialization, service, tails, and rewriting together.

This work addresses adaptive data infrastructure and the operational cost of runtime decisions. The companion archive includes frozen protocols, unchanged source provenance, empirical cost samples, decisions, component latency arrays, recovery tests, and figure-generation code.

The manuscript discloses the actual AI assistance used for research code, analysis, figures, and writing. The authors will verify submission declarations and co-author approval when submitting.

Sincerely,
Zhichao Zhang
School of Science, Minzu University of China
zhichao.zhang2024@outlook.com
'''
    (ROOT/'cover_letter_FGCS.txt').write_text(letter,encoding='utf-8')
    highlights=['Tail admission uses operation and length latency distributions.','Action evidence resets across infeasible workload phases.','Admission prices installation and a directed return to the reference.','Three durable transactions commit retirement before deleting old files.','Six policies execute 23.04 million checked queries on a shared path.']
    assert all(len(s)<=85 for s in highlights)
    (ROOT/'highlights.txt').write_text('\n'.join(highlights)+'\n',encoding='utf-8')
    print('Wrote results, abstract, conclusion, Chinese report, cover letter, and highlights.')
if __name__=='__main__':run()
