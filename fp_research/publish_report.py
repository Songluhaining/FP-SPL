"""将已通过逐行审计的实验表格排成中文研究报告。"""
import argparse
from pathlib import Path
import pandas as pd

NAMES={"full":"新方法","CLAP":"有监督分类（CLAP）","PULP":"正例与未标注学习（PULP）",
       "frequency":"子集频率","no_assertion":"检出网络移除断言输入","no_config":"移除配置适用条件"}
SYS={"bank":"银行账户","email":"邮件"}


def table(header,rows):
    return "\n".join(["| "+" | ".join(header)+" |","|"+ "|".join(["---"]*len(header))+"|",
                      *["| "+" | ".join(map(str,r))+" |" for r in rows]])


def pct(x):
    return f"{100*x:.2f}%"


def ordered(t):
    return t.assign(_system=t.system.map({"bank":0,"email":1}),_method=t.method.map(
        {"full":0,"CLAP":1,"PULP":2,"frequency":3,"no_assertion":4,"no_config":5})).sort_values(["_system","_method"])


def performance(t):
    entries=list(ordered(t).itertuples())
    quality=table(["评估系统","方法","精确率","召回率","调和平均值","准确率","平均精确率"],
                  [(SYS[r.system],NAMES[r.method],pct(r.precision),pct(r.recall),pct(r.f1),
                    pct(r.accuracy),pct(r.average_precision)) for r in entries])
    counts=table(["系统","方法","评分/候选","部分证据¹","证据不足","检出","误报","正确排除","漏报","等额检查命中"],
                 [(SYS[r.system],NAMES[r.method],f"{r.scored}/{r.n}",r.partial,r.insufficient,
                   r.tp,r.fp,r.tn,r.fn,f"{r.hits}/{r.budget}") for r in entries])
    return quality+"\n\n"+counts


def generate(root,paired=None):
    main=pd.read_csv(root/"main/metrics.csv")
    masked=pd.read_csv(root/"masked/metrics.csv")
    frame=pd.read_csv(root/"main/predictions.csv")
    primary=frame[(frame.method=="full") & frame.split.str.contains("-fold")]
    budgets=pd.read_csv(root/"main/replay-summary.csv")
    versions=pd.read_csv(root/"versions.csv")
    summary=main[(main.subset=="all")&~main.transfer&main.method.isin(("full","CLAP","PULP"))]
    result=[]
    for s,t in summary.groupby("system"):
        best=t.loc[t.f1.idxmax()]
        own=t[t.method=="full"].iloc[0]
        result.append(f"{SYS[s]}系统中新方法的调和平均值为 {pct(own.f1)}，该系统最高值为{NAMES[best.method]}的 {pct(best.f1)}。")
    docs=["# 两个软件产品线的假通过识别实验","".join(result),
          "## 主实验","主实验评价原始测试全部通过的配置。固定按缺陷版本划分五折，每个配置在其版本留出时预测一次。基准阳性标签来自目标构建覆盖表与真实缺陷语句的匹配。",
          performance(summary),"¹ 成功评分包含仅部分失效组可用的配置。完全无证据的配置保留在分母，按未报出计。等额检查为每个版本检查排名最高的十分之一配置并向上取整，并列分数按配置标识排序。",
          "## 跨系统迁移","表中的系统是测试系统，另一个系统提供训练版本。目标版本的可见失败记录可用于计算运行特征。目标通过配置的真假标签只在评价时读取。",
          performance(main[(main.subset=="all")&main.transfer&main.method.isin(("full","CLAP","PULP"))]),
          "## 哪一部分改变了排序","集合模型使用操作、输入及断言的匹配特征，根据历史测试子集能否揭露同一失效进行训练。下表复用五折划分和同一批配置。",
          table(["系统","方法","精确率","召回率","调和平均值","平均精确率"],
                [(SYS[r.system],NAMES[r.method],pct(r.precision),pct(r.recall),pct(r.f1),pct(r.average_precision))
                 for r in ordered(main[(main.subset=="all")&~main.transfer&~main.method.isin(("CLAP","PULP"))]).itertuples()])]
    if paired and paired.is_file():
        pairs=pd.read_csv(paired)
        notes=[]
        for s in SYS:
            selected=pairs[(pairs.system==s)&pairs.split.str.contains("-fold")]
            n=len(selected)
            good=int((selected.p_detected>selected.p_missed).sum())
            notes.append(f"{SYS[s]}有 {n} 对，其中 {good} 对的预测方向与实际检出变化一致。" if n else f"{SYS[s]}没有满足这些条件的对照对，该系统无法用此检查验证机制。")
        docs.append("另外在留出版本中固定配置、用例数量与公共用例，比较操作相同、覆盖交并比至少为 0.8 且断言不同的两组测试。"+"".join(notes))
    docs.extend(["## 补测复核","实际补测采用系统内留出模型。每个配置使用相同的可移植候选池，分别按模型检出增益、随机顺序和新增可覆盖语句选择测试。用例先在相同特征的参考产品上运行。确认要求它在参考产品上通过，在目标产品的两次独立执行中发生相同断言错误。多种策略选中同一用例时共享实测回执。各行从第一条起累计，候选池较小时实际选例数会少于名额。",
          table(["系统","选择方式","每配置上限","有候选的配置","实选用例","确认配置","编译失败","参考执行次数","目标执行次数","超时"],
                [(SYS[r.system],{"model":"模型增益","random":"随机","coverage":"覆盖优先"}[r.strategy],r.budget,
                  r.eligible,r.selected,r.confirmed,r.compile_errors,r.reference_checks,r.target_checks,r.timeouts) for r in budgets.itertuples()]),
          "## 真实删减测试的补充实验","另将银行 377 个、邮件 223 个原失败配置的失败用例及覆盖移除，重跑其全部剩余测试。此表包含原本通过和实际转换的配置，三种方法重新计算可见特征。仅转换样本的结果另存于指标表，其标签均为阳性。",
          performance(masked[(masked.subset=="all")&~masked.transfer&masked.method.isin(("full","CLAP","PULP"))]),
          "## 数据、失败证据与解释边界"])
    notes=[]
    for s in SYS:
        part=primary[primary.system==s]
        n=len(part)
        pos=int(part.label.sum())
        hit=int(part.executed_label.sum())
        stage=pd.read_csv(root/"main/stage-checks.csv")
        stage=stage[stage.system==s]
        notes.append(f"- {SYS[s]}从 {int(versions[versions.system==s].configurations.sum())} 个配置中得到 {n} 个原本通过配置。基准阳性为 {pos} 个，要求故障语句实际执行后为 {hit} 个。{int(stage.failures.sum())} 个失败用例中有 {int(stage.paired.sum())} 个取得符合规则的通过对照。{int(stage.groups.sum())} 个失效组中有 {int(stage.scored_groups.sum())} 个产生有效后验。")
    docs.append("\n".join(notes))
    docs.append("对照用例要求观察相同的输出。若框架抛出非法参数异常的测试替身，重放抛出其对应的标准异常，只有核实继承关系且源码位置相同才接受。普通执行与插桩成败仍有差异的失效组逐组列于未评分清单（unavailable-groups.csv）。")
    validation=[]
    for s in SYS:
        row=budgets[(budgets.system==s)&(budgets.budget==5)].set_index("strategy")
        validation.append(f"{SYS[s]}最多补测五条时，模型选例确认 {int(row.loc['model','confirmed'])} 个配置，"
                          f"随机选例确认 {int(row.loc['random','confirmed'])} 个，覆盖优先确认 {int(row.loc['coverage','confirmed'])} 个。")
    docs.append("".join(validation))
    validation=[]
    for s in SYS:
        rows=main[(main.system==s)&(~main.transfer)&(main.subset=="all")].set_index("method")
        validation.append(f"{SYS[s]}中，集合网络与子集频率估计的调和平均值分别为 {pct(rows.loc['full','f1'])} 和 {pct(rows.loc['frequency','f1'])}。")
    docs.append("".join(validation))
    cell=main[(~main.transfer)&main.method.isin(("full","CLAP","PULP"))].set_index(["system","method","subset"])
    subgroup=[]
    for system in SYS:
        for method in ("full","CLAP","PULP"):
            base=cell.loc[(system,method,"base")]
            other=cell.loc[(system,method,"other")]
            subgroup.append((SYS[system],NAMES[method],int(base.n),pct(base.recall),int(other.n),pct(other.f1)))
    docs.extend(["基础特征缺陷版本单列召回率，其他版本另列调和平均值。",table(
        ["系统","方法","基础特征组配置","该组召回率","其他配置","其他组调和平均值"],subgroup)])
    docs.extend(["邮件系统的 17 个标签随执行次数口径改变，同一批预测在两种口径下的结果如下。",
        table(["方法","原基准调和平均值","要求正执行次数的调和平均值"],
            [(NAMES[m],pct(cell.loc[("email",m,"all")].f1),pct(cell.loc[("email",m,"positive_execution")].f1))
             for m in ("full","CLAP","PULP")])])
    docs.extend(["三种方法按预先确定的规则判定。新方法用风险分数 0.5 为阈值，有监督基准沿用线性分类器的决策边界，正例与未标注基准采用原装袋模型的加权概率阈值。排名还以平均精确率评价。",
                 "原始失败记录与当前重放偶有不一致，插桩超时或改变观察到的成败时，该组不产生后验。可疑语句权重和与漏检估计构成排序分数。对照标签衡量基准所定义的可疑缺陷存在，补测确认衡量实际可重复的规格违例。",
                 "## 复算材料","同目录的版本清单（versions.csv）记录原始报告和产品源码摘要。补充构造清单（construction-checks.csv）列出每个转换配置移除及留存的测试数。两个实验子目录中的逐配置表（predictions.csv）、指标表（metrics.csv）、逐组证据（groups.jsonl.gz）、阶段核验（stage-checks.csv）和版本划分（folds.csv）可用于独立复算。主实验还附逐用例复核结果（replay-outcomes.csv）、逐配置名额明细（probe-budgets.csv）及独立重算的逐用例检出增益（probe-evidence.csv）。运行命令见仓库内的实验说明（fp_research/README.md）。"])
    qa=root/"main/backend-check.csv"
    if qa.is_file():
        verified=pd.read_csv(qa)
        docs.append(f"主实验有 {int(verified.both_scored.sum())} 个组同时具备常规插桩与单方法插桩结果，"
                    f"语句集合不同的组有 {int(verified.different_statements.sum())} 个，"
                    f"可比语句后验的最大绝对差为 {verified.max_absolute_difference.max():g}。"
                    "两种基准在固定划分与种子下分别重新训练，主实验与补充实验的逐配置预测文件均与首次运行逐字节相同。")
    docs.extend(["## 比较方法来源",
                 "[有监督的假通过检测研究](https://doi.org/10.1016/j.infsof.2022.107080) 和[正例与未标注学习研究](https://conf.researchr.org/details/issta-2026/issta-2026-research-papers/121/Integrating-Multiple-Features-for-Weakly-Supervised-False-Passing-Products-Detection-) 提供比较方法。表中数字是本轮双系统统一协议下的实测值。"])
    (root/"实验报告.md").write_text("\n\n".join(docs)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--paired",type=Path)
    a=p.parse_args()
    generate(a.results,a.paired)
