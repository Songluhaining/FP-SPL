"""由归档的双系统结果生成中文报告。"""
import argparse
from pathlib import Path
import pandas as pd

SYS={"elevator":"电梯","examdb":"考试数据库"}
METHOD={"statement":"贝叶斯诊断","CLAP":"有监督分类（CLAP）","PULP":"正例与未标注学习（PULP）"}


def table(header, rows):
    return "\n".join(["| "+" | ".join(header)+" |","|"+"|".join(["---"]*len(header))+"|",
                      *["| "+" | ".join(str(x) for x in r)+" |" for r in rows]])


def performance(data,protocol):
    out=[]
    for s in SYS:
        for m in METHOD:
            d=data[(data.system==s)&(data.protocol==protocol)&(data.method==m)&(data.subset=="all")].iloc[0]
            out.append([SYS[s],METHOD[m],f"{d.scored}/{d.n}",d.insufficient,f"{d.tp}/{d.fp}/{d.tn}/{d.fn}",
                        *[f"{d[k]:.2%}" for k in ["precision","recall","f1","accuracy","average_precision"]],
                        f"{int(d.hits)}/{d.budget}"])
    return table(["系统","方法","评分/候选","无分","检出/误报/正确排除/漏报","精确率","召回率","调和平均值","准确率",
                  "平均精确率","等额检查命中/名额"],out)


def make(root):
    main=pd.read_csv(root/"main/metrics.csv")
    masked=pd.read_csv(root/"masked/metrics.csv")
    pred=pd.read_csv(root/"main/predictions.csv")
    counts=pd.read_csv(root/"main/versions.csv")
    ci=pd.read_csv(root/"main/paired-differences.csv")
    construction=pd.read_csv(root/"masked/construction.csv")
    groups=construction.groupby(["system","status"]).size()
    missing=pred[(pred.status=="insufficient")&(pred.protocol=="fivefold")]
    if set(missing.method)!={"CLAP"} or missing.system.unique().tolist()!=["examdb"]:
        raise ValueError("单类折与核验预期不符")
    def cell(data,s,m,sub="all"):
        return data[(data.protocol=="fivefold")&(data.system==s)&(data.method==m)&(data.subset==sub)].iloc[0]
    statements=[]
    for s in SYS:
        b=cell(main,s,"statement")
        best=max(["CLAP","PULP"],key=lambda m:cell(main,s,m).f1)
        baseline=cell(main,s,best)
        statements.append(f"{SYS[s]}系统中，贝叶斯诊断的调和平均值为 {b.f1:.2%}，{METHOD[best]}为 {baseline.f1:.2%}。")
    raw=[
        "# 电梯与考试数据库单缺陷版本的假通过识别实验",
        "".join(statements)+"考试数据库的主实验只有一个基准阳性，应结合误报数、跨系统预测和补测结果解读。",
        "## 原本通过配置的主实验",
        "三种方法在完全相同的原本通过配置上预测。基准阳性要求目标配置启用故障特征，且已知故障行出现在源码映射中。另按该行实际执行重新评价，两系统的主实验标签均与原口径相同。",
        table(["系统","缺陷版本","全部配置","原本通过","基准假通过","逐用例覆盖文件"],[
            [SYS[s],len(g),g.configurations.sum(),g.candidates.sum(),g.positives.sum(),g.tests.sum()]
            for s,g in counts.groupby("system")]),
        performance(main,"fivefold"),
        "各系统按缺陷版本做固定五折。有监督分类只用训练折的真假标签。正例与未标注学习以训练折的可见失败配置作正例，用原装袋实现训练 30 个分类器。贝叶斯诊断在目标版本内使用可见失败和覆盖，保持原先的 0.5 判定阈值。三种方法均可见目标版本的原始测试状态。",
        f"考试数据库唯一基准阳性为版本 225 的配置 0006。留出该版本时，有监督分类的训练集只含真通过，原版二分类器无法拟合。该折 {missing.version.nunique()} 个版本、{len(missing)} 个配置保留在总分母中，记作未报出并标记无分。完整评分的跨系统对照见下节。",
        "精确率表示报出者中的基准阳性比例，召回率表示找出的基准阳性比例，调和平均值综合两者。沿用前次实验的零分母记零规则。平均精确率按系统汇总排序，无分配置并列末位。等额检查按每个版本候选数的十分之一向上取整；无分配置不能取得命中。",
        "## 跨系统预测",
        "下表分别用考试数据库训练后预测电梯、用电梯训练后预测考试数据库。每个训练系统使用全部版本。",
        performance(main,"transfer"),
        "下表只在原银行账户和邮件系统的 109 个版本训练两种基线，再在两个新增系统预测。贝叶斯诊断无需跨系统拟合，各表使用同一目标版本的推断结果。",
        performance(main,"external"),
        "## 同一配置内移除失败用例的补充实验",
        "按配置名排序，在各版本的失败配置中每隔一个选择一个，始终保留首个失败配置作参照。实际移除选中配置的失败用例及覆盖，重新编译并执行留存测试，程序逐项核对留存测试非空、全部通过，且源码和覆盖中均无隐藏用例。",
        table(["系统","原失败配置","实际转换","保留作失败参照","不具备留存测试","移除用例","重跑通过用例"],[
            [SYS[s],len(g),groups.get((s,"passed"),0),groups.get((s,"retained_failure"),0),
             groups.get((s,"ineligible"),0),int(g.removed.sum()),int(g.remaining.sum())] for s,g in construction.groupby("system")]),
        "补充表包含原本通过和实际转换的配置。两种基线用删减后的数据重新训练，贝叶斯诊断从剩余可见证据重新推断。仅转换样本及其余划分的结果见补充指标明细。",
        performance(masked,"fivefold"),
        "## 逐用例证据与版本差异",
        table(["系统","仅用编入关系的调和平均值","加入逐用例命中与成败","相对基线","提升百分点及 95% 配对区间"],[
            [SYS[r.system],f"{cell(main,r.system,'configuration').f1:.2%}",f"{cell(main,r.system,'statement').f1:.2%}",METHOD[r.baseline],
             f"{100*(cell(main,r.system,'statement').f1-cell(main,r.system,r.baseline).f1):+.2f} （{100*r.lower:+.2f} 至 {100*r.upper:+.2f}）"] for r in ci.itertuples()]),
        "电梯系统加入逐用例命中与成败后，调和平均值略有下降。考试数据库两种设置得分相同。此次比较未显示逐用例证据带来额外识别收益。区间由系统内按缺陷版本配对、有放回重采样一万次得到。考试数据库的大量重采样没有基准阳性。配置编入关系与评价标签的定义密切相关。",
        "## 在同一批配置上实际补测",
        "对所有原本通过配置，优先从启用特征最相近的失败配置中选取该目标原有测试未包含的失败用例。三种识别方法共享选例顺序，与各自的分数及基准标签无关。用例在同特征的参考产品上通过、在目标上两次独立运行出现同一断言违例，才记为确认。"]
    replay=pd.read_csv(root/"main/replay-summary.csv")
    outcomes=pd.read_csv(root/"main/replay-outcomes.csv")
    replay_table=[]
    for (s,b),g in replay.groupby(["system","budget"]):
        r=g.iloc[0]
        replay_table.append([SYS[s],b,r.with_tests,r.planned_tests,r.compile_errors,r.confirmed,r.benchmark_negative,
                             *[int(g.loc[g.method==m,"detected"].iloc[0]) for m in METHOD]])
    raw.extend([table(["系统","每配置上限","有候选的配置","实选用例","编译失败","确认配置","其中基准阴性",
                       "贝叶斯报出","有监督报出","正例与未标注报出"],replay_table),
                "同一配置可在多个预算行重复出现。各方法的报出数只统计已确认配置中的交集。预算内未复现的配置仍可带有缺陷。逐次状态、断言信息与参考执行分列在补测明细。"])
    for s,g in outcomes.groupby("system"):
        raw.append(f"{SYS[s]}系统的 {len(g)} 条补测中，编译失败 {sum(g.status=='compile_error')} 条，超时 {sum(g.status.str.contains('timeout'))} 条，参考产品断言失败 {sum(g.status=='reference_assertion')} 条，目标运行异常 {sum(g.status=='runtime_error')} 条。")
    raw.extend(["## 复算与版本",
                "实现基于已冻结的单缺陷贝叶斯诊断版本 c0da9ce。源码分别来自本机的有监督分类和正例与未标注学习仓库。分类器输入是 6 维与 50 维，后者沿用抽样比例、随机种子 42 和原加权判定。故障位置与补测结果仅供评价。两个新增系统没有参与贝叶斯参数或阈值选择。",
                "在原仓库根目录使用下列命令。默认原始数据为 D:/dataset/whn，工作副本与全部派生缓存写入其中独立的 fp_extension_work 目录。补充实验重建切片时，只在源代码和可见汇总覆盖一致时复用主实验的切片。外部训练使用原双系统实验保留的特征和训练标签。",
                "```powershell\n$env:OPENBLAS_NUM_THREADS='1'\npython -m fp_research.extension mask --scenario masked\nforeach ($scenario in 'main','masked') {\n  foreach ($stage in 'observe','slices','CLAP','PULP') { python -m fp_research.extension $stage --scenario $scenario }\n  foreach ($stage in 'truth','CLAP','PULP','audit') { python -m fp_research.extension_evaluate $stage --scenario $scenario }\n}\npython -m fp_research.extension_replay\npython -m fp_research.extension_evaluate replay\npython -m fp_research.checks.verify_extension experimental_results/latent_extension\npython -m fp_research.extension_report experimental_results/latent_extension\n```",
                "两种基线均在主实验及补充实验重训一次，同种子两次生成的逐配置分数与判定逐字节一致。可用重训入口（python -m fp_research.extension_evaluate repeat --method PULP --scenario main）复核，替换方法与场景可验证其余三组。",
                "主实验与补充实验目录各含逐配置预测（predictions.csv）、评价标签（truth.csv）、指标（metrics.csv）、训练与留出版本（folds.csv）及覆盖核验（versions.csv）。另附两种基线的输入特征和源码摘要。补充构造清单（construction.csv）包含保留与转换的全部原失败配置。"])
    (root/"实验报告.md").write_text("\n\n".join(raw)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    make(p.parse_args().results)
