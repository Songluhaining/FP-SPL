"""从已核验的归档生成新增双系统中文报告，不读取训练中的临时分数。"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .case_study import ROOT
from .extension_report import table

SYSTEMS={"gpl":"图处理（GPL）","zipme":"压缩（ZipMe）"}
METHODS={"statement":"贝叶斯诊断","CLAP":"有监督分类（CLAP）","PULP":"正例与未标注学习（PULP）"}


def performance(data,protocol="fivefold",subset="all"):
    rows=[]
    for system in SYSTEMS:
        for method in METHODS:
            selected=data[(data.system==system)&(data.method==method)&(data.protocol==protocol)&(data.subset==subset)]
            if len(selected)!=1:raise ValueError(f"指标不完整 {system}/{method}/{protocol}/{subset}")
            r=selected.iloc[0]
            rows.append([SYSTEMS[system],METHODS[method],f"{r.scored}/{r.n}",r.insufficient,
                         f"{r.tp}/{r.fp}/{r.tn}/{r.fn}",
                         *[f"{r[k]:.2%}" for k in ("precision","recall","f1","accuracy","average_precision")],
                         f"{int(r.hits)}/{r.budget}"])
    return table(["系统","方法","评分数/候选数","证据不足","检出/误报/正确排除/漏报",
                  "精确率","召回率","调和平均值","准确率","平均精确率","检查命中/名额"],rows)


def make(root):
    frames={}
    for scenario in ("main","masked"):
        folder=root/scenario
        receipt=json.loads((folder/"registry-verified.json").read_text(encoding="utf8"))
        for name in ("metrics","predictions"):
            if hashlib.sha256((folder/(name+".csv")).read_bytes()).hexdigest()!=receipt[name+"_sha256"]:
                raise ValueError("结果与独立核验凭据不一致")
        frames[scenario]=pd.read_csv(folder/"metrics.csv")
    main,masked=frames["main"],frames["masked"]
    versions=pd.read_csv(root/"main/versions.csv")
    truth=pd.read_csv(root/"main/truth.csv")
    construction=pd.read_csv(root/"masked/construction.csv")
    replay=pd.read_csv(root/"main/replay-summary.csv")
    ci=pd.read_csv(root/"main/paired-differences.csv")
    conclusions=[]
    for system in SYSTEMS:
        part=main[(main.system==system)&(main.protocol=="fivefold")&(main.subset=="all")].set_index("method")
        new=part.loc["statement","f1"]
        changes=[f"相对{METHODS[m]}{'提高' if new>=part.loc[m,'f1'] else '降低'}{abs(new-part.loc[m,'f1'])*100:.2f}个百分点"
                 for m in ("CLAP","PULP")]
        conclusions.append(f"{SYSTEMS[system]}的贝叶斯诊断调和平均值为{new:.2%}，"+"，".join(changes)+"。")
    raw=["# 图处理与压缩系统单缺陷比较实验",
         "本次比较覆盖全部160个单缺陷版本、11770个配置，以其中8709个原本通过配置作为主实验对象。"+"".join(conclusions),
         "## 主实验范围与结果",
         table(["系统","版本","全部配置","原本通过","基准阳性","逐用例覆盖文件"],
               [[SYSTEMS[s],len(g),int(g.configurations.sum()),int(g.candidates.sum()),int(g.positives.sum()),int(g.tests.sum())]
                for s,g in versions.groupby("system")]),
         "以下为相对基准标签的识别结果。两个系统分别按缺陷版本固定五折，每个候选恰有一次折外预测，三种方法使用相同候选分母和可见运行证据。",
         performance(main),
         "有监督分类仅使用训练折的真假通过标签。正例与未标注学习调用原装袋学习入口，以训练折的已知失败配置为正例，其余配置保持未标注。贝叶斯诊断使用目标版本内的可见成败与覆盖，沿用既定参数及0.5阈值，没有用新增系统标签调参。",
         "复现保留原特征公式与分类器，兼容层处理Windows路径、实验输出、十组频谱权重接口及折外分数导出。有监督方法的6列输入、正例与未标注方法的50列输入，以及各折训练样本、抽样比例和源码摘要均已归档。",
         "混淆矩阵按检出、误报、正确排除、漏报顺序展示。精确率、召回率等比例的零分母沿用前次实验记零规则；证据不足的配置保留在候选分母中。排序检查预算为每个版本候选数的十分之一向上取整，无分配置不取得命中。",
         "## 标签口径核对",
         "基准阳性依据变异日志中的缺陷语句与产品中的特征源码身份匹配，不要求执行次数为正。下表对同一批预测另按缺陷行的执行或分支计数为正评价，不改变训练标签或重新训练，避免把编入关系直接解释为已复现的规格违例。",
         table(["系统","基准阳性","缺陷行执行过的阳性","基础特征版本"],
               [[SYSTEMS[s],int(g.label.sum()),int(g.executed_label.sum()),g.loc[g.base==1,"version"].nunique()]
                for s,g in truth.groupby("system")]),
         performance(main,subset="executed"),
         "## 跨系统预测",
         "两种基线分别进行图处理训练后预测压缩、压缩训练后预测图处理。贝叶斯诊断不需要跨系统拟合，沿用同一目标版本的推断结果。",
         performance(main,"transfer"),
         "再用此前银行账户、邮件、电梯和考试数据库共178个版本训练两种基线，预测这两个新增系统。",
         performance(main,"external"),
         "## 移除失败用例的补充实验",
         "在每个版本按配置名排序的失败配置中，每隔一个选择一个作转换，保留首个失败配置作参照。实际移除被选配置的失败测试及对应覆盖，在独立环境重新编译并执行留存测试，再从剩余证据重建三种方法的输入。",
         f"其中{int(construction.get('environment_retry',pd.Series(dtype=float)).sum())}个配置在初次运行出现环境错误后，对隔离编译副本中的临时目录、工作目录和用户目录作本机映射后通过。原测试断言、产品源码及报告不变；真实案例核验确认映射仍保留原有失败，详见environment-verification.json，各次映射与初次错误见补充目录construction-environment.json。",
         table(["系统","原失败配置","实际转换","保留失败参照","无可留存测试","其他失败","移除测试","重跑通过测试"],
               [[SYSTEMS[s],len(g),int((g.status=="passed").sum()),int((g.status=="retained_failure").sum()),
                 int((g.status=="ineligible").sum()),int((~g.status.isin(["passed","retained_failure","ineligible"])).sum()),
                 int(g.removed.sum()),int(g.remaining.sum())] for s,g in construction.groupby("system")]),
         "下表包含原本通过与实际转换的配置；只看实际转换样本的结果见补充目录指标表的转换样本子集。",
         performance(masked),
         "## 逐用例证据的作用",
         table(["系统","仅用配置编入关系的调和平均值","加入逐用例证据的调和平均值","比较方法","差值的95%配对区间"],
               [[SYSTEMS[r.system],
                 f"{main[(main.system==r.system)&(main.method=='configuration')&(main.protocol=='fivefold')&(main.subset=='all')].iloc[0].f1:.2%}",
                 f"{main[(main.system==r.system)&(main.method=='statement')&(main.protocol=='fivefold')&(main.subset=='all')].iloc[0].f1:.2%}",
                 METHODS[r.baseline],f"{100*r.lower:+.2f}至{100*r.upper:+.2f}个百分点"] for r in ci.itertuples()]),
         "区间按系统内缺陷版本配对、有放回重采样一万次得到。区间跨零时，不能仅据点估计认定稳定提升。配置编入关系与基准标签定义密切相关，需结合执行标签和实际补测理解结果。",
         "## 实际补测",
         "三种方法共享与预测分数及标签无关的固定候选池和选例顺序。候选测试须在同特征参考产品上通过，且在目标配置的两次独立运行中触发同一断言违例，才记为确认；编译失败、超时和运行异常不计为确认。",
         table(["系统","每配置预算","有候选配置","实际候选测试","确认配置","编译失败","超时","运行异常","参考断言不通过",
                "贝叶斯报出","有监督报出","正例与未标注报出"],
               [[SYSTEMS[s],b,int(g.iloc[0].with_tests),int(g.iloc[0].planned_tests),int(g.iloc[0].confirmed),
                 int(g.iloc[0].compile_errors),int(g.iloc[0].timeouts),int(g.iloc[0].runtime_errors),int(g.iloc[0].reference_assertion_failures),
                 *[int(g.loc[g.method==m,"detected"].iloc[0]) for m in METHODS]] for (s,b),g in replay.groupby(["system","budget"])]),
         "各方法报出数是其预测阳性与已确认配置的交集，不是不同选例策略的实验。补测存在大量跨配置测试编译失败，实际确认数受测试可移植性限制；未确认配置不能据此当作真通过。各预算行的配置可重复出现。",
         "运行异常包括参考产品和目标产品两侧的异常；参考产品断言不通过另列。每个预算下全部候选测试的互斥状态计数见replay-status-counts.csv，其合计必须等于实际候选测试数。补测仍按原环境记录报告，未把补充实验的环境重试结果混入补测确认数。",
         "图处理参考源码来自[官方示例的固定版本](https://github.com/FeatureIDE/FeatureIDE/tree/d769691754fd63cf757902fc1b3daf28d9333535/plugins/de.ovgu.featureide.examples/featureide_examples/GPL-FH-Java)，57份源码均已与未受相应变异影响的本地文件核对。参考源码缺失是共享模板问题，原105个案例的产品源码及测试完整。",
         "## 保存位置与复现",
         "六系统统一表保存为上级目录的“六系统单缺陷实验比较.csv”。主实验和补充实验分别保存逐配置预测、指标、评价标签、训练划分、输入特征、源码摘要及独立核验凭据。所有派生数据均位于原案例内的.fp_research目录或同系统的共享缓存，原源码、测试、覆盖和结果报告保持不变。",
         "两种基线分别在两个场景重训一次，同一随机种子的预测必须逐字节相同。独立统计脚本核对配置无遗漏、无重复、三种方法分母一致，以及训练和评估版本无交叉。等价加速的整表及切片核验见acceleration-verification.json。",
         "```powershell\n$env:PYTHONUTF8='1'\n$env:OPENBLAS_NUM_THREADS='1'\n$env:FP_SLICE_BACKEND='indexed'\n$env:FP_SLICE_WORKERS='4'\n$env:FP_SLICE_HEAP_MB='1536'\npython -m fp_research.case_study run --workers 4 --pending-only\npython -m fp_research.case_replay --workers 4\npython -m fp_research.case_finish --include-masked\npython -m fp_research.checks.verify_extension experimental_results/gpl_zipme\npython -m fp_research.case_report\npython -m fp_research.result_registry\n```"]
    (root/"实验报告.md").write_text("\n\n".join(raw)+"\n",encoding="utf8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"experimental_results/gpl_zipme")
    make(parser.parse_args().results)
