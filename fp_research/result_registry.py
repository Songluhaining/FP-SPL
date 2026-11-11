"""统一保存六系统实验结果；从逐配置预测复核统计，未完成结果保留空值。"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

import pandas as pd

from .case_study import ROOT,SYSTEMS,case_versions,results

NAMES={"bank":"银行账户（BankAccountTP）","email":"邮件（Email）",
       "elevator":"电梯（Elevator）","examdb":"考试数据库（ExamDB）",
       "gpl":"图处理（GPL）","zipme":"压缩（ZipMe）"}
METHODS={"statement":"贝叶斯诊断","CLAP":"有监督分类（CLAP）",
         "PULP":"正例与未标注学习（PULP）"}
SCENARIOS={"main":"原本通过配置","masked":"移除失败用例后的补充实验"}
PROTOCOLS={"fivefold":"按缺陷版本五折","transfer":"两个系统互相迁移","external":"此前系统训练后迁移"}
SUBSETS={"all":"全部","base":"基础特征缺陷","other":"其他特征缺陷",
         "executed":"缺陷行实际执行标签","positive_execution":"缺陷行实际执行标签","converted":"仅转换配置"}
KEYS=["system","version","configuration"]
METRICS={"n":"候选配置数","scored":"成功评分数","insufficient":"证据不足数",
         "tp":"检出数","fp":"误报数","tn":"正确排除数","fn":"漏报数",
         "precision":"精确率","recall":"召回率","f1":"调和平均值","accuracy":"准确率",
         "average_precision":"平均精确率","hits":"等额检查命中数","budget":"等额检查名额"}
COLUMNS=["系统","方法","实验场景","版本划分","评价子集","完成状态","总缺陷版本数","已完成准备版本数",
         "全部配置数",*METRICS.values(),"原始通过配置数","主实验观察核验完成版本数","主实验补测完成版本数",
         "结果来源","预测明细来源","来源文件摘要","结果归档提交","更新时间","说明"]


def write_csv(path,rows,columns=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def commit(path):
    return subprocess.check_output(["git","-c",f"safe.directory={ROOT.as_posix()}","log","-1","--format=%H","--",str(path)],
                                   cwd=ROOT,text=True).strip()


def check(frame,table):
    if frame.duplicated(KEYS).any():raise ValueError("逐配置预测重复")
    label="executed_label" if table.get("subset") in {"executed","positive_execution"} else "label"
    if label not in frame or frame[label].isna().any():raise ValueError("评价标签缺失")
    y=frame[label].astype(int)
    prediction=frame.prediction.astype(int)
    tp=int(((y==1)&(prediction==1)).sum()); fp=int(((y==0)&(prediction==1)).sum())
    tn=int(((y==0)&(prediction==0)).sum()); fn=int(((y==1)&(prediction==0)).sum())
    n=len(frame); scored=int(frame.score.notna().sum())
    expected=dict(n=n,scored=scored,insufficient=n-scored,tp=tp,fp=fp,tn=tn,fn=fn,
                  precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),accuracy=(tp+tn)/n,
                  hits=0,budget=0)
    for _,group in frame.groupby("version"):
        budget=math.ceil(len(group)/10)
        ranked=group[group.score.notna()].sort_values(["score","configuration"],ascending=[False,True]).head(budget)
        expected["budget"]+=budget
        expected["hits"]+=int(ranked[label].sum())
    for name,value in expected.items():
        if not math.isclose(value,float(table[name]),rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError(f"独立复算不一致 {name}: {value} != {table[name]}")


def inventory(data):
    counts={}; detail=[]
    old=pd.read_csv(ROOT/"experimental_results/fp_detection/versions.csv")
    more=pd.read_csv(ROOT/"experimental_results/latent_extension/main/versions.csv")
    for table,passed in [(old,"passed"),(more,"candidates")]:
        for system,part in table.groupby("system"):
            counts[system]=dict(versions=len(part),configurations=int(part.configurations.sum()),passed=int(part[passed].sum()),
                                observed=len(part),completed=len(part),replayed=None)
    for system in SYSTEMS:
        vs=case_versions(data,system)
        info=dict(versions=len(vs),configurations=0,passed=0,observed=0,completed=0,replayed=0)
        for version in vs:
            report=version/"config.report.csv.done"
            if not report.exists():report=version/"config.report.csv"
            with report.open(encoding="utf-8-sig",newline="") as stream:
                rows=list(csv.reader(stream))[1:]
            passed=sum(r[-1]=="__PASSED__" for r in rows)
            if any(r[-1] not in {"__PASSED__","__FAILED__"} for r in rows):raise ValueError("原始状态异常")
            main=results(version,"main")
            observed=(main/"observed.json").exists()
            completed=(main/"complete.json").exists()
            replayed=(main/"replay-complete.json").exists()
            failures=[]
            for name,finished in [("failure.json",completed),("replay-failure.json",replayed)]:
                if (main/name).exists() and not finished:
                    error=json.loads((main/name).read_text(encoding="utf8"))
                    reason=error["reason"]
                    if "inventory.json" in reason and "FileNotFoundError"==error["type"]:
                        reason="此前读取未生成的观察清单；执行顺序已修复，等待重跑完成。"
                    failures.append(error["type"]+": "+reason)
            info["configurations"]+=len(rows); info["passed"]+=passed
            for name,value in [("observed",observed),("completed",completed),("replayed",replayed)]:info[name]+=int(value)
            detail.append({"系统":NAMES[system],"版本":version.name,"全部配置数":len(rows),"原始通过配置数":passed,
                           "观察核验完成":int(observed),"三方法数据准备完成":int(completed),"补测完成":int(replayed),
                           "未完成记录":"；".join(failures),"结果目录":str(main)})
        counts[system]=info
    return counts,detail


def make(data,output):
    counts,progress=inventory(data)
    records=[]; sources={}; frames={}
    now=datetime.now().astimezone().isoformat(timespec="seconds")
    def read(path):
        if path not in frames:
            payload=path.read_bytes()
            sources[str(path)]=hashlib.sha256(payload).hexdigest()
            frames[path]=pd.read_csv(path,low_memory=False)
        return frames[path]
    def add(metrics_path,scenario=None,protocol=None,new=False):
        tables=read(metrics_path)
        version=commit(metrics_path)
        for item in tables.to_dict("records"):
            s,m=item["system"],item["method"]
            if m not in METHODS:continue
            scene=scenario or item["scenario"]
            setting=protocol or item.get("protocol","fivefold")
            subset=item.get("subset","all")
            if s in {"bank","email"}:
                prediction_path=ROOT/"experimental_results"/("latent_fault/predictions.csv" if m=="statement" else f"fp_detection/{scene}/predictions.csv")
            else:prediction_path=metrics_path.parent/"predictions.csv"
            frame=read(prediction_path)
            chosen=frame[(frame.system==s)&(frame.method==m)]
            if "scenario" in chosen:chosen=chosen[chosen.scenario==scene]
            if "protocol" in chosen:chosen=chosen[chosen.protocol==setting]
            elif m!="statement" or setting=="fivefold":
                chosen=chosen[chosen.split.str.contains("-fold" if setting=="fivefold" else "-to-")]
            if subset in {"base","other"}:chosen=chosen[chosen.base==int(subset=="base")]
            if subset=="converted":chosen=chosen[chosen.converted==1]
            check(chosen,item)
            if subset=="all" and scene=="main" and len(chosen)!=counts[s]["passed"]:
                raise ValueError("主实验分母与原始通过清单不符")
            note="相对基准标签的识别结果；比例取值范围为0至1；不代表实际补测全部确认。"
            if s=="examdb":note+="原本通过配置仅1个基准阳性；有监督分类五折含61个单类训练无法评分的配置，保留在总分母中。"
            if setting=="external":note+="训练来源为此前四系统。" if new else "训练来源为银行账户和邮件。"
            info=counts[s]
            row={"系统":NAMES[s],"方法":METHODS[m],"实验场景":SCENARIOS[scene],"版本划分":PROTOCOLS[setting],
                 "评价子集":SUBSETS[subset],"完成状态":"已完成并复算核验","总缺陷版本数":info["versions"],
                 "已完成准备版本数":info["versions"],"全部配置数":info["configurations"],
                 **{cn:item[key] for key,cn in METRICS.items()},"原始通过配置数":info["passed"],
                 "主实验观察核验完成版本数":info["observed"],"主实验补测完成版本数":info["replayed"],
                 "结果来源":str(metrics_path),"预测明细来源":str(prediction_path),"来源文件摘要":sources[str(metrics_path)],
                 "结果归档提交":version or "待归档提交","更新时间":now,"说明":note}
            records.append(row)
    add(ROOT/"experimental_results/latent_fault/metrics.csv")
    add(ROOT/"experimental_results/latent_fault/transfer-comparison.csv",protocol="transfer")
    for scene in SCENARIOS:
        add(ROOT/f"experimental_results/latent_extension/{scene}/metrics.csv",scenario=scene)
    for scene in SCENARIOS:
        path=ROOT/f"experimental_results/gpl_zipme/{scene}/metrics.csv"
        receipt=path.parent/"registry-verified.json"
        approved=False
        if path.exists() and receipt.exists():
            proof=json.loads(receipt.read_text(encoding="utf8"))
            approved=(proof.get("metrics_sha256")==hashlib.sha256(path.read_bytes()).hexdigest()
                      and proof.get("predictions_sha256")==hashlib.sha256((path.parent/"predictions.csv").read_bytes()).hexdigest())
        if approved:add(path,scenario=scene,new=True)
        else:
            for s in SYSTEMS:
                for method in METHODS:
                    info=counts[s]
                    records.append({"系统":NAMES[s],"方法":METHODS[method],"实验场景":SCENARIOS[scene],
                                    "版本划分":PROTOCOLS["fivefold"],"评价子集":"全部","完成状态":"未完成，暂无完整实验分数",
                                    "总缺陷版本数":info["versions"],"已完成准备版本数":info["completed"] if scene=="main" else 0,
                                    "全部配置数":info["configurations"],"候选配置数":info["passed"] if scene=="main" else None,
                                    "原始通过配置数":info["passed"],"主实验观察核验完成版本数":info["observed"],
                                    "主实验补测完成版本数":info["replayed"],"结果来源":str(data.joinpath(*SYSTEMS[s])),
                                    "更新时间":now,"说明":"各案例目录保存中间产物，可续跑。空白指标不是0，不得作为完整比较结果使用。"})
    unique=[tuple(r[k] for k in ["系统","方法","实验场景","版本划分","评价子集"]) for r in records]
    if len(set(unique))!=len(unique):raise ValueError("汇总行重复")
    system_order={v:i for i,v in enumerate(NAMES.values())}
    method_order={v:i for i,v in enumerate(METHODS.values())}
    def order(r):
        primary=(r["实验场景"]==SCENARIOS["main"] and r["版本划分"]==PROTOCOLS["fivefold"] and r["评价子集"]=="全部")
        return (not primary,system_order[r["系统"]],r["实验场景"],r["版本划分"],r["评价子集"],method_order[r["方法"]])
    records.sort(key=order)
    write_csv(output,records,COLUMNS)
    write_csv(output.with_name("新增系统逐版本运行记录.csv"),progress)
    manifest=output.with_suffix(".sources.json")
    manifest.write_text(json.dumps(dict(generated=now,rows=len(records),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                                        source_files=sources,counts=counts),ensure_ascii=False,indent=2),encoding="utf8")
    with output.open(encoding="utf-8-sig",newline="") as stream:
        reread=list(csv.DictReader(stream))
    assert len(reread)==len(records)
    for row in reread:
        if row["完成状态"].startswith("未完成"):
            assert all(not row[k] for k in ["精确率","召回率","调和平均值","准确率","成功评分数"])
    print(output,len(records),"rows verified",flush=True)
    for row in records[:18]:print(row["系统"],row["方法"],row.get("调和平均值","未完成"),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("D:/dataset/whn"))
    p.add_argument("--output",type=Path,default=ROOT/"experimental_results/六系统单缺陷实验比较.csv")
    a=p.parse_args(); make(a.data,a.output)
