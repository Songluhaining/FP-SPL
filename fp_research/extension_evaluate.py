"""评价新增系统。分类只用训练标签，评价器按原始配置清单核对分母。"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import tempfile

import pandas as pd

from .audit_evidence import check_masked
from .compare import load_features, run as train
from .compare_latent import KEYS, paired_difference
from .extension import ROOT, SYSTEMS, versions
from .folds import splits
from .report import metrics
from .truth import ground_truth

METHODS=("statement","configuration","CLAP","PULP")


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def data_tables(a):
    truth=pd.DataFrame(ground_truth(a.data,SYSTEMS)).assign(converted=0)
    audits=[]
    constructed=[]
    for system in SYSTEMS:
        for v in versions(a.data,system):
            stage=a.work/a.scenario/system/v.name
            rows=read_json(stage/"inventory.json")
            by_name={r["configuration"]:r for r in rows}
            raw=a.work/"main"/system/v.name
            obs=read_json(raw/"inventory.json")
            report=v/"config.report.csv.done"
            with report.open(encoding="utf-8-sig",newline="") as f:
                raw_results={r[0]:r[-1].strip("_").lower() for r in list(csv.reader(f))[1:]}
            if raw_results!={r["configuration"]:r["outcome"] for r in obs}:
                raise ValueError(f"原始状态与登记表冲突 {v}")
            native=truth[(truth.system==system)&(truth.version==v.name)]
            for r in native.itertuples():
                path=v/"variants"/r.configuration/"coverage/spectrum_passed_coverage.xml"
                marked=any(line.get("buggy")=="true" for line in ET.parse(path).iter("line"))
                if marked!=bool(r.label):
                    raise ValueError(f"故障标记与变异日志重建标签不同 {path}")
            if a.scenario=="masked":
                receipt=read_json(stage/"construction.json")
                check_masked(v,a.work/"visible"/system/v.name,by_name,receipt)
                base=truth.loc[(truth.system==system)&(truth.version==v.name),"base"].unique().item()
                for r in obs:
                    cid=r["configuration"]
                    if r["outcome"]!="failed":
                        continue
                    item=receipt.get(cid,{"status":"retained_failure"})
                    if item["status"]=="passed":
                        truth.loc[len(truth)]=[system,v.name,cid,1,None,base,1]
                    constructed.append(dict(system=system,version=v.name,configuration=cid,**item))
            checked=read_json(stage/"coverage-audit.json")
            covered=sum(len(t) for r in rows for t in r["tests"].values())
            if checked["tests"]!=covered or checked["changed_coverage_counts"]:
                raise ValueError(f"逐用例核验失败 {v}")
            passed={r["configuration"] for r in rows if r["outcome"]=="passed"}
            current=truth[(truth.system==system)&(truth.version==v.name)]
            if len(current)!=len(passed) or set(current.configuration)!=passed:
                raise ValueError(f"通过配置和标签不匹配 {v}")
            audits.append(dict(**checked,configurations=len(rows),candidates=len(passed),
                               positives=int(current.label.sum()),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()))
    if constructed:
        pd.DataFrame(constructed).to_csv(a.output/"construction.csv",index=False)
    pd.DataFrame(audits).to_csv(a.output/"versions.csv",index=False)
    return truth


def external(a):
    previous=a.prior/("fp" if a.scenario=="main" else "supplement")
    training=sorted(p.parent for p in previous.glob("*/*/inventory.json") if p.parent.parent.name in {"bank","email"})
    if len(training)!=109:
        raise ValueError("缺少前两个系统的训练版本")
    target=a.work/a.scenario
    return [("bank-email-to-"+s,training,[target/s/v.name for v in versions(a.data,s)]) for s in SYSTEMS]


def fit_baseline(a,method,target):
    labels={}
    if method=="CLAP":
        prior=a.prior/"evaluation"/("truth.csv" if a.scenario=="main" else "supp-truth.csv")
        truth=pd.concat([pd.read_csv(prior),pd.read_csv(a.output/"truth.csv")],ignore_index=True)
        labels=dict(zip(truth[KEYS].itertuples(index=False,name=None),truth.label))
    train(a.work/a.scenario,a.sources/method,method,labels,target,
          systems=tuple(SYSTEMS),extra_splits=external(a))


def classify(a):
    fit_baseline(a,a.stage,a.output/(a.stage+".csv"))
    frames=[]
    for s in SYSTEMS:
        for v in versions(a.data,s):
            ids,observations,numbers=load_features(a.work/a.scenario/s/v.name,a.stage)
            frames.append(numbers.assign(system=s,version=v.name,configuration=ids,observed=observations))
    pd.concat(frames,ignore_index=True).to_csv(a.output/("features-"+a.stage+".csv.gz"),index=False,
                                              compression={"method":"gzip","mtime":0})


def repeat(a):
    method=a.method
    with tempfile.TemporaryDirectory(dir=a.work) as tmp:
        copy=Path(tmp)/"predictions.csv"
        fit_baseline(a,method,copy)
        if copy.read_bytes()!=(a.output/(method+".csv")).read_bytes():
            raise ValueError(f"固定随机种子未复现预测 {a.scenario}/{method}")


def eval_metrics(part,label="label"):
    row=metrics(part,label)
    # 得分缺失者可计作未报出；无分配置不能占用排名预算或取得命中。
    row["hits"]=sum(g.dropna(subset=["score"]).sort_values(["score","configuration"],ascending=[False,True])
                    .head(math.ceil(len(g)*0.1))[label].sum() for _,g in part.groupby("version"))
    return row


def audit(a):
    root=a.work/a.scenario
    truth=pd.read_csv(a.output/"truth.csv")
    expected=set(truth[KEYS].itertuples(index=False,name=None))
    if truth.duplicated(KEYS).any():
        raise ValueError("真值主键重复")
    parts=[pd.read_csv(a.output/(m+".csv")) for m in ("CLAP","PULP")]
    models=pd.concat([pd.read_csv(root/s/v.name/"latent.csv") for s in SYSTEMS for v in versions(a.data,s)])
    details=[]
    for name,training,testing in [*splits(root,tuple(SYSTEMS)),*external(a)]:
        tr={(v.parent.name,v.name) for v in training}
        te={(v.parent.name,v.name) for v in testing}
        if tr & te: raise ValueError(f"版本泄漏 {name}")
        for role,rows in [("train",tr),("test",te)]:
            details.extend(dict(split=name,role=role,system=s,version=v) for s,v in sorted(rows))
        subset=models[[k in te for k in zip(models.system,models.version)]]
        parts.append(subset.assign(split=name,status="scored"))
    pd.DataFrame(details).to_csv(a.output/"folds.csv",index=False)
    frame=pd.concat(parts,ignore_index=True).merge(truth,on=KEYS,validate="many_to_one",how="left")
    frame["protocol"]=["fivefold" if "-fold" in s else "external" if s.startswith("bank-email-") else "transfer"
                       for s in frame.split]
    mapping={(r["system"],r["version"],r["split"]) for r in details if r["role"]=="test"}
    if any((r.system,r.version,r.split) not in mapping for r in frame.itertuples()):
        raise ValueError("配置属于错误验证折")
    if frame.label.isna().any() or set(frame.method)!=set(METHODS): raise ValueError("标签或方法缺失")
    if frame.groupby(["protocol","method"]).ngroups!=3*len(METHODS): raise ValueError("缺少协议或方法")
    for _,g in frame.groupby(["protocol","method"]):
        if len(g)!=len(expected) or set(g[KEYS].itertuples(index=False,name=None))!=expected:
            raise ValueError("方法之间出现分母差异")
    frame.to_csv(a.output/"predictions.csv",index=False)
    out=[]
    for (sys,protocol,m),g in frame.groupby(["system","protocol","method"]):
        samples={"all":g,"base":g[g.base==1],"other":g[g.base==0]}
        if a.scenario=="masked": samples["converted"]=g[g.converted==1]
        for name,sub in samples.items():
            if len(sub): out.append(dict(system=sys,protocol=protocol,method=m,subset=name,**eval_metrics(sub)))
        if a.scenario=="main":
            out.append(dict(system=sys,protocol=protocol,method=m,subset="executed",**eval_metrics(g,"executed_label")))
    pd.DataFrame(out).to_csv(a.output/"metrics.csv",index=False)
    ci=[]
    for sys,g in frame[frame.protocol=="fivefold"].groupby("system"):
        for m in ("CLAP","PULP"):
            low,high=paired_difference(g,m)
            ci.append(dict(system=sys,baseline=m,lower=low,upper=high))
    pd.DataFrame(ci).to_csv(a.output/"paired-differences.csv",index=False)
    files=[]
    for method in ("CLAP","PULP"):
        source=a.sources/method
        for folder,dirs,names in os.walk(source):
            dirs[:]=sorted(d for d in dirs if d not in {"venv",".venv","__pycache__",".git","projects","logs"})
            for name in sorted(names):
                path=Path(folder)/name
                if path.suffix==".py" or name in {"all-statements-slicing.jar","featurehouse_20210219.jar"}:
                    files.append(dict(method=method,path=path.relative_to(source).as_posix(),
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(files).to_csv(a.output/"source-hashes.csv",index=False)


def replays(a):
    truth=pd.read_csv(a.output/"truth.csv").set_index(KEYS)
    results=[]
    for s in SYSTEMS:
        for v in versions(a.data,s):
            root=a.work/"main"/s/v.name
            plan=read_json(root/"replay-plan.json")
            labels=truth.loc[(s,v.name)]
            if set(plan)!=set(labels.index): raise ValueError(f"复核配置遗漏 {v}")
            for cid,r in plan.items():
                receipts=read_json(root/"executions"/(cid+".json"))["results"]
                if set(receipts)!=set(r["coverage"]): raise ValueError(f"补测用例遗漏 {v}/{cid}")
                for position,key in enumerate(r["coverage"],1):
                    item=receipts[key]
                    if item["status"]=="confirmed" and not (
                        item["reference"][0]=="passed" and item["first"][0]=="assertion" and item["first"]==item["second"]
                    ): raise ValueError("确认记录未满足双次复现要求")
                    results.append(dict(system=s,version=v.name,configuration=cid,position=position,
                                        donor_test=key,status=item["status"],first=json.dumps(item.get("first")),
                                        second=json.dumps(item.get("second")),reference=json.dumps(item.get("reference"))))
    out=pd.DataFrame(results)
    out.to_csv(a.output/"replay-outcomes.csv",index=False)
    frames=pd.read_csv(a.output/"predictions.csv")
    rows=[]
    for budget in (1,3,5):
        sample=out[out.position<=budget]
        confirmed=sample[sample.status=="confirmed"][KEYS].drop_duplicates()
        for (s,m),g in frames[frames.protocol=="fivefold"].groupby(["system","method"]):
            keys=confirmed[confirmed.system==s]
            hits=g.merge(keys,on=KEYS,validate="one_to_one")
            trials=sample[sample.system==s]
            rows.append(dict(system=s,method=m,budget=budget,planned_tests=len(trials),
                             with_tests=len(trials[KEYS].drop_duplicates()),confirmed=len(hits),
                             detected=int(hits.prediction.sum()),benchmark_negative=int((hits.label==0).sum()),
                             compile_errors=int((trials.status=="compile_error").sum())))
    pd.DataFrame(rows).to_csv(a.output/"replay-summary.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["truth","CLAP","PULP","audit","replay","repeat"])
    p.add_argument("--method",choices=["CLAP","PULP"])
    p.add_argument("--scenario",choices=["main","masked"],default="main")
    p.add_argument("--data",type=Path,default=Path("D:/dataset/whn"))
    p.add_argument("--work",type=Path,default=Path("D:/dataset/whn/fp_extension_work"))
    p.add_argument("--sources",type=Path,default=ROOT.parent)
    p.add_argument("--prior",type=Path,default=ROOT/"result")
    p.add_argument("--output",type=Path,default=ROOT/"experimental_results/latent_extension")
    a=p.parse_args()
    a.output=a.output/a.scenario
    a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=="truth": data_tables(a).to_csv(a.output/"truth.csv",index=False)
    elif a.stage=="audit": audit(a)
    elif a.stage=="replay": replays(a)
    elif a.stage=="repeat":
        if not a.method: p.error("重训核验需要指定比较方法")
        repeat(a)
    else: classify(a)
