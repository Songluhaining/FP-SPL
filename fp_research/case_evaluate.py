"""从案例内实验文件对齐新增系统的预测并独立评价。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd

from .case_study import ROOT,SYSTEMS,case_versions,results
from .compare import run,load_features
from .compare_latent import KEYS,paired_difference
from .extension_evaluate import eval_metrics
from .truth import ground_truth


def partitions(data,scenario,with_external=True):
    identities={}
    versions={}
    for system in SYSTEMS:
        versions[system]=[results(v,scenario) for v in case_versions(data,system)]
        for path in versions[system]:
            identities[path]=(system,path.parent.parent.name)
    splits=[]
    for s,paths in versions.items():
        ordered=sorted(paths,key=lambda p:hashlib.sha256(("42/"+s+"/"+identities[p][1]).encode()).digest())
        for fold in range(5):
            test=ordered[fold::5]
            splits.append((f"{s}-fold{fold}",[p for p in ordered if p not in test],test))
        other=next(k for k in SYSTEMS if k!=s)
        splits.append((f"{other}-to-{s}",versions[other],paths))
    if with_external:
        previous=ROOT/"result"/("fp" if scenario=="main" else "supplement")
        paths=sorted(p.parent for p in previous.glob("*/*/inventory.json") if p.parent.parent.name in {"bank","email"})
        more=data/"fp_extension_work"/scenario
        paths+=sorted(p.parent for p in more.glob("*/*/inventory.json") if p.parent.parent.name in {"elevator","examdb"})
        if len(paths)!=178: raise ValueError("先前四系统的178个训练版本不齐")
        for p in paths: identities[p]=(p.parent.name,p.name)
        for s in SYSTEMS: splits.append(("previous-four-to-"+s,paths,versions[s]))
    return splits,identities


def prepare(a):
    truth=pd.DataFrame(ground_truth(a.data,SYSTEMS)).assign(converted=0)
    version_rows=[]
    construction=[]
    for system in SYSTEMS:
        for v in case_versions(a.data,system):
            folder=results(v,a.scenario)
            rows=json.loads((folder/"inventory.json").read_text(encoding="utf8"))
            audit=json.loads((folder/"coverage-audit.json").read_text(encoding="utf8"))
            complete=json.loads((folder/"complete.json").read_text(encoding="utf8"))
            if a.scenario=="masked":
                receipt=json.loads((folder/"construction.json").read_text(encoding="utf8"))
                original=json.loads((results(v,"main")/"inventory.json").read_text(encoding="utf8"))
                base=truth.loc[(truth.system==system)&(truth.version==v.name),"base"].unique().item()
                for r in original:
                    if r["outcome"]!="failed": continue
                    c=r["configuration"]
                    item=receipt.get(c,{"status":"retained_failure"})
                    construction.append(dict(system=system,version=v.name,configuration=c,**item))
                    if item["status"]=="passed":truth.loc[len(truth)]=[system,v.name,c,1,None,base,1]
            selected=truth[(truth.system==system)&(truth.version==v.name)]
            passed={r["configuration"] for r in rows if r["outcome"]=="passed"}
            if set(selected.configuration)!=passed or len(selected)!=len(passed):
                raise ValueError("原本通过清单与评价真值不同")
            if audit["changed_coverage_counts"] or audit["tests"]!=sum(len(t) for r in rows for t in r["tests"].values()):
                raise ValueError("单测试覆盖核验失败")
            version_rows.append(dict(system=system,version=v.name,configurations=len(rows),candidates=len(passed),
                                     positives=int(selected.label.sum()),tests=audit["tests"],**complete))
    truth.to_csv(a.output/"truth.csv",index=False)
    pd.DataFrame(version_rows).to_csv(a.output/"versions.csv",index=False)
    if construction:pd.DataFrame(construction).to_csv(a.output/"construction.csv",index=False)


def classify(a,target):
    split,identities=partitions(a.data,a.scenario)
    truth={}
    if a.method=="CLAP":
        old=ROOT/"result/evaluation"/("truth.csv" if a.scenario=="main" else "supp-truth.csv")
        sources=[old,ROOT/"experimental_results/latent_extension"/a.scenario/"truth.csv",a.output/"truth.csv"]
        labels=pd.concat(pd.read_csv(path) for path in sources)
        if labels.duplicated(KEYS).any():raise ValueError("重复训练标签")
        truth=dict(zip(labels[KEYS].itertuples(index=False,name=None),labels.label))
    run(None,ROOT.parent/a.method,a.method,truth,target,partitions=split,identities=identities)


def evaluate(a):
    truth=pd.read_csv(a.output/"truth.csv")
    ids=set(truth[KEYS].itertuples(index=False,name=None))
    raw=pd.concat(pd.read_csv(results(v,a.scenario)/"latent.csv") for s in SYSTEMS for v in case_versions(a.data,s))
    parts=[pd.read_csv(a.output/(m+".csv")) for m in ("CLAP","PULP")]
    assignments=[]
    training=[]
    split,identities=partitions(a.data,a.scenario)
    for name,train,test in split:
        tr={identities[p] for p in train}
        te={identities[p] for p in test}
        if tr & te:raise ValueError("训练与评价版本相交")
        for role,items in [("train",tr),("test",te)]:
            assignments.extend(dict(split=name,role=role,system=s,version=v) for s,v in sorted(items))
        chosen=raw[[i in te for i in zip(raw.system,raw.version)]]
        parts.append(chosen.assign(split=name,status="scored"))
        for method in ("CLAP","PULP"):
            n,positive=0,0
            for path in train:
                names,observed,_=load_features(path,method)
                n+=len(names)
                if method=="PULP":positive+=observed.count("F")
            training.append(dict(split=name,method=method,training_configurations=n,
                                 visible_failure_positives=positive if method=="PULP" else None,
                                 sampling_ratio=(positive+1.e-6)/(n-positive) if method=="PULP" else None))
    frame=pd.concat(parts,ignore_index=True).merge(truth,on=KEYS,how="left",validate="many_to_one")
    frame["protocol"]=["fivefold" if "-fold" in s else "external" if s.startswith("previous-four-") else "transfer"
                       for s in frame.split]
    if frame.label.isna().any():raise ValueError("预测缺少评价标签")
    for _,g in frame.groupby(["protocol","method"]):
        if len(g)!=len(ids) or set(g[KEYS].itertuples(index=False,name=None))!=ids:
            raise ValueError("候选有遗漏或重复")
    tables=[]
    for (s,p,m),g in frame.groupby(["system","protocol","method"]):
        groups={"all":g,"base":g[g.base==1],"other":g[g.base==0]}
        if a.scenario=="masked":groups["converted"]=g[g.converted==1]
        for name,sub in groups.items():
            if len(sub): tables.append(dict(system=s,protocol=p,method=m,subset=name,**eval_metrics(sub)))
        if a.scenario=="main":
            tables.append(dict(system=s,protocol=p,method=m,subset="executed",**eval_metrics(g,"executed_label")))
    frame.to_csv(a.output/"predictions.csv",index=False)
    pd.DataFrame(assignments).to_csv(a.output/"folds.csv",index=False)
    pd.DataFrame(training).to_csv(a.output/"training.csv",index=False)
    pd.DataFrame(tables).to_csv(a.output/"metrics.csv",index=False)
    intervals=[]
    for s,g in frame[frame.protocol=="fivefold"].groupby("system"):
        for method in ("CLAP","PULP"):
            low,high=paired_difference(g,method)
            intervals.append(dict(system=s,baseline=method,lower=low,upper=high))
    pd.DataFrame(intervals).to_csv(a.output/"paired-differences.csv",index=False)
    for method in ("CLAP","PULP"):
        rows=[]
        for s in SYSTEMS:
            for v in case_versions(a.data,s):
                names,observed,table=load_features(results(v,a.scenario),method)
                rows.append(table.assign(system=s,version=v.name,configuration=names,observed=observed))
        pd.concat(rows,ignore_index=True).to_csv(a.output/("features-"+method+".csv.gz"),index=False,
                                                compression={"method":"gzip","mtime":0})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["truth","fit","audit","repeat"])
    p.add_argument("--scenario",choices=["main","masked"],default="main")
    p.add_argument("--method",choices=["CLAP","PULP"])
    p.add_argument("--data",type=Path,default=Path("D:/dataset/whn"))
    p.add_argument("--output",type=Path,default=ROOT/"experimental_results/gpl_zipme")
    a=p.parse_args()
    a.output=a.output/a.scenario
    a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=="truth":prepare(a)
    elif a.stage=="audit":evaluate(a)
    elif a.stage=="repeat":
        with tempfile.TemporaryDirectory(dir=a.output) as temp:
            target=Path(temp)/"repeat.csv"
            classify(a,target)
            if target.read_bytes()!=(a.output/(a.method+".csv")).read_bytes():raise ValueError("重训结果不一致")
    else:classify(a,a.output/(a.method+".csv"))


if __name__=="__main__":
    main()
