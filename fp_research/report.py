"""从逐配置明细独立重算指标并校验分母、真值与折外版本身份。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

METHODS=("full","CLAP","PULP","no_config","no_assertion","frequency")
ZH={"full":"新方法","CLAP":"有监督分类（CLAP）","PULP":"正例与未标注学习（PULP）",
    "no_config":"移除配置适用条件","no_assertion":"检出网络移除断言输入","frequency":"子集频率替代网络"}


def audit(data,truth,root):
    ids=["system","version","configuration"]
    if truth.duplicated(ids).any() or data.duplicated(ids+["split","method"]).any():
        raise ValueError("逐配置主键重复")
    expected={}
    for system in ("bank","email"):
        files=list((root/system).glob("*/inventory.json"))
        order=sorted([f.parent.name for f in files],key=lambda v:hashlib.sha256(("42/"+system+"/"+v).encode()).digest())
        fold={v:i%5 for i,v in enumerate(order)}
        for f in files:
            for row in json.loads(f.read_text(encoding="utf8")):
                if row["outcome"]=="passed":
                    expected[(system,f.parent.name,row["configuration"])]=fold[f.parent.name]
    if set(expected)!=set(truth[ids].itertuples(index=False,name=None)):
        raise ValueError("评价真值与原始测试状态清单不一致")
    for method in METHODS:
        subset=data[data.method==method]
        for transfer in (True,False):
            d=subset[subset.split.str.contains("-to-")==transfer]
            keys=list(d[ids].itertuples(index=False,name=None))
            if len(keys)!=len(expected) or set(keys)!=set(expected):
                raise ValueError(f"分母不一致 {method}, transfer={transfer}")
            for key,s in zip(keys,d.split):
                dst=key[0]
                name=(("email" if dst=="bank" else "bank")+"-to-"+dst) if transfer else f"{dst}-fold{expected[key]}"
                if s!=name:
                    raise ValueError("评估折号与独立重建结果不一致")
    if set(data.method)!=set(METHODS):
        raise ValueError("方法清单异常")
    return data.merge(truth,on=ids,how="left",validate="many_to_one")


def metrics(d,label="label"):
    y=d[label].to_numpy(dtype=int)
    p=d.prediction.to_numpy(dtype=int)
    if not set(p)<=set([0,1]):
        raise ValueError("非法决策值")
    tp=int(np.sum((y==1)&(p==1)))
    tn=int(np.sum((y==0)&(p==0)))
    fp=int(np.sum((y==0)&(p==1)))
    fn=int(np.sum((y==1)&(p==0)))
    prec=tp/max(1,tp+fp)
    rec=tp/max(1,tp+fn)
    scores=d.score.fillna(-1e30)
    hits=0
    budget=0
    for _,g in d.groupby("version"):
        selected=g.sort_values(["score","configuration"],ascending=[False,True],na_position="last").head(math.ceil(len(g)*0.1))
        hits+=int(selected[label].sum())
        budget+=len(selected)
    return dict(n=len(d),scored=int(scores.size-d.score.isna().sum()),partial=int((d.status=="partial").sum()),
                insufficient=int((d.status=="insufficient").sum()),tp=tp,fp=fp,tn=tn,fn=fn,precision=prec,
                recall=rec,f1=2*tp/max(1,2*tp+fp+fn),accuracy=(tp+tn)/len(d),average_precision=float(average_precision_score(y,scores)) if y.sum() else None,
                hits=hits,budget=budget)


def write_tables(frame,destination):
    records=[]
    for (system,transfer,method),g in frame.groupby(["system",frame.split.str.contains("-to-"),"method"]):
        for subset,sub in [("all",g),("base",g[g.base==1]),("other",g[g.base==0])]:
            if len(sub):
                records.append(dict(system=system,transfer=transfer,method=method,subset=subset,**metrics(sub)))
        if "converted" in g:
            sub=g[g.converted==1]
            if len(sub):
                records.append(dict(system=system,transfer=transfer,method=method,subset="converted_only",**metrics(sub)))
        elif g.executed_label.notna().all():
            records.append(dict(system=system,transfer=transfer,method=method,subset="positive_execution",**metrics(g,"executed_label")))
    table=pd.DataFrame.from_records(records)
    table.to_csv(destination,index=False)
    return table


def replay_table(root,truth,output):
    truth_keys=set(truth[["system","version","configuration"]].itertuples(index=False,name=None))
    seen=set()
    rows=[]
    receipts=[]
    for f in root.glob("*/*/probe-plan.json"):
        system,version=f.parent.parent.name,f.parent.name
        for c,plan in json.loads(f.read_text(encoding="utf8")).items():
            key=(system,version,c)
            if key in seen:
                raise ValueError("补测清单重复")
            seen.add(key)
            receipt=f.parent/"executions"/(c+".json")
            results=json.loads(receipt.read_text(encoding="utf8"))["results"]
            eligible={x for strategy in ("model","random","coverage") for x in plan.get(strategy,[])}
            if eligible!=set(results):
                raise ValueError(f"补测执行与已选候选不符 {key}")
            if any(len(choice)!=len(set(choice)) or len(choice)!=min(5,plan.get("pool",0))
                   for choice in (plan.get(k,[]) for k in ("model","random","coverage"))):
                raise ValueError(f"补测名额或候选唯一性异常 {key}")
            for outcome in results.values():
                if outcome["status"]=="confirmed" and not (
                    outcome["reference"][0]=="passed" and outcome["first"][0]=="assertion"
                    and outcome["first"]==outcome["second"]
                ):
                    raise ValueError(f"确认记录缺少双次独立复现 {key}")
            for strategy in ("model","random","coverage"):
                for k in (1,3,5):
                    xs=plan.get(strategy,[])[:k]
                    statuses=[results[x]["status"] for x in xs]
                    rows.append(dict(system=system,version=version,configuration=c,strategy=strategy,budget=k,
                        selected=len(xs),eligible=int(plan.get("pool",0)>0),eligible_pool=plan.get("pool",0),confirmed=int("confirmed" in statuses),
                        compile_errors=statuses.count("compile_error"),
                        reference_checks=sum(s!="compile_error" for s in statuses),
                        target_checks=sum("first" in results[x] for x in xs)+sum(results[x].get("second") is not None for x in xs),
                        timeouts=sum(results[x].get("status")=="reference_timeout" or any(results[x].get(at) and results[x][at][0]=="timeout" for at in ("first","second")) for x in xs)))
            for x,e in results.items():
                receipts.append(dict(system=system,version=version,configuration=c,donor=x,**e))
    if seen!=truth_keys:
        raise ValueError("补测清单未覆盖全部主实验配置")
    records=pd.DataFrame(rows)
    records.to_csv(output/"probe-budgets.csv",index=False)
    pd.DataFrame(receipts).to_csv(output/"replay-outcomes.csv",index=False)
    return records.groupby(["system","strategy","budget"])[["eligible","selected","confirmed","compile_errors","reference_checks","target_checks","timeouts"]].sum().reset_index()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("input","truth","risk","clap","pulp","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--replay",type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    y=pd.read_csv(a.truth)
    data=pd.concat([pd.read_csv(f) for f in (a.risk,a.clap,a.pulp)],ignore_index=True)
    joined=audit(data,y,a.input)
    joined.to_csv(a.output/"predictions.csv",index=False)
    table=write_tables(joined,a.output/"metrics.csv")
    if a.replay:
        replay_table(a.replay,y,a.output).to_csv(a.output/"replay-summary.csv",index=False)
    print(table[(table.subset=="all")&table.method.isin(("full","CLAP","PULP"))].to_string(index=False))
