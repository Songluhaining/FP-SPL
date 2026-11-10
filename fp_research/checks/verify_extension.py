"""从归档明细独立核验版本划分、混淆矩阵、分母、排序预算与单类训练折。"""
import argparse
from collections import Counter
import csv
import hashlib
import math
from pathlib import Path

KEY=("system","version","configuration")


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def identity(row):
    return tuple(row[k] for k in KEY)


def verify(path):
    truth=read(path/"truth.csv")
    keys={identity(row) for row in truth}
    labels={identity(r):int(r["label"]) for r in truth}
    if len(keys)!=len(truth): raise ValueError("真值配置重复")
    rows=read(path/"predictions.csv")
    unique={(identity(r),r["method"],r["protocol"]):r for r in rows}
    if len(rows)!=len(unique) or len(rows)!=len(keys)*12: raise ValueError("预测行数或唯一性不符")
    partitions=read(path/"folds.csv")
    for split in {r["split"] for r in partitions}:
        related=[r for r in partitions if r["split"]==split]
        train={(r["system"],r["version"]) for r in related if r["role"]=="train"}
        test={(r["system"],r["version"]) for r in related if r["role"]=="test"}
        if train & test: raise ValueError("训练与留出版本相交")
        if "-fold" in split:
            s,fold=split.split("-fold")
            order=sorted({v for system,v,c in keys if system==s},
                         key=lambda v:hashlib.sha256(f"42/{s}/{v}".encode()).digest())
            if test!={(s,v) for v in order[int(fold)::5]}: raise ValueError("独立重建折号不同")
            one_class=len({y for (s,v,c),y in labels.items() if (s,v) in train})==1
            for r in rows:
                if r["split"]==split and r["method"]=="CLAP":
                    if (r["status"]=="insufficient")!=one_class: raise ValueError("单类弃判错误")
        for r in rows:
            if r["split"]==split and ((r["system"],r["version"]) not in test or
                                       labels[identity(r)]!=int(r["label"])):
                raise ValueError("留出身份或真值错位")
    for table in read(path/"metrics.csv"):
        case=[r for r in rows if all(r[k]==table[k] for k in ("system","protocol","method"))]
        if {identity(r) for r in case}!={k for k in keys if k[0]==table["system"]}:
            raise ValueError("方法分母不同")
        subset=table["subset"]
        if subset in {"base","other"}: case=[r for r in case if int(r["base"])==int(subset=="base")]
        if subset=="converted": case=[r for r in case if int(r["converted"])==1]
        y="executed_label" if subset=="executed" else "label"
        counts=Counter((int(float(r[y])),int(r["prediction"])) for r in case)
        tp,fp,tn,fn=(counts[k] for k in ((1,1),(0,1),(0,0),(1,0)))
        scored=sum(bool(r["score"]) for r in case)
        expected=dict(n=len(case),tp=tp,fp=fp,tn=tn,fn=fn,scored=scored,insufficient=len(case)-scored,
                      precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),accuracy=(tp+tn)/len(case),
                      f1=2*tp/max(1,2*tp+fp+fn),hits=0,budget=0)
        for version in {r["version"] for r in case}:
            group=[r for r in case if r["version"]==version]
            budget=math.ceil(len(group)/10)
            ranked=sorted((r for r in group if r["score"]),key=lambda r:(-float(r["score"]),r["configuration"]))
            expected["hits"]+=sum(int(float(r[y])) for r in ranked[:budget])
            expected["budget"]+=budget
        if any(not math.isclose(val,float(table[k]),abs_tol=1e-12) for k,val in expected.items()):
            raise ValueError(f"复算失败 {table}")
    print(path.name,len(keys),"unique configurations,",len(rows),"verified predictions")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    args=parser.parse_args()
    for scenario in ("main","masked"):
        verify(args.results/scenario)
