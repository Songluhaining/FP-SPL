"""仅使用发布的明细表复核分类指标、排序指标和每项补测预算。"""
import argparse
from collections import Counter,defaultdict
import csv
from itertools import groupby
import math
from pathlib import Path


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def check_metrics(folder,counts):
    predictions=read(folder/"predictions.csv")
    groups=defaultdict(list)
    keys=set()
    for row in predictions:
        group=(row["system"],row["method"],"-to-" in row["split"])
        k=(*group,row["version"],row["configuration"])
        if k in keys:
            raise ValueError(f"重复预测 {k}")
        keys.add(k)
        groups[group].append(row)
    summary=[r for r in read(folder/"metrics.csv") if r["subset"]=="all"]
    assert len(summary)==len(groups)==24
    for metric in summary:
        rows=groups[metric["system"],metric["method"],metric["transfer"]=="True"]
        assert len(rows)==counts[metric["system"]]==int(metric["n"])
        matrix=Counter((int(r["label"]),int(r["prediction"])) for r in rows)
        tp,fp,tn,fn=(matrix[k] for k in [(1,1),(0,1),(0,0),(1,0)])
        assert [tp,fp,tn,fn]==[int(metric[k]) for k in ("tp","fp","tn","fn")]
        precision=tp/max(1,tp+fp)
        recall=tp/max(1,tp+fn)
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        def score(r):
            return float(r["score"]) if r["score"] else float("-inf")
        found,inspected,ap=0,0,0.0
        for _,tied in groupby(sorted(rows,key=score,reverse=True),score):
            tied=list(tied)
            added=sum(int(r["label"]) for r in tied)
            found+=added
            inspected+=len(tied)
            ap+=(added/max(1,tp+fn))*(found/inspected)
        for k,v in dict(precision=precision,recall=recall,f1=f1,accuracy=(tp+tn)/len(rows),average_precision=ap).items():
            if not math.isclose(float(metric[k]),v,abs_tol=1e-12):
                raise ValueError(f"{folder.name} {metric['system']} {metric['method']} {k}: {v}")
    return len(predictions)


def check_replays(root):
    outcomes={(r["system"],r["version"],r["configuration"],r["donor"]):r for r in read(root/"replay-outcomes.csv")}
    ranks=defaultdict(list)
    evidence=read(root/"probe-evidence.csv")
    for r in evidence:
        key=tuple(r[k] for k in ("system","version","configuration"))
        probe=outcomes[(*key,r["donor_test"])]
        assert math.isclose(float(r["augmented_detection"])-float(r["suite_detection"]),float(r["gain"]),abs_tol=1e-6)
        for strategy in ("model","random","coverage"):
            if r[strategy+"_rank"]:
                ranks[(*key,strategy)].append((int(r[strategy+"_rank"]),probe["status"]))
    assert len(evidence)==len(outcomes)
    budgets=read(root/"probe-budgets.csv")
    assert len(budgets)==2202*3*3
    for row in budgets:
        key=tuple(row[k] for k in ("system","version","configuration","strategy"))
        k=int(row["budget"])
        chosen=[status for pos,status in ranks[key] if pos<=k]
        assert len(chosen)==int(row["selected"])==min(k,int(row["eligible_pool"]))
        assert int("confirmed" in chosen)==int(row["confirmed"])
    return len(outcomes),len(budgets)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    root=p.parse_args().results
    n=check_metrics(root/"main",dict(bank=1697,email=505))
    m=check_metrics(root/"masked",dict(bank=2074,email=728))
    probes,budgets=check_replays(root/"main")
    print("verified",n+m,"predictions,",probes,"actual probes,",budgets,"candidate-budget decisions")
