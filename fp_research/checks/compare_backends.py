"""对照已获得的常规插桩结果，逐语句核对相同执行集合的后验。"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def compare(old,new):
    rows=[]
    for system in ("bank","email"):
        counts=Counter()
        largest=0.0
        for newer in (new/system).glob("*/graphs/*.json"):
            prior=old/system/newer.parent.parent.name/"graphs"/newer.name
            if not prior.is_file():
                continue
            b=json.loads(newer.read_text(encoding="utf8"))
            a=json.loads(prior.read_text(encoding="utf8"))
            if b.get("backend")!="isolated-method" or a["executions"]!=b["executions"]:
                continue
            counts["overlap"]+=1
            if a["status"]!="ok" and b["status"]=="ok":
                counts["recovered"]+=1
            if a["status"]=="ok" and b["status"]!="ok":
                counts["rejected"]+=1
            if a["status"]!=b["status"] or b["status"]!="ok":
                continue
            counts["both_scored"]+=1
            if set(a["scores"])!=set(b["scores"]):
                counts["different_statements"]+=1
                continue
            difference=max((abs(a["scores"][k]-x) for k,x in b["scores"].items()),default=0)
            largest=max(largest,difference)
            counts["different_posteriors"]+=int(difference>1e-6)
        rows.append(dict(system=system,**{key:counts[key] for key in
                    ("overlap","both_scored","different_statements","different_posteriors","recovered","rejected")},
                    max_absolute_difference=largest))
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("earlier","final","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    rows=compare(a.earlier,a.final)
    with a.output.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(rows)
