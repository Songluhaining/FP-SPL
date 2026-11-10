"""从留出模型重新计算每条已选补测用例的预测增益，并核对模型选例顺序。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .detection import DetectionNet,encode,predict
from .fit import descriptors


def write_evidence(root,plans,models,risk,output):
    cases_by_version={}
    with risk.open(encoding="utf8",newline="") as f:
        for row in csv.DictReader(f):
            if row["method"]=="full" and "-fold" in row["split"]:
                cases_by_version.setdefault((row["system"],row["version"]),[]).append(row)
    torch.set_num_threads(1)
    networks={}
    total=0
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf8") as f:
        w=csv.writer(f)
        w.writerow(["system","version","configuration","group","risk","donor_test","suite_detection",
                    "augmented_detection","gain","model_rank","random_rank","coverage_rank"])
        for (system,version),rows in cases_by_version.items():
            v=root/system/version
            sem,cases=descriptors(v)
            suites={r["configuration"]:r["tests"]["passed"] for r in json.loads((v/"inventory.json").read_text(encoding="utf8"))}
            plan=json.loads((plans/system/version/"probe-plan.json").read_text(encoding="utf8"))
            split=rows[0]["split"]
            if split not in networks:
                net=DetectionNet()
                net.load_state_dict(torch.load(models/split/"model.pt",map_location="cpu",weights_only=True))
                networks[split]=net.eval()
            model=networks[split]
            for row in rows:
                c=row["configuration"]
                choices=plan[c]
                if not choices.get("model"):
                    continue
                query=cases[sem["groups"][row["group"]][0]]
                suite=encode(query,[cases[c+"/"+k] for k in suites[c]])
                base=float(predict(model,[suite])[0])
                ids=sorted({k for mode in ("model","random","coverage") for k in choices[mode]})
                probs=predict(model,[np.vstack([suite,encode(query,[cases[k]])]) for k in ids])
                gains={k:float(p)-base for k,p in zip(ids,probs)}
                chosen=choices["model"]
                if any(gains[a]+1e-5<gains[b] for a,b in zip(chosen,chosen[1:])):
                    raise ValueError(f"模型选择顺序与重算增益不一致 {system}/{version}/{c}")
                if any(gains[k]>gains[chosen[-1]]+1e-5 for k in set(ids)-set(chosen)):
                    raise ValueError(f"模型选例遗漏更高增益的已检查候选 {system}/{version}/{c}")
                for k,p in zip(ids,probs):
                    ranks=[choices[m].index(k)+1 if k in choices[m] else "" for m in ("model","random","coverage")]
                    w.writerow([system,version,c,row["group"],row["score"],k,base,float(p),gains[k],*ranks])
                    total+=1
    print("verified selected test scores",total)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("input","plans","models","risk","output"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    write_evidence(a.input,a.plans,a.models,a.risk,a.output)
