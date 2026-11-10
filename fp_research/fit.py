"""按完整版本交叉拟合检出网络和简单子集频率基线。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


from .detection import descriptor, encode, samples, train_model, predict, frequency_features
from .folds import splits
from .inventory import atomic_json


class FrequencyEstimate:
    def __init__(self, bags, labels):
        self.features=frequency_features(bags)
        self.labels=labels

    def predict(self, queries):
        k=min(32,len(self.labels))
        result=[]
        for query in queries:
            distances=np.square(self.features-query).sum(axis=1)
            indices=np.argpartition(distances,k-1)[:k]
            result.append(float(self.labels[indices].mean()))
        return np.array(result)


def descriptors(version):
    raw=json.loads((version/"semantics.json").read_text(encoding="utf8"))
    cases={k:descriptor(v) for k,v in raw["cases"].items()}
    return raw,cases


def estimate(version,model,ablated,frequency):
    raw,cases=descriptors(version)
    rows=json.loads((version/"inventory.json").read_text(encoding="utf8"))
    passing=[r for r in rows if r["outcome"]=="passed"]
    result={r["configuration"]:{} for r in passing}
    suites=[[cases[r["configuration"]+"/"+t] for t in r["tests"]["passed"]] for r in passing]
    for group, members in raw["groups"].items():
        query=cases[members[0]]
        bags=[encode(query,suite) for suite in suites]
        complete=predict(model,bags)
        no_assertions=predict(ablated,bags,True)
        rates=frequency.predict(frequency_features(bags))
        for i,row in enumerate(passing):
            result[row["configuration"]][group]=[float(v[i]) for v in (complete,no_assertions,rates)]
    return result


def run(root,output,selected=None):
    experiments=[s for s in splits(root) if not selected or s[0]==selected]
    # 每个版本独立采样；各折的交集只发生在训练版本之间。
    parts={v:samples(v) for v in sorted({v for _,tr,te in experiments for v in tr+te})}
    for name,train,test in experiments:
        if set(train)&set(test):
            raise ValueError("版本划分有交叉")
        dest=output/name
        dest.mkdir(parents=True,exist_ok=True)
        bags=[b for v in train for b in parts[v][0]]
        labels=np.concatenate([parts[v][1] for v in train])
        model=train_model(bags,labels)
        ablated=train_model(bags,labels,True)
        frequency=FrequencyEstimate(bags,labels)
        torch.save(model.state_dict(),dest/"model.pt")
        brier_sum=0.0
        heldout=0
        for version in test:
            estimates=estimate(version,model,ablated,frequency)
            atomic_json(dest/(version.name+".json"),estimates)
            vs,ys=parts[version]
            for start in range(0,len(vs),128):
                pred=predict(model,vs[start:start+128])
                brier_sum+=float(np.square(pred-ys[start:start+128]).sum())
                heldout+=len(pred)
        source_files=[Path(__file__),Path(__file__).with_name("detection.py"),Path(__file__).with_name("folds.py")]
        fingerprint=hashlib.sha256(b"".join(f.read_bytes() for f in source_files)).hexdigest()
        atomic_json(dest/"fold.json",dict(train=[p.as_posix() for p in train],test=[p.as_posix() for p in test],
                    train_bags=len(labels),held_out_bags=heldout,brier=brier_sum/max(1,heldout),
                    implementation_sha256=fingerprint,seed=42))
        print(name,len(labels),heldout,brier_sum/max(1,heldout),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--split")
    a=p.parse_args()
    run(a.input,a.output,a.split)
