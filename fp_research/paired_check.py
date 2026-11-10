"""在留出版本中比较同配置、同操作、近似覆盖且断言不同的两组测试。"""
import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from .detection import DetectionNet,encode,predict
from .fit import descriptors
from .folds import splits
from .inventory import SYSTEMS


@lru_cache(maxsize=256)
def coverage(path):
    return {(f.get("name"),line.get("num")) for f in ET.parse(path).iterfind(".//package/file")
            for line in f.findall("line") if line.get("type")=="stmt" and int(line.get("count",0))>0}


def check(v,source):
    sem,cases=descriptors(v)
    inventory={r["configuration"]:r["tests"] for r in json.loads((v/"inventory.json").read_text(encoding="utf8"))}
    for group,members in sem["groups"].items():
        for identity in members:
            config,failed=identity.split("/")
            others=[k for k in members if k.split("/")[0]!=config]
            if not others:
                continue
            query=cases[others[0]]
            target=cases[identity]
            folder=source/"variants"/config/"coverage"
            fa=coverage(folder/"failed"/(failed+".coverage.xml"))
            passed=inventory[config]["passed"]
            for name in passed:
                key=config+"/"+name
                control=cases[key]
                if target[0]!=control[0] or target[1]==control[1]:
                    continue
                fb=coverage(folder/"passed"/(name+".coverage.xml"))
                similarity=len(fa&fb)/max(1,len(fa|fb))
                if similarity<0.8:
                    continue
                common=[cases[config+"/"+n] for n in passed if n!=name][:3]
                yield (group,identity,key,similarity,encode(query,common+[target]),encode(query,common+[control]))
                break


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("input","data","models","output"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    cache={}
    with a.output.open("w",newline="",encoding="utf8") as f:
        w=csv.writer(f)
        w.writerow(["split","system","version","group","detected_test","missed_test","coverage_jaccard","p_detected","p_missed"])
        for split,tr,te in splits(a.input):
            net=DetectionNet()
            net.load_state_dict(torch.load(a.models/split/"model.pt",weights_only=True,map_location="cpu"))
            net.eval()
            for v in te:
                sys=v.parent.name
                if v not in cache:
                    cache[v]=list(check(v,a.data.joinpath(*SYSTEMS[sys],v.name)))
                for g,left,right,sim,found,miss in cache[v]:
                    probs=predict(net,[found,miss])
                    w.writerow([split,sys,v.name,g,left,right,sim,*probs])
