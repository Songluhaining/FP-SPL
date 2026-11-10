"""为同一配置生成三种策略共享的补测候选池，并输出每种策略的前五项。"""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET
import numpy as np
import torch

from .applicability import build_evidence
from .detection import DetectionNet, encode, predict
from .feature_sources import load_feature_map
from .fit import descriptors
from .inventory import SYSTEMS,atomic_json
from .test_sources import methods


@lru_cache(maxsize=4096)
def method_source(variant,identity):
    cls,method=identity.rsplit(".",1)
    paths=list((variant/"test").rglob(cls.split(".")[-1]+".java"))
    if len(paths)!=1:
        raise ValueError(f"测试类不唯一 {identity}")
    source=paths[0].read_text(encoding="utf-8-sig")
    spans={name:(a,b) for name,a,b in methods(source)}
    a,b=spans[method]
    return paths[0],source,source[a:b]


def donor_coverage(folder,name,mapping):
    report=folder/"coverage/failed"/(name+".coverage.xml")
    reached=set()
    for pkg in ET.parse(report).iterfind(".//package"):
        for f in pkg.findall("file"):
            cname=pkg.get("name")+"."+Path(f.get("path")).stem
            for line in f.findall("line"):
                if int(line.get("count",0))>0:
                    key=mapping.get((cname,int(line.get("num"))))
                    if key:
                        reached.add(key)
    return reached


def plan_version(vdir,source,models,rows):
    sem,cases=descriptors(vdir)
    inventory={r["configuration"]:r for r in json.loads((vdir/"inventory.json").read_text(encoding="utf8"))}
    api={}
    for cfg in inventory:
        files=list((source/"variants"/cfg/"src").rglob("*.java"))
        api[cfg]=({f.stem for f in files},set().union(*[set(re.findall(
            r"\b([A-Za-z_$][\w$]*)\s*\([^;{}]*\)\s*(?:throws\s+[\w., ]+)?\{",
            f.read_text(encoding="utf-8-sig",errors="replace"))) for f in files]))
    candidates={}
    mapped={}
    for key in sem["pairs"]:
        donor,test=key.split("/")
        folder=source/"variants"/donor
        _,_,body=method_source(folder,test)
        if donor not in mapped:
            mapped[donor]=load_feature_map(folder)
        candidates[key]=(body,donor_coverage(folder,test,mapped[donor]))
    net=DetectionNet()
    net.load_state_dict(torch.load(models/rows[0]["split"]/"model.pt",map_location="cpu",weights_only=True))
    net.eval()
    torch.set_num_threads(1)
    result={}
    for row in rows:
        name=row["configuration"]
        group=row["group"]
        rankings={}
        result[name]=rankings
        if not group:
            continue
        reference=cases[sem["groups"][group][0]]
        cfg=inventory[name]
        target=source/"variants"/name
        present,covered=build_evidence(target)
        original=set()
        for id_ in cfg["tests"]["passed"]:
            _,_,body=method_source(target,id_)
            original.add(body[body.index("{"):].strip())
        suite=encode(reference,[cases[name+"/"+id_] for id_ in cfg["tests"]["passed"]])
        unique={}
        for key,(body,coverage) in candidates.items():
            canonical=body[body.index("{"):].strip()
            if canonical in original or not cases[key][0] & reference[0]:
                continue
            # 同体测试只保留一个候选。相同源码类与不同脚手架仍需分别检查。
            donor,test=key.split("/")
            donor_types,donor_methods=api[donor]
            target_types,target_methods=api[name]
            if (set(re.findall(r"\b[A-Z][\w$]*\b",body)) & donor_types)-target_types:
                continue
            if (cases[key][0] & donor_methods)-target_methods:
                continue
            cls=test.rsplit(".",1)[0]
            scaffold=list((source/"variants"/donor/"test").rglob(cls.split(".")[-1]+"_scaffolding.java"))
            scaffold_hash=hashlib.sha256(scaffold[0].read_bytes()).digest() if scaffold else b""
            unique.setdefault((cls,canonical,scaffold_hash),key)
        keys=sorted(unique.values())
        if not keys:
            continue
        base=float(predict(net,[suite])[0])
        gains=[]
        for start in range(0,len(keys),64):
            augmented=[np.vstack([suite,encode(reference,[cases[k]])]) for k in keys[start:start+64]]
            gains.extend(predict(net,augmented)-base)
        random_order=keys.copy()
        random.Random("42/"+vdir.parent.name+"/"+vdir.name+"/"+name).shuffle(random_order)
        rankings.update(pool=len(keys),model=[keys[i] for i in sorted(range(len(keys)),key=lambda i:(-gains[i],keys[i]))[:5]],
                        random=random_order[:5],coverage=sorted(keys,key=lambda k:(-len((candidates[k][1] & present)-covered),k))[:5])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("risk","input","data","models","output"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    groups={}
    with a.risk.open(encoding="utf8") as s:
        for r in csv.DictReader(s):
            if r["method"]=="full" and "-fold" in r["split"]:
                groups.setdefault((r["system"],r["version"]),[]).append(r)
    for (system,v),rows in groups.items():
        result=plan_version(a.input/system/v,a.data.joinpath(*SYSTEMS[system],v),a.models,rows)
        atomic_json(a.output/system/v/"probe-plan.json",result)
        print(system,v,len(result),flush=True)
