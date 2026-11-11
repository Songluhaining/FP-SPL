"""按原切片工具实际读取的源码与正执行语句集合登记等价输入。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from .case_study import ROOT,SYSTEMS,case_versions
from .inventory import atomic_json


def signatures(variant):
    source=hashlib.sha256(); raw=hashlib.sha256(); executed=set()
    for p in sorted((variant/"src").rglob("*.java")):
        item=p.relative_to(variant).as_posix().encode()+b"\0"+hashlib.sha256(p.read_bytes()).digest()
        source.update(item); raw.update(item)
    for p in sorted((variant/"coverage").glob("spectrum_*_coverage.xml")):
        payload=p.read_bytes()
        raw.update(p.relative_to(variant).as_posix().encode()+b"\0"+hashlib.sha256(payload).digest())
        if p.name in {"spectrum_failed_coverage.xml","spectrum_passed_coverage.xml"}:
            for line in ET.fromstring(payload).iter("line"):
                if "count" in line.attrib and int(line.get("count"))>0:
                    executed.add(line.get("featureClass","")+"."+line.get("featureLineNum",""))
    canonical=source.hexdigest()+json.dumps(sorted(executed),ensure_ascii=True,separators=(",",":"))
    return raw.hexdigest(),hashlib.sha256(canonical.encode()).hexdigest()


def run(system,data):
    jar=hashlib.sha256((ROOT.parent/"PULP/plugins/all-statements-slicing.jar").read_bytes()).hexdigest()
    variants=[p for v in case_versions(data,system) for p in sorted((v/"variants").iterdir())]
    cache=data.joinpath(*SYSTEMS[system],".fp_research/slice-cache")
    entries=[]; representatives={}; duplicates=0
    def one(v):
        raw,canonical=signatures(v)
        key=hashlib.sha256((jar+raw).encode()).hexdigest()
        return dict(variant=str(v),raw=key,canonical=canonical,completed=(cache/key/"complete.json").exists())
    with ThreadPoolExecutor(max_workers=8) as pool:
        for item in pool.map(one,variants):
            entries.append(item)
            if item["completed"]:
                existing=representatives.get(item["canonical"])
                if existing and existing!=item["raw"]:
                    for name in ("forward","backward"):
                        filename=f"slice.all_statements.{name}_slicing.log"
                        def read(key):return {k:set(v) for k,v in json.loads((cache/key/filename).read_text()).items()}
                        if read(existing)!=read(item["raw"]):raise ValueError("等价输入的既有切片不同")
                    duplicates+=1
                else:representatives[item["canonical"]]=item["raw"]
    target=cache.parent/"slice-signatures.json"
    atomic_json(target,dict(jar=jar,entries=entries,representatives=representatives,duplicates_verified=duplicates))
    print(system,"products",len(entries),"exact",len({r["raw"] for r in entries}),"semantic",len({r["canonical"] for r in entries}),
          "existing",len(representatives),"duplicates verified",duplicates,flush=True)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--system",choices=SYSTEMS,required=True)
    p.add_argument("--data",type=Path,default=Path("D:/dataset/whn"))
    a=p.parse_args();run(a.system,a.data)
