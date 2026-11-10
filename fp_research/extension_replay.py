"""在全部原本通过的配置上使用相同补测规则，独立复核三种识别方法。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from .extension import ROOT, SYSTEMS, versions
from .inventory import atomic_json
from .replay import replay_version
from .test_sources import PACKAGE, methods


def bodies(variant):
    for file in sorted((variant/"test").rglob("*Test.java")):
        source=file.read_text(encoding="utf-8-sig")
        pkg=PACKAGE.search(source)
        cls=(pkg[1]+"." if pkg else "")+file.stem
        for name,start,end in methods(source):
            body=source[start:end]
            yield cls+"."+name,(cls,body[body.index("{"):])


def run(job):
    s,version,work,projects,java=job
    root=work/"main"/s/version.name
    inventory=root/"inventory.json"
    rows=json.loads(inventory.read_text(encoding="utf8"))
    failures=[]
    for r in rows:
        if r["outcome"]=="failed":
            cases=dict(bodies(version/"variants"/r["configuration"]))
            for test in r["tests"]["failed"]:
                key=r["configuration"]+"/"+test
                failures.append((key,cases[test],set(r["features"])))
    plan={}
    for r in rows:
        if r["outcome"]!="passed":
            continue
        cid=r["configuration"]
        seen={body for _,body in bodies(version/"variants"/cid)}
        selected=set(r["features"])
        def order(candidate):
            key,_,features=candidate
            tie=hashlib.sha256(("/".join(("42",s,version.name,cid,key))).encode()).digest()
            return (-len(selected & features)/len(selected | features),tie)
        chosen=[]
        for key,body,_ in sorted(failures,key=order):
            if body in seen:
                continue
            chosen.append(key)
            seen.add(body)
            if len(chosen)==5:
                break
        plan[cid]={"coverage":chosen}
    target=root/"replay-plan.json"
    atomic_json(target,plan)
    replay_version((version,target,inventory,projects,projects.parent/"plugins/featurehouse_20210219.jar",
                    work/"replay-cache",java))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("D:/dataset/whn"))
    parser.add_argument("--work",type=Path,default=Path("D:/dataset/whn/fp_extension_work"))
    parser.add_argument("--projects",type=Path,default=ROOT.parent/"CLAP/projects")
    parser.add_argument("--java-home",type=Path,default=Path("E:/Program Files/Java/jdk-1.8"))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--system",choices=SYSTEMS)
    parser.add_argument("--version")
    a=parser.parse_args()
    jobs=[(s,v,a.work.resolve(),a.projects.resolve(),a.java_home.resolve())
          for s in ([a.system] if a.system else SYSTEMS) for v in versions(a.data.resolve(),s)
          if not a.version or v.name==a.version]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run,jobs))
