"""核对测试框架的异常替身，只恢复发生在相同源码位置的标准异常。"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from .inventory import SYSTEMS,atomic_json
from .runtime import LIB
from .traced_graphs import Engine,jobs

MOCK="org.evosuite.runtime.mock.java.lang.MockIllegalArgumentException"
STANDARD="java.lang.IllegalArgumentException"


def translated(reason):
    marker=" 失效位置在插桩后改变 "
    if marker not in reason:
        return None
    identity,detail=reason.split(marker,1)
    before,after=map(ast.literal_eval,detail.split(" -> ",1))
    if not isinstance(before,list) or not isinstance(after,list) or len(before)!=3 or len(after)!=3:
        return None
    if before[0]==MOCK and after[0]==STANDARD and before[1:]==after[1:]:
        return identity,before,after
    return None


def recover(engine,job):
    version,out,group,members,pairs,sites=job
    path=out/"graphs"/(group+".json")
    record=json.loads(path.read_text(encoding="utf8"))
    evidence=[]
    adjusted={}
    while record["status"]=="error":
        change=translated(record.get("reason",""))
        if change is None:
            break
        identity,before,after=change
        if identity in adjusted or identity not in members or sites[identity]["site"]!=before:
            break
        adjusted[identity]=dict(sites[identity],site=after)
        evidence.append(dict(test=identity,ordinary=before,validated_equivalent=after))
        engine.infer((version,out,group,members,pairs,{**sites,**adjusted}))
        record=json.loads(path.read_text(encoding="utf8"))
    if evidence:
        record["verified_exception_wrappers"]=evidence
        atomic_json(path,record)
    return bool(evidence)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","input","cache","java-home"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--workers",type=int,default=6)
    a=p.parse_args()
    home=a.java_home.resolve()
    declaration=subprocess.check_output([home/"bin/javap.exe","-classpath",LIB,MOCK],text=True)
    if f"{MOCK} extends {STANDARD} implements org.evosuite.runtime.mock.OverrideMock" not in declaration:
        raise RuntimeError("当前依赖的异常替身声明与预期不符")
    engine=Engine(home,a.cache.resolve())
    selected=[]
    for job in jobs(a.data.resolve(),a.input.resolve()):
        record=json.loads((job[1]/"graphs"/(job[2]+".json")).read_text(encoding="utf8"))
        if record["status"]=="error" and translated(record.get("reason","")):
            selected.append(job)
    print("targeted",len(selected),"groups",flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        print("rechecked",sum(pool.map(lambda job:recover(engine,job),selected)),flush=True)
