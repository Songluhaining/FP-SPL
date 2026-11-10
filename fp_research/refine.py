"""用普通执行的失败位置校验语义分组，并复用成员及输入指纹均相同的图。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

from .graphs import process_version
from .inventory import SYSTEMS,atomic_json
from .sites import refine


def metadata(source,dest,sites):
    sem=json.loads((source/"semantics.json").read_text(encoding="utf8"))
    records={}
    configs={k.split("/")[0] for k in sem["pairs"]}
    for c in sorted(configs):
        # 转换为通过的配置不会进入此集合；它们原来失败的执行始终不可见。
        document=json.loads((sites/(c+".json")).read_text(encoding="utf8"))
        records.update({c+"/"+k:value for k,value in document["tests"].items() if c+"/"+k in sem["pairs"]})
    if set(records)!=set(sem["pairs"]):
        raise ValueError("失效位置核验遗漏用例："+str(source))
    before=len(sem["groups"])
    groups=refine(sem,records)
    dest.mkdir(parents=True,exist_ok=True)
    if source.resolve()!=dest.resolve():
        for file in ("inventory.json","construction.json"):
            if (source/file).exists():
                shutil.copyfile(source/file,dest/file)
    sem["groups"]=groups
    sem["locations"]=records
    sem["groups_before_locations"]=sem.get("groups_before_locations",before)
    atomic_json(dest/"semantics.json",sem)
    return groups


def verify_one(job):
    source,dest,locations,data,cache,home,stage=job
    if stage=="metadata":
        groups=metadata(source,dest,locations)
    else:
        groups=json.loads((dest/"semantics.json").read_text(encoding="utf8"))["groups"]
        folder=dest/"graphs"
        folder.mkdir(exist_ok=True)
        for g in groups:
            old=source/"graphs"/(g+".json")
            if old.exists() and not (folder/old.name).exists():
                shutil.copyfile(old,folder/old.name)
        process_version((data,dest,cache,home,None))
    print(stage,dest.parent.name,dest.name,len(groups),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("input","output","sites","data","runtime","java-home"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--stage",choices=["metadata","graphs"],required=True)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--system",choices=SYSTEMS)
    a=p.parse_args()
    jobs=[]
    for s in [a.system] if a.system else SYSTEMS:
        for v in sorted((a.input/s).glob("_MultipleBugs_*")):
            jobs.append((v.resolve(),(a.output/s/v.name).resolve(),a.sites/s/v.name,
                         a.data.joinpath(*SYSTEMS[s],v.name).resolve(),a.runtime.resolve(),a.java_home.resolve(),a.stage))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(verify_one,jobs))
