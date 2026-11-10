"""对每个已完成建图的版本立即选例和复核，避免等待其他缺陷版本。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .folds import splits
from .inventory import SYSTEMS,atomic_json
from .probes import plan_version
from .replay import replay_version
from .risk import score_version


def complete(job):
    v,source,models,name,plans,projects,tool,home=job
    rows=[r for r in score_version(v,source,models/name,name) if r["method"]=="full"]
    path=plans/v.parent.name/v.name/"probe-plan.json"
    outcomes=plan_version(v,source,models,rows)
    atomic_json(path,outcomes)
    replay_version((source,path,v/"inventory.json",projects,tool,plans/"build",home))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","input","models","plans","projects","tool","java-home"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--system",choices=SYSTEMS,required=True)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    jobs=[]
    for name,_,test in splits(a.input.resolve()):
        if name.startswith(a.system+"-fold"):
            for v in test:
                groups=json.loads((v/"semantics.json").read_text(encoding="utf8"))["groups"]
                if all((v/"graphs"/(g+".json")).exists() for g in groups):
                    jobs.append((v,a.data.joinpath(*SYSTEMS[a.system],v.name).resolve(),a.models.resolve(),
                                 name,a.plans.resolve(),a.projects.resolve(),a.tool.resolve(),a.java_home.resolve()))
    print("ready versions",len(jobs),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        list(executor.map(complete,jobs))
