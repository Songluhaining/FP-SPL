"""对已核验的可见数据重新建图、重建模型输入，使原失败证据无法进入转换配置。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import shutil

from .graphs import process_version
from .inventory import SYSTEMS,atomic_json,records
from .intents import version_intents
from .pairing import pair_cases


def prepare_one(job):
    version,out,main,cache,home,stage=job
    rows=list(records(version))
    if not all(r["coverage_agrees"] for r in rows):
        raise ValueError("移除失败执行后配置状态冲突："+str(version))
    if stage=="semantics":
        atomic_json(out/"inventory.json",rows)
        cases,groups=version_intents(version,rows)
        pairs=pair_cases(cases,rows)
        atomic_json(out/"semantics.json",dict(cases=cases,groups=groups,pairs=pairs))
    else:
        groups=json.loads((out/"semantics.json").read_text(encoding="utf8"))["groups"]
        (out/"graphs").mkdir(exist_ok=True)
        for group in groups:
            src=main/"graphs"/(group+".json")
            target=out/"graphs"/src.name
            if src.exists() and not target.exists():
                shutil.copyfile(src,target)
        # process_version逐组校验执行成员、源码、映射、代理版本与参数；仅精确命中时复用。
        process_version((version,out,cache,home,None))
    print(stage,version.name,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","original-results","output","java-home"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--stage",choices=["semantics","graphs"],required=True)
    p.add_argument("--system",choices=SYSTEMS)
    p.add_argument("--workers",type=int,default=3)
    a=p.parse_args()
    jobs=[]
    for sys in [a.system] if a.system else SYSTEMS:
        for folder in sorted(a.data.joinpath(*SYSTEMS[sys]).glob("_MultipleBugs_*")):
            if not (folder/"config.report.csv.done").is_file():
                raise ValueError("转换仍未通过核验："+str(folder))
            jobs.append((folder.resolve(),a.output.resolve()/sys/folder.name,a.original_results.resolve()/sys/folder.name,
                         a.original_results.resolve()/"runtime",a.java_home.resolve(),a.stage))
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        list(executor.map(prepare_one,jobs))
