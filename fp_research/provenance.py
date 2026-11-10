"""核验静态切片所属构建、基准实现文件与每个缺陷版本的实验清单。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .inventory import SYSTEMS
from .risk import save_rows


def digest_tree(folder,pattern):
    h=hashlib.sha256()
    files=sorted(folder.rglob(pattern))
    for f in files:
        h.update(f.relative_to(folder).as_posix().encode()+b"\0")
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest(),len(files)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","input","workspace","output"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for system,relative in SYSTEMS.items():
        staged=a.workspace/("pulp_trial" if system=="bank" else "email_trial")/"dataset"
        for v in sorted(a.data.joinpath(*relative).glob("_MultipleBugs_*")):
            src_hash=hashlib.sha256()
            slices_hash=hashlib.sha256()
            manifest=a.input/system/v.name/"inventory.json"
            configs=json.loads(manifest.read_text(encoding="utf8"))
            count=0
            for r in configs:
                key=r["configuration"]
                source=v/"variants"/key/"src"
                prior=staged/v.name/"variants"/key
                left,n=digest_tree(source,"*.java")
                right,_=digest_tree(prior/"src","*.java")
                if not n or left!=right:
                    raise ValueError("源码树与静态切片来源不匹配："+str(source))
                src_hash.update((key+left).encode())
                for f in sorted(prior.glob("slice.all_statements.*_slicing.log")):
                    slices_hash.update(key.encode()+f.name.encode()+hashlib.sha256(f.read_bytes()).digest())
                    count+=1
            if count!=2*len(configs):
                raise ValueError(f"切片文件不齐 {v}: {count}")
            rows.append(dict(system=system,version=v.name,configurations=len(configs),
                             passed=sum(r["outcome"]=="passed" for r in configs),failed=sum(r["outcome"]=="failed" for r in configs),
                             failed_tests=sum(len(r["tests"]["failed"]) for r in configs),
                             passed_tests=sum(len(r["tests"]["passed"]) for r in configs),
                             source_sha256=src_hash.hexdigest(),slice_sha256=slices_hash.hexdigest(),slice_files=count,
                             report_sha256=hashlib.sha256((v/"config.report.csv.done").read_bytes()).hexdigest(),
                             inventory_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
    save_rows(a.output/"versions.csv",rows)
    inputs=[]
    for n in ("CLAP","PULP"):
        root=a.workspace/n
        for f in sorted(root.rglob("*.py")):
            if any(v in ("venv","__pycache__",".venv") for v in f.relative_to(root).parts):
                continue
            inputs.append(dict(method=n,path=f.relative_to(root).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    save_rows(a.output/"baseline-source-hashes.csv",inputs)
    print("verified",len(rows),"versions,",sum(r["slice_files"] for r in rows),"slice files")
