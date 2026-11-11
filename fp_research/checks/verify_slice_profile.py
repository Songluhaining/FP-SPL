"""在真实配置上逐键逐集合比较加速入口与原切片输出。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from fp_research.case_study import ROOT,JAVA,SYSTEMS
from fp_research.slice_reuse import signatures
from fp_research.inventory import atomic_json
from fp_research.slice_runner import compiled_runner


def run(system,version,configuration,indexed=False,adjacency=False):
    base=Path("D:/dataset/whn").joinpath(*SYSTEMS[system])
    original=base/version/"variants"/configuration
    root=base/version/(".fp_research/profile-adjacency" if adjacency else ".fp_research/profile-indexed" if indexed else ".fp_research/profile-check")/configuration
    product=root/"variants"/configuration
    product.mkdir(parents=True,exist_ok=True)
    shutil.copytree(original/"src",product/"src",dirs_exist_ok=True)
    (product/"coverage").mkdir(exist_ok=True)
    for p in (original/"coverage").glob("spectrum_*_coverage.xml"):shutil.copyfile(p,product/"coverage"/p.name)
    jar=ROOT.parent/"PULP/plugins/all-statements-slicing.jar"
    raw,_=signatures(original)
    key=hashlib.sha256((hashlib.sha256(jar.read_bytes()).hexdigest()+raw).encode()).hexdigest()
    cached=base/".fp_research/slice-cache"/key
    if not (cached/"complete.json").exists():raise ValueError("缺少原切片参考结果")
    started=time.monotonic()
    runner_hash=hashlib.sha256((ROOT/"fp_research/checks/SliceProfile.java").read_bytes()).hexdigest()
    runner,runner_key=compiled_runner()
    binary_hash=hashlib.sha256((runner/"SliceProfile.class").read_bytes()).hexdigest()
    with (root/"run.log").open("w") as log:
        subprocess.run([JAVA/"bin/java.exe","-Xmx1536m","-XX:+UseSerialGC","-Dskip.unexecuted=true",
                        "-Dindexed.propagation="+str(indexed).lower(),
                        "-Dcached.adjacency="+str(adjacency).lower(),
                        "-Djava.io.tmpdir="+str(root),"-Dvariants_dir="+str(root/"variants"),"-cp",
                        str(runner)+";"+str(jar),"SliceProfile",str(product/"src")],
                       cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
    counts={}
    for direction in ("forward","backward"):
        name=f"slice.all_statements.{direction}_slicing.log"
        def read(path):return {k:set(v) for k,v in json.loads(path.read_text()).items()}
        old=read(cached/name); new=read(product/name)
        if old!=new:raise ValueError(f"切片集合发生改变 {direction}")
        counts[direction]=len(old)
    atomic_json(root/"verified.json",dict(system=system,version=version,configuration=configuration,cache=key,
                seconds=time.monotonic()-started,counts=counts,runner_sha256=runner_hash,binary_sha256=binary_hash,
                runner_key=runner_key,indexed=indexed,adjacency=adjacency))
    print(system,version,configuration,"exact sets verified",counts,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("system");p.add_argument("version");p.add_argument("configuration")
    p.add_argument("--indexed",action="store_true")
    p.add_argument("--adjacency",action="store_true")
    a=p.parse_args();run(a.system,a.version,a.configuration,a.indexed,a.adjacency)
