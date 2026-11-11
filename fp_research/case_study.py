"""在原案例目录下完成新增系统实验，原始输入只读，工作视图仅作临时文件。"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from .inventory import records, atomic_json
from .latent_fault import infer, read_version
from .checks.rebuild_latent import verify_version
from .baseline_features import worker
from .risk import save_rows
from .supplement import version_stage
from .audit_evidence import check_masked

ROOT=Path(__file__).resolve().parents[1]
SYSTEMS={"gpl":("GPL","4wise-GPL-1BUG-Full"),"zipme":("ZipMe","4wise-ZipMe-1BUG-Full")}
JAVA=Path("E:/Program Files/Java/jdk-1.8")
_LOCKS={}
_GUARD=threading.Lock()
_ALIASES={}
_AUDIT_POOL=None
_SLICE_WORKERS=int(os.environ.get("FP_SLICE_WORKERS","4"))
_SLICE_TIMEOUT=int(os.environ.get("FP_SLICE_TIMEOUT","7200"))
_SLICE_BACKEND=os.environ.get("FP_SLICE_BACKEND","original")
_SLICE_HEAP_MB=int(os.environ.get("FP_SLICE_HEAP_MB","1536"))
_SLICE_SLOTS=threading.BoundedSemaphore(_SLICE_WORKERS)


def case_versions(data,system):
    return sorted(data.joinpath(*SYSTEMS[system]).glob("_MultipleBugs_.NOB_1.ID_*"))


def results(version,scenario):
    return version/".fp_research"/scenario


def raw_digest(version):
    h=hashlib.sha256()
    paths=[]
    pending=[version]
    while pending:
        folder=pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_dir():
                    if entry.name!=".fp_research":pending.append(Path(entry.path))
                else:
                    # Windows目录枚举已有这些属性，无需逐文件再次查询。
                    paths.append((Path(entry.path),entry.stat()))
    for p,stat in sorted(paths,key=lambda pair:pair[0]):
        rel=p.relative_to(version)
        h.update((rel.as_posix()+str(stat.st_size)+str(stat.st_mtime_ns)).encode())
        if (rel.parts[0] in {"features"} or p.name.startswith("config.report")
            or "src" in rel.parts or "test" in rel.parts or p.name.startswith("spectrum_")):
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def slices(source,output,cache):
    from .slice_reuse import signatures
    from .slice_runner import compiled_runner
    jar=ROOT.parent/"PULP/plugins/all-statements-slicing.jar"
    jar_hash=hashlib.sha256(jar.read_bytes()).hexdigest()
    builds=output/"slices/variants"
    names=[f"slice.all_statements.{d}_slicing.log" for d in ("forward","backward")]
    references={}
    runner,runner_key=compiled_runner() if _SLICE_BACKEND=="indexed" else (None,jar_hash)
    with _GUARD:
        if cache not in _ALIASES:
            mapping={}
            index=cache.parent/"slice-signatures.json"
            if index.exists():
                contents=json.loads(index.read_text(encoding="utf8"))
                if contents["jar"]!=jar_hash:raise ValueError("切片索引的工具版本不符")
                for item in contents["entries"]:
                    if (cache/item["raw"]/"complete.json").exists():mapping[item["canonical"]]=item["raw"]
            _ALIASES[cache]=mapping
        aliases=_ALIASES[cache]
    # 同一源码和可见频谱只切片一次，切片不读取缺陷日志或标签。
    def one(variant):
        raw,canonical=signatures(variant)
        original_key=hashlib.sha256((jar_hash+raw).encode()).hexdigest()
        with _GUARD:
            lock=_LOCKS.setdefault((str(cache),canonical),threading.Lock())
        with lock, _SLICE_SLOTS:
            key=aliases.get(canonical,original_key)
            target=cache/key
            if not (target/"complete.json").exists():
                target.mkdir(parents=True,exist_ok=True)
                with tempfile.TemporaryDirectory(dir=target) as temp:
                    stage=Path(temp)
                    product=stage/"variants"/variant.name
                    shutil.copytree(variant/"src",product/"src")
                    (product/"coverage").mkdir()
                    for p in (variant/"coverage").glob("spectrum_*_coverage.xml"):
                        shutil.copyfile(p,product/"coverage"/p.name)
                    with (target/"slicing.log").open("w") as log:
                        command=[JAVA/"bin/java.exe","-Xmx1024m","-XX:+UseSerialGC",
                                 "-Djava.io.tmpdir="+str(stage),"-Dvariants_dir="+str(stage/"variants"),"-jar",jar]
                        if _SLICE_BACKEND=="indexed":
                            command=[JAVA/"bin/java.exe",f"-Xmx{_SLICE_HEAP_MB}m","-XX:+UseSerialGC",
                                     "-Djava.io.tmpdir="+str(stage),"-Dvariants_dir="+str(stage/"variants"),
                                     "-Dskip.unexecuted=true","-Dindexed.propagation=true","-cp",
                                     str(runner)+os.pathsep+str(jar),"SliceProfile",str(product/"src")]
                        elif _SLICE_BACKEND!="original":raise ValueError("未知切片后端")
                        subprocess.run(command,cwd=stage,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=_SLICE_TIMEOUT)
                    hashes={}
                    for name in names:
                        payload=(product/name).read_bytes()
                        if not isinstance(json.loads(payload),dict):
                            raise ValueError("切片格式异常")
                        (target/name).write_bytes(payload)
                        hashes[name]=hashlib.sha256(payload).hexdigest()
                    atomic_json(target/"complete.json",dict(key=key,files=hashes,backend=_SLICE_BACKEND,
                                runner_sha256=runner_key))
            aliases[canonical]=key
            references[variant.name]=target
            checked=json.loads((target/"complete.json").read_text(encoding="utf8"))
            dest=builds/variant.name
            dest.mkdir(parents=True,exist_ok=True)
            for name in names:
                payload=(target/name).read_bytes()
                if hashlib.sha256(payload).hexdigest()!=checked["files"][name]:
                    raise ValueError("切片缓存摘要不符")
                temporary=dest/(name+".new")
                if temporary.exists():
                    temporary.unlink()
                try:
                    os.link(target/name,temporary)
                except OSError:
                    temporary.write_bytes(payload)
                os.replace(temporary,dest/name)
    with ThreadPoolExecutor(max_workers=_SLICE_WORKERS) as pool:
        list(pool.map(one,sorted((source/"variants").iterdir())))
    atomic_json(output/"slice-cache.json",{k:str(v) for k,v in references.items()})


def observe(system,source,out):
    rows=list(records(source))
    if any(not r["source"] or not r["test_source"] or not r["coverage_agrees"] for r in rows):
        raise ValueError(f"原始观测不完整 {source}")
    atomic_json(out/"inventory.json",rows)
    expected=[dict(system=system,**r) for r in infer(read_version(source),source.name)]
    actual,audit=(_AUDIT_POOL.submit(verify_version,(system,source)).result()
                  if _AUDIT_POOL is not None else verify_version((system,source)))
    if actual!=expected or audit["changed_coverage_counts"]:
        raise ValueError(f"逐用例复算不一致 {source}")
    save_rows(out/"latent.csv",actual)
    atomic_json(out/"coverage-audit.json",audit)
    return rows


def checked_observation_cache(system,source,out,before):
    """输入未变且当前推断逐值相等时，保留已经完成的逐用例覆盖核验。"""
    marker=out/"observed.json"
    if not marker.exists() or json.loads(marker.read_text(encoding="utf8")).get("raw_digest")!=before:
        return None
    rows=json.loads((out/"inventory.json").read_text(encoding="utf8"))
    audit=json.loads((out/"coverage-audit.json").read_text(encoding="utf8"))
    if audit["changed_coverage_counts"] or audit["tests"]!=sum(len(ts) for r in rows for ts in r["tests"].values()):
        raise ValueError("既有逐用例覆盖核验凭据不符")
    with (out/"latent.csv").open(encoding="utf-8-sig",newline="") as stream:
        saved=list(csv.DictReader(stream))
    expected=[dict(system=system,**r) for r in infer(read_version(source),source.name)]
    if len(saved)!=len(expected):raise ValueError("既有推断结果行数不符")
    for left,right in zip(saved,expected):
        if set(left)!=set(right):raise ValueError("既有推断列不符")
        for key,value in right.items():
            actual=float(left[key]) if isinstance(value,(int,float)) else left[key]
            if actual!=value:raise ValueError("当前推断与既有核验结果不一致")
    return rows


def features(source,out,method):
    env=os.environ.copy()
    env.update(TEMP=str(out),TMP=str(out),PYTHONUTF8="1",OPENBLAS_NUM_THREADS="1")
    args=[sys.executable,"-m","fp_research.case_study","worker","--source",str(source),
          "--output",str(out),"--method",method]
    with (out/(method+"-run.log")).open("w",encoding="utf8") as log:
        subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def process_case(job):
    system,version,scenario,only_observe=job
    out=results(version,scenario)
    out.mkdir(parents=True,exist_ok=True)
    before=raw_digest(version)
    code=hashlib.sha256()
    for name in ("case_study.py","baseline_features.py","baseline_cache.py","latent_fault.py","supplement.py","test_environment.py","checks/rebuild_latent.py"):
        code.update((ROOT/"fp_research"/name).read_bytes())
    fingerprint=hashlib.sha256((before+code.hexdigest()+scenario).encode()).hexdigest()
    stamp=out/("observed.json" if only_observe else "complete.json")
    if stamp.exists() and json.loads(stamp.read_text(encoding="utf8"))["fingerprint"]==fingerprint:
        return system,version.name,"cached"
    started=time.monotonic()
    cache=version.parent/".fp_research"/"slice-cache"
    with tempfile.TemporaryDirectory(dir=out) as temp:
        source=version
        if scenario=="masked":
            rows=list(records(version))
            failed=sorted(r["configuration"] for r in rows if r["outcome"]=="failed")
            selection=out/"selection.csv"
            save_rows(selection,[{"VARIANT":r["configuration"],"FP transformed from F":int(r["configuration"] in failed[1::2])}
                                 for r in rows])
            if failed[1::2]:
                source=Path(temp)/version.name
                version_stage((version,source,selection,out/"construction.json",JAVA))
            else:
                atomic_json(out/"construction.json",{})
        observed=out/"observed.json"
        reusable=checked_observation_cache(system,source,out,before) if scenario=="main" else None
        if reusable is not None:
            rows=reusable
        elif observed.exists() and json.loads(observed.read_text(encoding="utf8"))["fingerprint"]==fingerprint:
            rows=json.loads((out/"inventory.json").read_text(encoding="utf8"))
        else:
            rows=observe(system,source,out)
            atomic_json(observed,dict(fingerprint=fingerprint,raw_digest=before))
        if scenario=="masked":
            check_masked(version,source,{r["configuration"]:r for r in rows},
                         json.loads((out/"construction.json").read_text(encoding="utf8")))
        if not only_observe:
            slices(source,out,cache)
            for method in ("CLAP","PULP"):
                features(source,out,method)
    if raw_digest(version)!=before:
        raise ValueError(f"原始输入发生变动 {version}")
    atomic_json(stamp,dict(fingerprint=fingerprint,raw_digest=before,seconds=time.monotonic()-started))
    if not only_observe:
        (out/"failure.json").unlink(missing_ok=True)
    return system,version.name,"verified"


def main():
    global _AUDIT_POOL
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["run","observe","worker"])
    p.add_argument("--scenario",choices=["main","masked"],default="main")
    p.add_argument("--data",type=Path,default=Path("D:/dataset/whn"))
    p.add_argument("--system",choices=SYSTEMS)
    p.add_argument("--version")
    p.add_argument("--exclude-version",action="append",default=[],help="排除已由其他工作进程独立处理的版本")
    p.add_argument("--pending-only",action="store_true",help="仅续跑无完成凭据的案例，最终评价再次核对输入摘要")
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--source",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--method",choices=["CLAP","PULP"])
    a=p.parse_args()
    if a.stage=="worker":
        worker((ROOT.parent/a.method,a.method,a.source.resolve(),a.output.resolve()/"slices",a.output.resolve()/a.method),
               optimize=True)
        return
    jobs=[(s,v,a.scenario,a.stage=="observe") for s in ([a.system] if a.system else SYSTEMS)
          for v in case_versions(a.data.resolve(),s) if (not a.version or v.name==a.version) and v.name not in a.exclude_version
          and (not a.pending_only or not (results(v,a.scenario)/"complete.json").exists())]
    if not jobs:
        if a.pending_only:
            print("所选版本均已有完成记录；最终评价仍会核对输入摘要",flush=True)
            return
        raise ValueError("没有找到选定版本")
    failures=[]
    with ProcessPoolExecutor(max_workers=min(4,a.workers)) as auditing, ThreadPoolExecutor(max_workers=a.workers) as pool:
        _AUDIT_POOL=auditing
        tasks={pool.submit(process_case,j):j for j in jobs}
        for future in as_completed(tasks):
            job=tasks[future]
            try:
                print(*future.result(),flush=True)
            except Exception as ex:
                root=results(job[1],a.scenario)
                atomic_json(root/"failure.json",dict(type=type(ex).__name__,reason=str(ex)))
                failures.append((job[0],job[1].name,str(ex)))
                print("FAILED",failures[-1],flush=True)
    if failures: raise RuntimeError(f"{len(failures)} 个版本需要重试，详情见各案例 failure.json")


if __name__=="__main__":
    main()
