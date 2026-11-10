"""电梯与考试数据库的单缺陷实验。所有派生文件写入隔离工作目录。"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import shutil
import subprocess
import tempfile

from .inventory import records, atomic_json
from .risk import save_rows
from .supplement import version_stage
from .baseline_features import worker
from .checks.rebuild_latent import verify_version
from .latent_fault import read_version, infer

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = {"elevator": ("Elevator-FH-JML", "4wise-Elevator-FH-JML-1BUG-Full"),
           "examdb": ("ExamDB", "4wise-ExamDB-1BUG-Full")}


def versions(data, system):
    paths = sorted(data.joinpath(*SYSTEMS[system]).glob("_MultipleBugs_.NOB_1.ID_*"))
    if not paths:
        raise ValueError(f"找不到单缺陷版本 {data}/{system}")
    return paths


def slice_inputs(product):
    files=sorted((product/"src").rglob("*.java")) + sorted((product/"coverage").glob("spectrum_*_coverage.xml"))
    h=hashlib.sha256()
    for p in files:
        h.update(p.relative_to(product).as_posix().encode()+b"\0"+hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def slice_version(version, destination, java, jar, reuse=None):
    builds = destination / "variants"
    builds.mkdir(parents=True,exist_ok=True)
    names=[f"slice.all_statements.{direction}_slicing.log" for direction in ("forward","backward")]
    with tempfile.TemporaryDirectory(dir=destination) as scratch:
        pending=Path(scratch)/"variants"
        outputs={}
        signatures={}
        jar_hash=hashlib.sha256(jar.read_bytes()).hexdigest()
        for variant in sorted((version / "variants").iterdir()):
            prior=reuse/variant.name if reuse else None
            signature=jar_hash+" "+slice_inputs(variant)
            signatures[variant.name]=signature
            if prior and all((prior/n).is_file() for n in names):
                stamp=prior/"inputs.sha256"
                if stamp.exists() and stamp.read_text()==signature:
                    outputs[variant.name]=prior
                    continue
                if not stamp.exists() and (prior/"src").is_dir() and jar_hash+" "+slice_inputs(prior)==signature:
                    outputs[variant.name]=prior
                    continue
            target=pending/variant.name
            shutil.copytree(variant/"src",target/"src")
            (target/"coverage").mkdir()
            for xml in (variant/"coverage").glob("spectrum_*_coverage.xml"):
                shutil.copyfile(xml,target/"coverage"/xml.name)
            outputs[variant.name]=target
        if pending.exists():
            with (destination/"slicing.log").open("w") as log:
                subprocess.run([java,"-Xmx512m","-XX:+UseSerialGC","-Dvariants_dir="+str(pending),"-jar",jar],
                               cwd=destination,stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=True)
        for name, product in outputs.items():
            target=builds/name
            target.mkdir(exist_ok=True)
            for file in names:
                payload=(product/file).read_bytes()
                if not isinstance(json.loads(payload),dict):
                    raise ValueError(f"切片格式异常 {product/file}")
                (target/file).write_bytes(payload)
            (target/"inputs.sha256").write_text(signatures[name],encoding="ascii")


def run_version(job):
    args, system, original = job
    root=args.work/args.scenario/system/original.name
    data=original if args.scenario=="main" else args.work/"visible"/system/original.name
    if args.stage=="mask":
        rows=list(records(original))
        failed=sorted(r["configuration"] for r in rows if r["outcome"]=="failed")
        labels=root/"selection.csv"
        # 每隔一个失败配置尝试转换，首个失败配置始终保留，不查阅真值。
        save_rows(labels,[{"VARIANT":r["configuration"],"FP transformed from F":int(r["configuration"] in failed[1::2])}
                          for r in rows])
        version_stage((original,data,labels,root/"construction.json",args.java_home))
    elif args.stage=="slices":
        prior=args.work/"main"/system/original.name/"slices/variants"
        slice_version(data,root/"slices",args.java_home/"bin/java.exe",
                      args.sources/"PULP/plugins/all-statements-slicing.jar",prior if args.scenario=="masked" else None)
    elif args.stage in {"PULP","CLAP"}:
        worker((args.sources/args.stage,args.stage,data,root/"slices",root/args.stage))
    elif args.stage=="observe":
        rows=list(records(data))
        if any(not r["coverage_agrees"] or not r["source"] or not r["test_source"] for r in rows):
            raise ValueError(f"输入不齐 {data}")
        atomic_json(root/"inventory.json",rows)
        model=infer(read_version(data),data.name)
        raw,audit=verify_version((system,data))
        if audit["changed_coverage_counts"] or raw!=[dict(system=system,**r) for r in model]:
            raise ValueError(f"逐用例覆盖与汇总结果不符 {data} {audit}")
        save_rows(root/"latent.csv",raw)
        atomic_json(root/"coverage-audit.json",audit)
    print(args.stage, args.scenario, system, data.name,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["mask","slices","PULP","CLAP","observe"])
    p.add_argument("--scenario",choices=["main","masked"],default="main")
    p.add_argument("--system",choices=SYSTEMS)
    p.add_argument("--version")
    p.add_argument("--data",type=Path,default=Path("D:/dataset/whn"))
    p.add_argument("--work",type=Path,default=Path("D:/dataset/whn/fp_extension_work"))
    p.add_argument("--sources",type=Path,default=ROOT.parent)
    p.add_argument("--java-home",type=Path,default=Path("E:/Program Files/Java/jdk-1.8"))
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if a.stage=="mask" and a.scenario!="masked":
        p.error("转换实验必须选择 masked")
    for attr in ("data","work","sources","java_home"):
        setattr(a,attr,getattr(a,attr).resolve())
    for relative in SYSTEMS.values():
        original=a.data.joinpath(*relative)
        if a.work.is_relative_to(original) or original.is_relative_to(a.work):
            p.error("工作目录与原始系统目录重叠")
    jobs=[(a,s,v) for s in ([a.system] if a.system else SYSTEMS) for v in versions(a.data,s)
          if not a.version or v.name==a.version]
    if not jobs:
        p.error("版本不存在")
    executor=ProcessPoolExecutor if a.stage in {"PULP","CLAP","observe"} else ThreadPoolExecutor
    with executor(max_workers=a.workers) as pool:
        list(pool.map(run_version,jobs))


if __name__=="__main__":
    main()
