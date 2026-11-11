"""构造真正移除失败用例的独立数据树。每个转换配置在独立进程重跑剩余测试。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from .inventory import SYSTEMS, atomic_json, records
from .runtime import LIB, Runtime, command
from .test_sources import PACKAGE, filtered
from .test_environment import recompile_scaffolding


def stage_file(src,dst):
    if dst.exists():
        if not os.path.samefile(src,dst) and src.read_bytes()!=dst.read_bytes():
            raise ValueError(f"已有输入与源文件不同 {dst}")
        return
    dst.parent.mkdir(parents=True,exist_ok=True)
    try:
        os.link(src,dst)
    except OSError:
        shutil.copyfile(src,dst)


def version_stage(job):
    src,dst,labels,receipt,home=job
    dst.mkdir(parents=True,exist_ok=True)
    report=src/"config.report.csv.done"
    raw=list(csv.reader(report.open(encoding="utf-8-sig",newline="")))
    with labels.open(encoding="utf-8-sig",newline="") as s:
        converted={r["VARIANT"] for r in csv.DictReader(s) if r["FP transformed from F"]=="1"}
    checks={}
    for row in records(src):
        name=row["configuration"]
        source=src/"variants"/name
        target=dst/"variants"/name
        masked=name in converted
        if masked and (not row["tests"]["failed"] or not row["tests"]["passed"]):
            checks[name]={"status":"ineligible"}
            masked=False
        for p in source.rglob("*"):
            if not p.is_file():
                continue
            rel=p.relative_to(source)
            if rel.parts[0] not in {"src","test","coverage"}:
                continue
            if masked and rel.parts[0]=="coverage" and ("failed" in rel.parts or "failed" in p.name):
                continue
            if masked and p.name.endswith("Test.java") and rel.parts[0]=="test":
                body=p.read_text(encoding="utf-8-sig")
                ns=PACKAGE.search(body)
                prefix=((ns[1]+".") if ns else "")+p.stem+"."
                hidden=set(row["tests"]["failed"])
                new=filtered(body,lambda m:prefix+m not in hidden)
                if new!=body:
                    dest=target/rel
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    # 独立文件，绝不就地改写链接的原始测试源码。
                    tmp=dest.with_suffix(".new")
                    tmp.write_text(new,encoding="utf8")
                    os.replace(tmp,dest)
                    continue
            stage_file(p,target/rel)
        if masked:
            with tempfile.TemporaryDirectory(dir=dst) as temp:
                working=Path(temp).resolve()
                rt=Runtime(home,working)
                try:
                    main,tests,_=rt.compile(target,working)
                    classes=sorted({t.rsplit(".",1)[0] for t in row["tests"]["passed"]})
                    args=[rt.java,"-cp",os.pathsep.join((str(main),str(tests),LIB)),
                          "org.junit.platform.console.ConsoleLauncher","--details=summary","--disable-ansi-colors"]
                    for cls in classes:
                        args.extend(["--select-class",cls])
                    compatibility={}
                    try:
                        log=command(args,working)
                    except RuntimeError as original:
                        # 生成器记录了Linux环境路径；只重编译隔离副本中的环境初始化。
                        mapping=recompile_scaffolding(rt,target,working,main,tests)
                        log=command(args,working)
                        compatibility=dict(environment_retry=dict(original_error=str(original),mapping=mapping))
                    success=re.search(r"(\d+) tests? successful\s*\]",log)
                    if not success or int(success[1])!=len(row["tests"]["passed"]):
                        raise ValueError("剩余通过数量与原记录不一致")
                    checks[name]={"status":"passed","removed":len(row["tests"]["failed"]),"remaining":int(success[1]),**compatibility}
                except (RuntimeError,ValueError) as ex:
                    checks[name]={"status":"verification_error","reason":str(ex)}
    for row in raw[1:]:
        if checks.get(row[0],{}).get("status")=="passed":
            row[-1]="__PASSED__"
    atomic_json(receipt,checks)
    if any(v["status"]=="verification_error" for v in checks.values()):
        raise RuntimeError(f"留存用例未核验通过 {receipt}")
    with (dst/"config.report.csv.done").open("w",encoding="utf8",newline="") as s:
        csv.writer(s).writerows(raw)
    # 变异日志仅供隔离评价器读；方法输入树不包含真实缺陷位置。
    print(src.name,len(converted),sum(v["status"]=="passed" for v in checks.values()),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","staging","labels","output","java-home"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--system",choices=SYSTEMS,required=True)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--version")
    a=p.parse_args()
    source=a.data.joinpath(*SYSTEMS[a.system]).resolve()
    stage=a.staging.joinpath(*SYSTEMS[a.system]).resolve()
    if stage.is_relative_to(source) or source.is_relative_to(stage):
        p.error("实验副本与原始系统目录重叠")
    jobs=[(v,stage/v.name,a.labels.resolve()/v.name/"variant_labels.csv",
           a.output.resolve()/a.system/v.name/"construction.json",a.java_home)
          for v in sorted(source.glob("_MultipleBugs_*")) if not a.version or v.name==a.version]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(version_stage,jobs))
