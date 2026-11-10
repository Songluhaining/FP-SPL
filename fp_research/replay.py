"""在相同特征的参考构建上核对用例适用性，再在目标产品上独立复现断言失效。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .inventory import SYSTEMS,atomic_json
from .runtime import LIB,command
from .test_sources import filtered

ROOT=Path(__file__).resolve().parents[1]
REFERENCE_PROJECTS={"bank":"BankAccountTP","email":"Email-FH-JML",
                    "elevator":"Elevator-FH-JML","examdb":"ExamDB-FH-JML"}


def compile_sources(javac,sources,directory,classpath):
    directory.mkdir(parents=True,exist_ok=True)
    command([javac,"-encoding","UTF-8","-g","-cp",classpath,"-d",directory,*sources],directory)


def security_key(paths):
    h=hashlib.sha256()
    for path in sorted(paths):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def reference_build(features,system,projects,tool,cache,home):
    original=projects/REFERENCE_PROJECTS[system]/"features"
    selected=[original/f for f in features]
    key=security_key([tool,*[p for feature in selected for p in feature.rglob("*.java")]])
    root=cache/"reference"/key
    classes=root/"classes"
    stamp=root/"complete"
    if not stamp.exists():
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            stage=Path(tmp)
            expr=stage/"reference.exp"
            expr.write_text("\n".join(features)+"\n",encoding="ascii")
            command([home/"bin/java.exe","-jar",tool,"--expression",expr,"--base-directory",original,
                     "--output-directory",stage],stage)
            compile_sources(home/"bin/javac.exe",sorted((stage/"reference").rglob("*.java")),stage/"classes",LIB)
            # 同一配置可被多个版本并发使用，只发布完整的编译结果。
            try:
                os.rename(stage/"classes",classes)
            except OSError:
                if not classes.is_dir():
                    raise
            stamp.write_text(key,encoding="ascii")
    return classes,key


def run_cases(home,products,tests,selected,work):
    cp=os.pathsep.join([str(tests),str(products),LIB])
    args=[home/"bin/java.exe","-cp",cp,"org.junit.platform.console.ConsoleLauncher",
          "--disable-ansi-colors","--details=summary","--reports-dir",str(work)]
    for identity in selected:
        args.extend(["--select-method",identity])
    proc=subprocess.run(args,cwd=work,capture_output=True,timeout=max(45,10*len(selected)))
    import xml.etree.ElementTree as ET
    outcomes={}
    for report in work.glob("TEST-*.xml"):
        for test in ET.parse(report).iter("testcase"):
            identity=test.get("classname")+"#"+test.get("name","").split("(")[0]
            failure=test.find("failure")
            if failure is None:
                failure=test.find("error")
            if failure is not None:
                typ=failure.get("type","")
                message=failure.get("message","")
                # 环境故障与普通崩溃不得用作规格违例。
                status="assertion" if typ.endswith(("AssertionError","ComparisonFailure","AssertionFailedError")) else "runtime_error"
                outcomes[identity]=(status,typ+":"+message)
            else:
                outcomes[identity]=("passed","") if test.find("skipped") is None else ("skipped","")
    return {k:outcomes.get(k,("runner_error",str(proc.returncode))) for k in selected}


def prepare_probe(donor,identity,dest,tag):
    cls,method=identity.rsplit(".",1)
    short=cls.rsplit(".",1)[-1]
    src=next((donor/"test").rglob(short+".java"))
    scaffold=next((donor/"test").rglob(short+"_scaffolding.java"),None)
    new="FPProbe"+tag
    files=[]
    for source in [src]+([scaffold] if scaffold else []):
        text=source.read_text(encoding="utf-8-sig")
        if source==src:
            text=filtered(text,lambda name:name==method)
        text=re.sub(r"\b"+re.escape(short)+r"(?=\b|_scaffolding\b)",new,text)
        path=dest/source.name.replace(short,new)
        path.write_text(text,encoding="utf8")
        files.append(path)
    return files,cls[:-len(short)]+new+"#"+method


def replay_version(job):
    source,plan_path,inventory,projects,tool,cache,home=job
    plan=json.loads(plan_path.read_text(encoding="utf8"))
    rows=json.loads(inventory.read_text(encoding="utf8"))
    system=plan_path.parent.parent.name
    dest=plan_path.parent/"executions"
    dest.mkdir(parents=True,exist_ok=True)
    for row in rows:
        cid=row["configuration"]
        if cid not in plan:
            continue
        ranked=plan[cid]
        keys=sorted({k for strategy in ("model","random","coverage") for k in ranked.get(strategy,[])})
        variant=source/"variants"/cid
        srcfiles=sorted((variant/"src").rglob("*.java"))
        semantic_inputs=[Path(__file__),Path(__file__).with_name("test_sources.py"),*srcfiles,tool]
        for key in keys:
            cfg,test=key.split("/")
            cls=test.rsplit(".",1)[0].rsplit(".",1)[-1]
            semantic_inputs.extend(sorted((source/"variants"/cfg/"test").rglob(cls+"*.java")))
        clean=projects/REFERENCE_PROJECTS[system]/"features"
        for feature in row["features"]:
            semantic_inputs.extend((clean/feature).rglob("*.java"))
        fingerprint=hashlib.sha256((security_key(semantic_inputs)+json.dumps(keys)+str(home)).encode()).hexdigest()
        receipt=dest/(cid+".json")
        if receipt.exists() and json.loads(receipt.read_text(encoding="utf8"))["fingerprint"]==fingerprint:
            continue
        outcomes={}
        with tempfile.TemporaryDirectory(dir=dest) as tmp:
            work=Path(tmp)
            if keys:
                reference,_=reference_build(row["features"],system,projects,tool,cache,home)
                target=work/"target"
                compile_sources(home/"bin/javac.exe",srcfiles,target,LIB)
                probes={}
                for i,key in enumerate(keys):
                    cfg,test=key.split("/")
                    files,identity=prepare_probe(source/"variants"/cfg,test,work,str(i))
                    probes[key]=(files,identity)
                tests=work/"tests"
                # 整批可编译时复用编译器启动，失败时逐项诊断。
                def compile_probes(items):
                    files=[f for key in items for f in probes[key][0]]
                    compile_sources(home/"bin/javac.exe",files,tests,os.pathsep.join([str(target),LIB]))
                try:
                    compile_probes(keys)
                    valid=keys[:]
                except RuntimeError:
                    valid=[]
                    for key in keys:
                        try:
                            compile_probes([key])
                            valid.append(key)
                        except RuntimeError as ex:
                            outcomes[key]=dict(status="compile_error",diagnostic=str(ex)[-300:])
                identifiers={k:probes[k][1] for k in valid}
                def execute(build,chosen,stage):
                    if not chosen:
                        return {}
                    folder=work/stage
                    folder.mkdir()
                    try:
                        values=run_cases(home,build,tests,[identifiers[k] for k in chosen],folder)
                        return {k:values[identifiers[k]] for k in chosen}
                    except subprocess.TimeoutExpired:
                        return {k:("timeout","") for k in chosen}
                control=execute(reference,valid,"reference-run")
                applicable=[k for k in valid if control[k][0]=="passed"]
                first=execute(target,applicable,"target-run-1")
                second=execute(target,[k for k in applicable if first[k][0]=="assertion"],"target-run-2")
                for k in valid:
                    if k not in applicable:
                        outcomes[k]=dict(status="reference_"+control[k][0])
                    else:
                        state=first[k][0]
                        if state=="assertion":
                            state="confirmed" if first[k]==second[k] else "not_reproduced"
                        outcomes[k]=dict(status=state,reference=control[k],first=first[k],second=second.get(k))
        atomic_json(receipt,dict(fingerprint=fingerprint,results=outcomes))
    print(system,source.name,"replay complete",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","plans","input","projects","tool","java-home"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--system",choices=SYSTEMS)
    p.add_argument("--version")
    a=p.parse_args()
    jobs=[]
    for system in [a.system] if a.system else SYSTEMS:
        for path in sorted((a.plans/system).glob("*/probe-plan.json")):
            v=path.parent.name
            if not a.version or v==a.version:
                jobs.append((a.data.joinpath(*SYSTEMS[system],v).resolve(),path.resolve(),
                             a.input/system/v/"inventory.json",a.projects.resolve(),a.tool.resolve(),
                             a.plans.resolve()/"build",a.java_home.resolve()))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(replay_version,jobs))
