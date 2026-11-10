"""独立重放已知失败测试，从异常栈定位产品源码或断言；用于检查组内失效位置。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from .feature_sources import load_feature_map
from .inventory import SYSTEMS,atomic_json
from .runtime import LIB,Runtime

STACK=re.compile(r"^\s*at\s+([\w.$]+)\.([\w$<>]+)\(([^():]+\.java):(\d+)\)",re.M)


def site_of(failure,variant,mapping):
    if failure is None:
        return None
    typ=failure.get("type","")
    for cls,method,file,num in STACK.findall(failure.text or ""):
        mapped=mapping.get((cls,int(num)))
        if mapped:
            return [typ,"product",mapped]
        if "_ESTest" in cls and not cls.endswith("_scaffolding"):
            sources=list((variant/"test").rglob(file))
            if len(sources)!=1:
                continue
            lines=sources[0].read_text(encoding="utf-8-sig",errors="replace").splitlines()
            if not 0<int(num)<=len(lines):
                continue
            # 跨配置测试编号与生成的局部变量名不稳定，保留行为和观察量。
            statement=re.sub(r'"(?:\\.|[^"\\])*"','STR',lines[int(num)-1].strip())
            statement=re.sub(r"\d+","N",statement)
            return [typ,"oracle",re.sub(r"\s+","",statement)]
    return None


def collect(job):
    version,inv,dest,home=job
    rows=json.loads(inv.read_text(encoding="utf8"))
    dest.mkdir(parents=True,exist_ok=True)
    for row in rows:
        names=row["tests"]["failed"]
        if not names:
            continue
        config=row["configuration"]
        variant=version/"variants"/config
        out=dest/(config+".json")
        h=hashlib.sha256((json.dumps(names)+str(home)).encode())
        for p in [Path(__file__),*[p for d in ("src","test") for p in sorted((variant/d).rglob("*.java"))]]:
            h.update(p.read_bytes())
        fingerprint=h.hexdigest()
        if out.exists() and json.loads(out.read_text(encoding="utf8"))["fingerprint"]==fingerprint:
            continue
        values={}
        with tempfile.TemporaryDirectory(dir=dest) as temp:
            work=Path(temp)
            rt=Runtime(home,work)
            main,tests,_=rt.compile(variant,work)
            cmd=[rt.java,"-cp",os.pathsep.join([str(tests),str(main),LIB]),"org.junit.platform.console.ConsoleLauncher",
                 "--details=summary","--disable-ansi-colors","--reports-dir",work]
            for name in names:
                cls,method=name.rsplit(".",1)
                cmd.extend(["--select-method",cls+"#"+method])
            try:
                proc=subprocess.run(cmd,cwd=work,capture_output=True,timeout=max(120,6*len(names)))
                mapping=load_feature_map(variant)
                for report in work.glob("TEST-*.xml"):
                    for t in ET.parse(report).iter("testcase"):
                        key=t.get("classname")+"."+t.get("name","").split("(")[0]
                        failure=t.find("failure")
                        if failure is None:
                            failure=t.find("error")
                        values[key]=dict(site=site_of(failure,variant,mapping),
                                         outcome="failed" if failure is not None else "aborted" if t.find("skipped") is not None else "passed")
            except subprocess.TimeoutExpired:
                pass
        for key in names:
            values.setdefault(key,dict(site=None,outcome="no_report"))
        if set(values)!=set(names):
            raise ValueError("复核的用例标识不一致："+str(variant))
        atomic_json(out,dict(fingerprint=fingerprint,tests=values))
    print(version.name,flush=True)


def refine(sem,by_test):
    groups={}
    for original,members in sem["groups"].items():
        sites={k:json.dumps(by_test[k]["site"],ensure_ascii=False) for k in members}
        known={s for s in sites.values() if s!="null"}
        if len(known)<=1:
            groups[original]=members
            continue
        for site in sorted(set(sites.values())):
            fragment=sorted(k for k in members if sites[k]==site)
            key=hashlib.sha256(json.dumps(fragment).encode()).hexdigest()[:16]
            groups[key]=fragment
    if sorted(k for g in groups.values() for k in g)!=sorted(sem["pairs"]):
        raise ValueError("失败位置核验改变了成员集合")
    return groups


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","input","output","java-home"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--version")
    p.add_argument("--system",choices=SYSTEMS)
    a=p.parse_args()
    jobs=[(a.data.joinpath(*SYSTEMS[s],v.name).resolve(),v/"inventory.json",a.output.resolve()/s/v.name,a.java_home.resolve())
          for s in ([a.system] if a.system else SYSTEMS) for v in sorted((a.input/s).glob("_MultipleBugs_*"))
          if not a.version or v.name==a.version]
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        list(executor.map(collect,jobs))
