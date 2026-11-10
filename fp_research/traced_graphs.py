"""按实际测试缓存隔离插桩轨迹，两个协议只复用内容和执行条件完全相同的证据。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from zipfile import ZipFile, ZIP_DEFLATED

from .feature_sources import load_feature_map
from .graphs import read_posterior
from .inventory import SYSTEMS,atomic_json
from .isolated_trace import single_trace
from .runtime import ROOT,Runtime


def digest(*items):
    h=hashlib.sha256()
    for item in items:
        data=item if isinstance(item,bytes) else str(item).encode()
        h.update(len(data).to_bytes(8,"big"))
        h.update(data)
    return h.hexdigest()


class Engine:
    def __init__(self,home,cache):
        self.cache=cache.resolve()
        self.cache.mkdir(parents=True,exist_ok=True)
        self.runtime=Runtime(home,self.cache/"java")
        self.runtime.build()
        sources=[Path(__file__),ROOT/"fp_research/graphs.py",ROOT/"fp_research/runtime.py",
                 ROOT/"fp_research/test_sources.py",ROOT/"fp_research/isolated_trace.py",
                 ROOT/"fp_research/sites.py",ROOT/"fp_research/feature_sources.py",
                 self.runtime.cache/"source.sha256",home/"release",home/"bin/java.exe",home/"bin/javac.exe"]
        sources+=sorted((ROOT/"lib").glob("*.jar"))
        self.environment=digest(*(digest(p.name,p.read_bytes()) for p in sources))
        self.locks={}
        self.guard=threading.Lock()

    def lock(self,key):
        with self.guard:
            return self.locks.setdefault(key,threading.Lock())

    @lru_cache(maxsize=4096)
    def program(self,variant):
        parts=[self.environment]
        for sub in ("src","test"):
            for p in sorted((variant/sub).rglob("*.java")):
                parts.append(digest(p.relative_to(variant),p.read_bytes()))
        return digest(*parts)

    def build(self,variant,program):
        folder=self.cache/"builds"/program
        receipt=folder/"classes.json"
        with self.lock(("build",program)):
            if not receipt.exists():
                folder.mkdir(parents=True,exist_ok=True)
                _,_,names=self.runtime.compile(variant,folder)
                atomic_json(receipt,names)
            return folder/"main",folder/"tests",json.loads(receipt.read_text(encoding="utf8"))

    @lru_cache(maxsize=4096)
    def mapping(self,variant):
        mapping=load_feature_map(variant)
        text="".join(f"{cls}#{n}\t{k}\n" for (cls,n),k in sorted(mapping.items()))
        return mapping,text,digest(text)

    def capture(self,variant,test,expected,known,program,key):
        folder=self.cache/"traces"/key[:2]
        folder.mkdir(parents=True,exist_ok=True)
        receipt=folder/(key+".json")
        archive=folder/(key+".zip")
        with self.lock(("trace",key)):
            if receipt.exists():
                cached=json.loads(receipt.read_text(encoding="utf8"))
                if cached["status"]!="ok" or archive.is_file():
                    return cached,archive
            errors=[]
            for attempt in (1,2):
                try:
                    with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
                        work=Path(tmp)
                        compiled=self.build(variant,program)
                        source,trace=single_trace(self.runtime,variant,compiled,test,expected,work/"run",
                                                  known,self.mapping(variant)[0])
                        logs=[p for p in source.glob("*.source.log") if p.name!="traced.source.log"]
                        if not logs:
                            raise ValueError("缺少程序静态语句日志")
                        temp_archive=folder/(key+f".{os.getpid()}.tmp")
                        with ZipFile(temp_archive,"w",compression=ZIP_DEFLATED,compresslevel=2) as z:
                            z.write(trace,"run.log")
                            for p in logs:
                                z.write(p,"static/"+p.name)
                        os.replace(temp_archive,archive)
                    result=dict(status="ok",attempt=attempt,prior_errors=errors)
                    break
                except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as e:
                    errors.append(str(e)[-1400:])
            else:
                result=dict(status="unavailable",attempt=2,prior_errors=errors)
            atomic_json(receipt,result)
            return result,archive

    def infer(self,job):
        version,out,group,members,pairs,sites=job
        spec={k:"failed" for k in members}
        spec.update({pairs[k]["control"]:"passed" for k in members if pairs[k]["control"]})
        items=[]
        for identity,expected in sorted(spec.items()):
            c,test=identity.split("/")
            variant=version/"variants"/c
            program=self.program(variant)
            mapping=self.mapping(variant)
            known=sites.get(identity)
            trace_key=digest(program,test,expected,json.dumps(known,sort_keys=True),mapping[2])
            items.append((identity,variant,test,expected,known,program,trace_key))
        fingerprint=digest(self.environment,*(digest(x[0],x[-1]) for x in items))
        canonical=self.cache/"groups"/fingerprint[:2]/(fingerprint+".json")
        target=out/"graphs"/(group+".json")
        with self.lock(("group",fingerprint)):
            if canonical.exists():
                record=json.loads(canonical.read_text(encoding="utf8"))
            else:
                record=dict(fingerprint=fingerprint,executions=spec,trace_keys={x[0]:x[-1] for x in items},
                            backend="isolated-method",environment=self.environment)
                try:
                    with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
                        work=Path(tmp)
                        manifest=[]
                        for index,(identity,variant,test,expected,known,program,key) in enumerate(items):
                            receipt,archive=self.capture(variant,test,expected,known,program,key)
                            if receipt["status"]!="ok":
                                raise ValueError(identity+" "+receipt["prior_errors"][-1])
                            stage=work/str(index)
                            stage.mkdir()
                            with ZipFile(archive) as z:
                                for name in z.namelist():
                                    if name!="run.log" and not (name.startswith("static/") and name.count("/")==1
                                                               and Path(name).name.endswith(".source.log")):
                                        raise ValueError("未知轨迹缓存条目")
                                    dest=stage/name
                                    dest.parent.mkdir(exist_ok=True)
                                    with z.open(name) as src,dest.open("wb") as dst:
                                        shutil.copyfileobj(src,dst)
                            location=stage/"map.tsv"
                            location.write_text(self.mapping(variant)[1],encoding="utf8")
                            manifest.append((identity,stage/"static",stage/"run.log",test,
                                             "FAIL" if expected=="failed" else "PASS",location))
                        scores=read_posterior(self.runtime.infer(work,manifest))
                        record.update(status="ok" if scores else "unmapped",scores=scores)
                except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as e:
                    record.update(status="error",reason=str(e)[-1600:])
                atomic_json(canonical,record)
            atomic_json(target,record)
        return record["status"]


def jobs(data,root,system=None,version=None,limit=None):
    batch=[]
    for s in ([system] if system else SYSTEMS):
        for v in sorted((root/s).glob("_MultipleBugs_*")):
            if version and v.name!=version:
                continue
            sem=json.loads((v/"semantics.json").read_text(encoding="utf8"))
            for g,items in list(sem["groups"].items())[:limit]:
                batch.append((data.joinpath(*SYSTEMS[s],v.name),v,g,items,sem["pairs"],sem["locations"]))
    return batch


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","input","cache","java-home"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--masked-data",type=Path)
    p.add_argument("--masked-input",type=Path)
    p.add_argument("--system",choices=SYSTEMS)
    p.add_argument("--version")
    p.add_argument("--limit-groups",type=int)
    p.add_argument("--workers",type=int,default=6)
    a=p.parse_args()
    engine=Engine(a.java_home.resolve(),a.cache.resolve())
    batch=jobs(a.data.resolve(),a.input.resolve(),a.system,a.version,a.limit_groups)
    if a.masked_data:
        batch+=jobs(a.masked_data.resolve(),a.masked_input.resolve(),a.system,a.version,a.limit_groups)
    print("scheduled",len(batch),"groups",engine.environment,flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,status in enumerate(pool.map(engine.infer,batch),1):
            if i%50==0 or i==len(batch):
                print("completed",i,"/",len(batch),flush=True)
