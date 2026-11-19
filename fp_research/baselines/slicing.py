"""同案例派生目录中的静态切片及跨产品只读缓存。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from ..config import ROOT,tool
from ..io import atomic_json
JAVA=tool('java_home')
_LOCKS={};_GUARD=threading.Lock();_ALIASES={}
_SLICE_WORKERS=int(os.environ.get('FP_SLICE_WORKERS','2'))
_SLICE_TIMEOUT=int(os.environ.get('FP_SLICE_TIMEOUT','7200'))
_SLICE_BACKEND=os.environ.get('FP_SLICE_BACKEND','indexed')
_SLICE_HEAP_MB=int(os.environ.get('FP_SLICE_HEAP_MB','1536'))
_SLICE_SLOTS=threading.BoundedSemaphore(_SLICE_WORKERS)
def slices(source,output,cache):
    from .signatures import signatures
    from .compiler import compiled_runner
    jar=tool("pulp")/"plugins/all-statements-slicing.jar"
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
