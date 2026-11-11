"""按源码和原工具摘要原子发布加速入口，运行期间不覆盖已加载的类文件。"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


@lru_cache(maxsize=1)
def compiled_runner():
    root=Path(__file__).resolve().parents[1]
    source=root/"fp_research/checks/SliceProfile.java"
    jar=root.parent/"PULP/plugins/all-statements-slicing.jar"
    key=hashlib.sha256(source.read_bytes()+hashlib.sha256(jar.read_bytes()).digest()).hexdigest()
    parent=root/"result/slice-runners"
    parent.mkdir(parents=True,exist_ok=True)
    target=parent/key
    if not (target/"complete.json").exists():
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            stage=Path(tmp)
            subprocess.run(["E:/Program Files/Java/jdk-1.8/bin/javac.exe","-encoding","UTF-8","-cp",jar,"-d",stage,source],
                           cwd=root,check=True,capture_output=True)
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.glob("*.class")}
            (stage/"complete.json").write_text(json.dumps(dict(key=key,files=hashes)),encoding="utf8")
            # 临时目录由上下文负责清理；发布其内部目录，避免清理已发布结果。
            publish=stage/"published"
            publish.mkdir()
            for p in list(stage.iterdir()):
                if p.is_file():os.replace(p,publish/p.name)
            try:os.rename(publish,target)
            except FileExistsError:
                if not (target/"complete.json").exists():raise
    proof=json.loads((target/"complete.json").read_text())
    for name,digest in proof["files"].items():
        if hashlib.sha256((target/name).read_bytes()).hexdigest()!=digest:raise ValueError("编译入口摘要不符")
    return target,key
