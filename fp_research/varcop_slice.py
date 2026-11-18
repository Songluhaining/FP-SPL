"""同一产品只建立一次依赖图，用官方函数计算多组SPC的特征切片。"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .inventory import atomic_json
from .localization_study import ROOT, digest

JAR = ROOT / "result/varcop_source/feature-slicing.jar"
JAVA = Path("E:/Program Files/Java/jdk-1.8/bin")


@lru_cache(maxsize=1)
def runner():
    source = ROOT / "fp_research/checks/VarcopFeatureProfile.java"
    key = hashlib.sha256(source.read_bytes()+JAR.read_bytes()).hexdigest()
    parent = ROOT / "result/slice-runners"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / key
    if not (target / "complete.json").exists():
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            stage = Path(tmp)
            subprocess.run([str(JAVA / "javac.exe"), "-encoding", "UTF-8", "-cp", str(JAR),
                            "-d", str(stage), str(source)], capture_output=True, check=True)
            publication = stage / "published"
            publication.mkdir()
            for file in stage.glob("*.class"):
                os.replace(file, publication / file.name)
            atomic_json(publication / "complete.json",
                        {f.name: digest(f) for f in publication.glob("*.class")})
            try:
                os.rename(publication, target)
            except FileExistsError:
                if not (target / "complete.json").exists():
                    raise
    for name, sha in json.loads((target / "complete.json").read_text()).items():
        if digest(target / name) != sha:
            raise ValueError("VARCOP切片适配编译结果改变")
    return target


def batch(version, plans, output):
    """plans 是 method -> (SPC日志路径, 最终JSON路径)。"""
    by_source = {}
    for method, (log, target) in plans.items():
        for line in log.read_text(encoding="utf-8").splitlines():
            _, source = line.split(";", 1)
            source = source.strip()
            by_source.setdefault(source, {})[method] = (log, target)
    compiled = runner()
    combined = {method: {} for method in plans}

    def one(item):
        source, tasks = item
        product = Path(source).parent.name
        folder = output / "shared-slicing" / product
        folder.mkdir(parents=True, exist_ok=True)
        batch_file = folder / "groups.tsv"
        results = {method: folder / f"{method}.json" for method in tasks}
        batch_file.write_text("".join(f"{log.as_posix()}\t{results[method].as_posix()}\n"
                                     for method, (log, _) in tasks.items()), encoding="utf-8")
        first_log = next(iter(tasks.values()))[0]
        command = [str(JAVA / "java.exe"), "-Xmx1536m", "-XX:+UseSerialGC",
                   "-Djava.io.tmpdir="+str(folder), "-Dcoverage_file_name=spectrum_failed_coverage.xml",
                   "-Dcached.adjacency=true", "-Dindexed.propagation=true",
                   "-cp", os.pathsep.join([str(compiled), str(JAR)]),
                   "VarcopFeatureProfile", source, str(first_log), str(batch_file)]
        with (folder / "run.log").open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=900)
        return {method: json.loads(path.read_text(encoding="utf-8")) for method, path in results.items()}

    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(one, sorted(by_source.items())):
            for method, products in result.items():
                if combined[method].keys() & products.keys():
                    raise ValueError("同一配置重复切片")
                combined[method].update(products)
    for method, (_, target) in plans.items():
        atomic_json(target, combined[method])
