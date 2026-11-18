"""同一产品只建立一次依赖图，用官方函数计算多组SPC的特征切片。"""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
import base64
import gzip
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

from .inventory import atomic_json
from .localization_study import ROOT, digest

JAR = ROOT / "result/varcop_source/feature-slicing.jar"
JAVA = Path("E:/Program Files/Java/jdk-1.8/bin")


@contextmanager
def locked(path):
    with path.open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(.1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def basis_key(source):
    source = Path(source)
    hashed = hashlib.sha256()
    for file in sorted(source.rglob("*.java")):
        hashed.update(file.relative_to(source).as_posix().encode())
        hashed.update(hashlib.sha256(file.read_bytes()).digest())
    executed = {line.get("featureClass")+"."+str(line.get("featureLineNum"))
                for line in ET.parse(source.parent / "coverage/spectrum_failed_coverage.xml").iterfind(".//package/file/line")
                if int(line.get("count", 0)) > 0}
    hashed.update(json.dumps(sorted(executed)).encode())
    hashed.update(bytes.fromhex(digest(JAR)))
    hashed.update(bytes.fromhex(digest(ROOT / "fp_research/checks/VarcopFeatureProfile.java")))
    return hashed.hexdigest()


def read_basis(path):
    forward, backward, mapping, executed, missing = {}, {}, {}, set(), set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            tag, *encoded = line.rstrip("\n").split("\t")
            items = [base64.b64decode(s).decode("utf-8") for s in encoded]
            if tag == "F":
                forward.setdefault(items[0], set()).add(items[1])
            elif tag == "Z":
                forward.setdefault(items[0], set())
            elif tag == "B":
                backward.setdefault(items[0], set()).add(items[1])
            elif tag == "M":
                mapping[items[0]] = items[1]
            elif tag == "E":
                executed.add(items[0])
            elif tag == "N":
                missing.add(items[0])
            else:
                raise ValueError("未知静态关系记录")
    return forward, backward, mapping, executed, missing


def from_basis(basis, log, source):
    forward, backward, mapping, executed, missing = basis
    interactions = Counter()
    # Java先对完整日志行去重，再过滤关闭的特征；保留组合内的特征次序。
    lines = set(log.read_text(encoding="utf-8").splitlines())
    for line in lines:
        combination, raw_source = line.split(";", 1)
        if raw_source.strip() != source:
            continue
        features = [f for f in combination.split(",") if "#" not in f]
        if not features or features[0] not in forward:
            continue
        common = set(forward[features[0]])
        for feature in features[1:]:
            if feature not in forward:
                break
            common.intersection_update(forward[feature])
            if not common:
                break
        selected = set(common)
        for key in list(common):
            if key in missing:
                raise ValueError("官方静态关系中缺少当前交点的后向映射")
            selected.update(backward.get(key, set()))
        interactions.update(mapping[key] for key in selected if key in mapping and mapping[key] in executed)
    return {Path(source).parent.name: {key: {"num_interactions": value} for key, value in interactions.items()}}


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
        key = basis_key(source)
        cache = version.parent / ".fp_research/varcop-slice-cache" / key
        cache.mkdir(parents=True, exist_ok=True)
        basis_path = cache / "basis.tsv.gz"
        proof = cache / "complete.json"
        with locked(cache / "publish.lock"):
            reused = proof.exists()
            if reused:
                if json.loads(proof.read_text())["sha256"] != digest(basis_path):
                    raise ValueError("特征切片静态关系缓存改变")
            else:
                temporary = folder / "basis.tsv.gz"
                command = [str(JAVA / "java.exe"), "-Xmx1536m", "-XX:+UseSerialGC",
                           "-XX:ActiveProcessorCount=2", "-XX:CICompilerCount=2",
                           "-Djava.io.tmpdir="+str(folder), "-Dcoverage_file_name=spectrum_failed_coverage.xml",
                           "-Dcached.adjacency=true",
                           "-Dexport.basis="+str(temporary),
                           "-cp", os.pathsep.join([str(compiled), str(JAR)]),
                           "VarcopFeatureProfile", source, str(first_log), str(batch_file)]
                with (folder / "run.log").open("w", encoding="utf-8") as stream:
                    subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT,
                                   check=True, timeout=900)
                basis = read_basis(temporary)
                for method, (log, _) in tasks.items():
                    expected = json.loads(results[method].read_text(encoding="utf-8"))
                    if from_basis(basis, log, source) != expected:
                        raise ValueError("静态关系重算与官方特征交互函数不同")
                os.replace(temporary, basis_path)
                atomic_json(proof, dict(key=key, sha256=digest(basis_path), source=source,
                                        groups_verified=len(tasks)))
            basis = read_basis(basis_path)
            data = {method: from_basis(basis, log, source) for method, (log, _) in tasks.items()}
            atomic_json(folder / "cache.json", dict(key=key, reused=reused, basis_sha256=digest(basis_path)))
            for method, value in data.items():
                atomic_json(results[method], value)
            return data

    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(one, sorted(by_source.items())):
            for method, products in result.items():
                if combined[method].keys() & products.keys():
                    raise ValueError("同一配置重复切片")
                combined[method].update(products)
    for method, (_, target) in plans.items():
        atomic_json(target, combined[method])
