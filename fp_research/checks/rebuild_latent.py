"""从单个测试的二值覆盖重建频谱；输入解析只取源码映射、行号和命中计数。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

from fp_research.inventory import SYSTEMS
from fp_research.latent_fault import Observation, infer, read_version
from fp_research.risk import save_rows


def recount(variant, status, selected):
    folder = variant / "coverage"
    summary = folder / ("spectrum_" + status + "_coverage.xml")
    tests = sorted((folder / status).glob("*.coverage.xml"))
    if not summary.exists():
        if tests:
            raise ValueError(f"缺失源码映射 {variant}/{status}")
        return 0, {}, set()
    names = set()
    mapping = {}
    for package in ET.parse(summary).iterfind(".//package"):
        for file in package.findall("file"):
            stem = Path(file.attrib["path"].replace(chr(92), "/")).stem
            for line in file.findall("line"):
                if not line.get("featureClass") or not line.get("featureLineNum"):
                    continue
                origin = (line.attrib["featureClass"], line.attrib["featureLineNum"])
                location = (package.get("name"), stem, line.attrib["num"])
                if mapping.setdefault(location, origin) != origin:
                    raise ValueError(f"冲突的源码映射 {location}")
                if origin[0].split(".")[0] in selected:
                    names.add(origin)
    counts = Counter()
    for test in tests:
        reached = set()
        for package in ET.parse(test).iterfind(".//package"):
            for file in package.findall("file"):
                stem = Path(file.attrib["path"].replace(chr(92), "/")).stem
                for line in file.findall("line"):
                    if int(line.get("count", 0)) > 0:
                        origin = mapping.get((package.get("name"), stem, line.attrib["num"]))
                        if origin is not None:
                            reached.add(origin)
        counts.update(reached)
    return len(tests), counts, names


def verify_version(job):
    system, version = job
    reference = read_version(version)
    raw = []
    changed_cells = 0
    test_count = 0
    with (version / "config.report.csv.done").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        for row, old in zip(reader, reference, strict=True):
            name = row[0]
            selected = {f for f, v in zip(header[1:-1], row[1:-1]) if v.strip()=="T"}
            variant = version / "variants" / name
            nf, f, present_f = recount(variant, "failed", selected)
            npass, p, present_p = recount(variant, "passed", selected)
            present = present_f | present_p
            if (name, nf, npass, present) != old[:4]:
                raise ValueError(f"配置/用例/源码映射不一致 {version}/{name}")
            test_count += nf + npass
            changed_cells += sum(a.get(k, 0) != b.get(k, 0) for a,b in [(old[4],f),(old[5],p)] for k in a.keys()|b.keys())
            raw.append(Observation(name,nf,npass,present,f,p))
    predictions = [dict(system=system, **r) for r in infer(raw, version.name)]
    audit = dict(system=system,version=version.name,tests=test_count,changed_coverage_counts=changed_cells)
    return predictions, audit


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    a=p.parse_args()
    jobs=[(system,v) for system,relative in SYSTEMS.items() for v in sorted(a.data.joinpath(*relative).glob("_MultipleBugs_*"))]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(verify_version,jobs))
    save_rows(a.output/"predictions.csv",[row for predictions,_ in results for row in predictions])
    save_rows(a.output/"audit.csv",[audit for _,audit in results])
