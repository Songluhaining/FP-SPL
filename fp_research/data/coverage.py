"""读取原始汇总覆盖，核对特征源码身份及产品结果。"""
import csv
from pathlib import Path
from typing import NamedTuple
import xml.etree.ElementTree as ET
class Observation(NamedTuple):
    configuration: str
    failed_tests: int
    passed_tests: int
    present: set
    failed_hits: dict
    passed_hits: dict

def spectrum(path):
    if not path.exists():
        return 0, {}
    root = ET.parse(path).getroot()
    tests = root.find("tests")
    if tests is None:
        raise ValueError(f"缺少测试清单 {path}")
    n = int(tests.attrib["count"])
    if len(tests.findall("test")) != n:
        raise ValueError(f"测试清单与计数不符 {path}")
    counts = {}
    for line in root.iterfind(".//package/file/line"):
        if not (line.get("featureClass") and line.get("featureLineNum")):
            continue
        key = (line.get("featureClass"), line.get("featureLineNum"))
        hits = int(line.get("count", "0"))
        if not 0 <= hits <= n:
            raise ValueError(f"覆盖次数超出测试数 {path}: {key}, {hits}/{n}")
        counts[key] = max(counts.get(key, 0), hits)
    return n, counts

def read_version(version):
    with (version / "config.report.csv.done").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    if header[-1].strip() != "__TEST_OUTPUT__" or not rows:
        raise ValueError(f"无效的配置报告 {version}")
    records = []
    seen = set()
    for row in rows:
        if len(row) != len(header) or row[-1].strip() not in {"__PASSED__", "__FAILED__"}:
            raise ValueError(f"无效的配置结果 {version}")
        if any(v.strip() not in {"T","F"} for v in row[1:-1]):
            raise ValueError(f"无效特征标记 {version}")
        name = row[0].strip()
        if name in seen:
            raise ValueError(f"配置重复 {version}/{name}")
        seen.add(name)
        selected = {k for k, v in zip(header[1:-1], row[1:-1]) if v.strip() == "T"}
        folder = version / "variants" / name / "coverage"
        nf, fail = spectrum(folder / "spectrum_failed_coverage.xml")
        npass, passed = spectrum(folder / "spectrum_passed_coverage.xml")
        if nf+npass == 0 or (nf > 0) != (row[-1].strip() == "__FAILED__"):
            raise ValueError(f"配置报告与覆盖不符 {version.name}/{name}")
        present = {k for k in fail.keys() | passed.keys() if k[0].split(".")[0] in selected}
        records.append(Observation(name, nf, npass, present, fail, passed))
    return records
