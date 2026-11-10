"""单缺陷版本内的贝叶斯诊断。只读取可见测试数量、源码映射与覆盖次数。"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import NamedTuple

import numpy as np
from scipy.special import betaln, softmax

from .inventory import SYSTEMS


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


def infer(records, version_name):
    failed = [r for r in records if r.failed_tests]
    if not failed:
        raise ValueError(f"没有已知失败配置 {version_name}")
    candidates = sorted(set.intersection(*(r.present for r in failed)))
    if not candidates:
        raise ValueError(f"没有所有失败配置共有的候选语句 {version_name}")
    # 对特征等先验，再在特征内对语句等分，避免行数较多的特征仅凭长度获益。
    features = [k[0].split(".")[0] for k in candidates]
    sizes = Counter(features)
    log_prior = -np.log([sizes[f] for f in features])
    nf = sum(r.failed_tests for r in records)
    npass = sum(r.passed_tests for r in records)
    missed = np.array([sum(r.failed_tests - r.failed_hits.get(k, 0) for r in failed) for k in candidates])
    pf = nf - missed
    pp = np.array([sum(r.passed_hits.get(k, 0) for r in records) for k in candidates])
    # 对命中后的失效率积分；未命中的失败允许小概率背景失效。
    background = 1 / (nf + npass + 1)
    ll = (betaln(pf + 0.5, pp + 0.5) - betaln(0.5, 0.5)
          + missed * np.log(background) + (npass - pp) * np.log1p(-background))
    masks = np.array([[k in r.present for k in candidates] for r in records], dtype=float)
    likelihoods = {
        "statement": ll,
        "configuration": betaln(len(failed) + 0.5, masks.sum(axis=0) - len(failed) + 0.5),
    }
    output = []
    for method, likelihood in likelihoods.items():
        weights = softmax(likelihood + log_prior)
        scores = np.clip(masks @ weights, 0, 1)
        leader = candidates[int(np.argmax(weights))]
        for r, score in zip(records, scores):
            if r.failed_tests == 0:
                output.append(dict(version=version_name, configuration=r.configuration, method=method,
                                   score=float(score), prediction=int(score >= 0.5),
                                   candidates=len(candidates), leading=":".join(leader)))
    return output


def run(data, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for system, relative in SYSTEMS.items():
        for version in sorted(data.joinpath(*relative).glob("_MultipleBugs_.NOB_1.ID_*")):
            results.extend(dict(system=system, **r) for r in infer(read_version(version), version.name))
    if not results:
        raise ValueError(f"找不到可评估的单缺陷版本 {data}")
    with output.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)
