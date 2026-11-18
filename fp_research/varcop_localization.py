"""调用固定提交的 VARCOP；仅适配路径、配置排除、缓存和结果导出。"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
import csv
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pandas as pd

from .inventory import atomic_json
from .localization_study import ROOT, OUTPUT, UPSTREAM, work_folder, inference_folder, read, digest, METRICS

JAVA = Path("E:/Program Files/Java/jdk-1.8/bin/java.exe")
JAR = ROOT / "result/varcop_source/feature-slicing.jar"
VENDOR = ROOT / "fp_research/vendor/varcop"


def source_hashes():
    paths = [*VENDOR.rglob("*.py"), ROOT / "fp_research/varcop_localization.py",
             ROOT / "fp_research/localization_study.py", ROOT / "fp_research/varcop_spcs.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def upstream():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from util import FileManager
    FileManager.join_path = lambda *p: Path(os.path.join(*p)).as_posix()
    from ranking import RankingManager as ranking
    from ranking import VariantLevelRankingManager as products
    from spc import SPCsManager
    logging.getLogger().setLevel(logging.WARNING)
    return ranking, products, SPCsManager


def input_fingerprint(version):
    paths = [version / "config.report.csv.done"]
    for variant in sorted((version / "variants").iterdir()):
        paths.extend(sorted((variant / "coverage").glob("spectrum_*_coverage.xml")))
        paths.extend(sorted((variant / "src").rglob("*.java")))
    hashes = {str(p.relative_to(version)): digest(p) for p in paths}
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return key, hashes


def correction_sets(job):
    version = Path(job["path"])
    with (version / "config.report.csv.done").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        configs = list(reader)
    passed = {r[0].strip() for r in configs if r[-1].strip() == "__PASSED__"}
    joint_path = inference_folder(job) / "joint.csv"
    joint = pd.read_csv(joint_path)
    if set(joint.configuration) != passed or joint.configuration.duplicated().any():
        raise ValueError("联合预测候选缺失或重复")
    choices = {"original": set(), "joint": set(joint.loc[joint.prediction == 1, "configuration"])}
    references = {"joint": dict(path=str(joint_path), sha256=digest(joint_path))}
    unavailable = {}
    for name in ("CLAP", "PULP", "PULP-default"):
        path = OUTPUT / f"{name}.csv"
        full = pd.read_csv(path)
        data = full[(full.system == job["system"]) & (full.version == job["version"])]
        if set(data.configuration) != passed or data.configuration.duplicated().any():
            raise ValueError(f"{name} 预测候选缺失或重复")
        choices[name] = set(data.loc[data.prediction == 1, "configuration"])
        unavailable[name] = int(data.score.isna().sum())
        references[name] = dict(path=str(path), sha256=digest(path))
    if any(not p <= passed for p in choices.values()):
        raise ValueError("修正集合包含非通过配置")
    return header, configs, choices, references, unavailable


@contextmanager
def excluded_products(ranking, products, excluded):
    old_ranking_dirs = ranking.get_all_variant_dirs
    old_product_dirs = products.get_all_variant_dirs
    old_list = ranking.list_dir

    def dirs(path):
        return [p for p in old_ranking_dirs(path) if Path(p).name not in excluded]

    def listing(path, *args, **kwargs):
        result = old_list(path, *args, **kwargs)
        if Path(path).name == "variants":
            result = [v for v in result if Path(v).name not in excluded]
        return result

    ranking.get_all_variant_dirs = dirs
    products.get_all_variant_dirs = dirs
    ranking.list_dir = listing
    try:
        yield
    finally:
        ranking.get_all_variant_dirs = old_ranking_dirs
        products.get_all_variant_dirs = old_product_dirs
        ranking.list_dir = old_list


def isolate(version, output, excluded, configs, header, spc):
    features = header[1:-1]
    passed, failed, names = [], [], {}
    for row in configs:
        cid = row[0].strip()
        if cid in excluded:
            continue
        flags = [v.strip() == "T" for v in row[1:-1]]
        names[tuple(flags)] = cid
        (passed if row[-1].strip() == "__PASSED__" else failed).append(flags)
    report = output / "observations.csv"
    with report.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(r for r in configs if r[0].strip() not in excluded)
    log = output / "spc.log"
    from .varcop_spcs import detect
    detect(features, passed, failed, names, (version / "variants").as_posix(), log)
    sliced = output / "slicing.json"
    if log.stat().st_size:
        command = [str(JAVA), "-Xmx512m", "-XX:+UseSerialGC",
                   "-Djava.io.tmpdir="+str(output),
                   "-Dspc_path="+str(log), "-Dslicing_output_path="+str(sliced),
                   "-Dcoverage_file_name=spectrum_failed_coverage.xml", "-jar", str(JAR)]
        with (output / "slice-run.log").open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=900)
        raw = read(sliced)
    else:
        atomic_json(sliced, {})
        raw = {}
    data = {}
    for cid, statements in raw.items():
        if cid not in names.values() or cid in excluded:
            raise ValueError("特征切片出现不可用配置")
        failed_path = version / "variants" / cid / "coverage/spectrum_failed_coverage.xml"
        executed = {line.get("featureClass")+"."+line.get("featureLineNum")
                    for line in ET.parse(failed_path).iterfind(".//package/file/line")
                    if int(line.get("count", 0)) > 0}
        data[cid] = {key: value for key, value in statements.items() if key in executed}
    return data


def variability_case(configs, header, system):
    bases = {"Base", "Compress"} if system == "zipme" else {"Base"}
    for row in configs:
        if row[-1].strip() == "__FAILED__":
            enabled = {f for f, flag in zip(header[1:-1], row[1:-1]) if flag.strip() == "T"}
            if enabled <= bases:
                return False
    return True


def rank_case(job):
    version = Path(job["path"])
    output = work_folder(job) / "varcop"
    output.mkdir(parents=True, exist_ok=True)
    header, configs, choices, references, unavailable = correction_sets(job)
    fingerprint, input_files = input_fingerprint(version)
    source = source_hashes()
    protocol = dict(input=fingerprint, source=source, references=references, upstream=UPSTREAM,
                    feature_slicing_sha256=digest(JAR), methods={m: sorted(s) for m, s in choices.items()},
                    metrics=METRICS, alpha=.5, normalization="ENABLE_NORMALIZATION",
                    aggregation="AGGREGATION_ARITHMETIC_MEAN", coverage_filter=0.)
    receipt = output / "complete.json"
    if receipt.exists():
        old = read(receipt)
        if old["protocol"] == protocol and all(digest(output / p) == sha for p, sha in old["files"].items()):
            return "cached"
    ranking, products, spc = upstream()
    # XML trees are treated as immutable by the public implementation.
    original_parse = ET.parse
    ET.parse = lru_cache(maxsize=512)(ET.parse)
    try:
        all_statements, failing_statements = ranking.get_executed_stms_of_the_system(
            version.as_posix(), "", 0.)
        universe = ranking.get_set_of_stms(all_statements)
        metric_names = [getattr(ranking, m.upper()) for m in METRICS]
        local = ranking.local_ranking_a_suspicious_list(
            version.as_posix(), failing_statements, metric_names, "")
        is_variability = variability_case(configs, header, job["system"])
        results = {}
        identical = {}
        for method, excluded in choices.items():
            identity = tuple(sorted(excluded))
            if identity in identical:
                results[method] = dict(reused=identical[identity])
                continue
            identical[identity] = method
            folder = output / method
            folder.mkdir(exist_ok=True)
            started = time.monotonic()
            isolated = isolate(version, folder, excluded, configs, header, spc)
            with excluded_products(ranking, products, excluded):
                kept_all = {c: data for c, data in all_statements.items() if c not in excluded}
                product_scores = ranking.product_based_assessment(
                    version.as_posix(), kept_all, metric_names, "")
            ranked = {}
            for metric in metric_names:
                full = ranking.global_ranking_a_suspicious_list(
                    kept_all, failing_statements, failing_statements, local[metric],
                    product_scores[metric], metric,
                    ranking.AGGREGATION_ARITHMETIC_MEAN, ranking.NORMALIZATION_ENABLE, .5)
                focus = ranking.global_ranking_a_suspicious_list(
                    kept_all, failing_statements, isolated, local[metric],
                    product_scores[metric], metric,
                    ranking.AGGREGATION_ARITHMETIC_MEAN, ranking.NORMALIZATION_ENABLE, .5) if isolated else []
                ranked[metric] = dict(full=full, isolated=focus)
            with gzip.open(folder / "rankings.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(ranked, stream, separators=(",", ":"), allow_nan=False)
            results[method] = dict(folder=method, removed=len(excluded), remaining=len(configs)-len(excluded),
                                   isolated_statements=len(ranking.get_set_of_stms(isolated)),
                                   seconds=time.monotonic()-started)
        if input_fingerprint(version)[0] != fingerprint:
            raise ValueError("定位操作改变原始输入")
        files = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name in {"rankings.json.gz", "observations.csv", "spc.log", "slicing.json"}}
        atomic_json(receipt, dict(protocol=protocol, input_files=input_files, universe=universe,
                                 variability_case=is_variability, methods=results,
                                 unavailable=unavailable, files=files))
        return "verified "+str({m: len(s) for m, s in choices.items()})
    finally:
        ET.parse = original_parse
