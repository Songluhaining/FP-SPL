"""保留全部配置，仅修改独立报告的产品状态；逐测试覆盖保持原样。"""
from contextlib import contextmanager
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from ..io import atomic_json
from . import common as backend
from ..config import ROOT
from ..experiments.catalog import HISTORY as PREVIOUS_OUTPUT,METRICS,UPSTREAM
from ..io import read,digest

SCENARIO = "localization"
PROTOCOL = "report-last-column-only-v2-refactor"


def folder(job):
    return Path(job["path"]) / ".fp_research" / SCENARIO / job.get("run_name", "manual")


def report_source(version):
    version = Path(version)
    source = version / "config.report.csv"
    if not source.is_file():
        source = version / "config.report.csv.done"
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def load_report(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header, rows = next(reader), list(reader)
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("配置报告中存在重复配置")
    if any(len(r) != len(header) or r[-1] not in {"__PASSED__", "__FAILED__"} for r in rows):
        raise ValueError("配置报告的列数或状态无效")
    return header, rows


def write_report(source, target, selected):
    """逐行替换最后一列，保留其余字节、配置次序及全部配置。"""
    source, target, selected = Path(source), Path(target), set(selected)
    if source.resolve() == target.resolve():
        raise ValueError("禁止覆盖原始报告")
    header, rows = load_report(source)
    passed = {r[0] for r in rows if r[-1] == "__PASSED__"}
    if not selected <= passed:
        raise ValueError("只允许修正原本通过的已知配置")
    raw = source.read_bytes()
    lines = raw.splitlines(keepends=True)
    if len(lines) != len(rows)+1:
        raise ValueError("报告含多行字段，不允许按行改写")
    patched = [lines[0]]
    for row, line in zip(rows, lines[1:]):
        if row[0] in selected:
            field = line.rsplit(b",", 1)
            if len(field) != 2 or not field[1].startswith(b"__PASSED__"):
                raise ValueError("报告最后一列不是预期的未加引号状态")
            line = field[0]+b","+field[1].replace(b"__PASSED__", b"__FAILED__", 1)
        patched.append(line)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".csv.tmp")
    temporary.write_bytes(b"".join(patched))
    temporary.replace(target)
    new_header, changed = load_report(target)
    if new_header != header or len(changed) != len(rows):
        raise ValueError("修正报告改变配置或表头")
    for before, after in zip(rows, changed):
        expected = before[:-1]+["__FAILED__" if before[0] in selected else before[-1]]
        if after != expected:
            raise ValueError("修正报告改变了状态以外的内容")
    return header, changed


@contextmanager
def use_report(version, report):
    """同时替换公共路径解析及已导入的路径函数；离开时恢复。"""
    ranking, products, spc = backend.upstream()
    from util import FileManager
    from ranking import VarBugManager
    modules = (FileManager, spc, VarBugManager)
    original = [(module, module.get_model_configs_report_path) for module in modules]
    expected = Path(version).resolve()

    def resolve(project):
        if Path(project).resolve() != expected:
            raise ValueError("报告替换只允许访问当前案例")
        return str(Path(report))

    for module, _ in original:
        module.get_model_configs_report_path = resolve
    try:
        yield spc, VarBugManager
    finally:
        for module, function in original:
            module.get_model_configs_report_path = function


def sources():
    result = backend.source_hashes()
    result["fp_research/localization/engine.py"] = digest(Path(__file__))
    return result


def write_spcs(version, report, target):
    """由官方配置读取函数读取修正文件，然后计算特征组合。"""
    from .spcs import detect
    with use_report(version, report) as (spc, _):
        resolved = spc.get_model_configs_report_path(str(version))
        _, rows = load_report(report)
        features, names, passed, failed = spc.load_configs(
            resolved, {row[0]: 1. for row in rows}, 0.)
        # 官方读取函数包含最后一列表头，组合计算只使用布尔特征位置。
        detect(features, passed, failed, names, (version / "variants").as_posix(), target)
    return dict(path=str(report), sha256=digest(report), passed=len(passed), failed=len(failed),
                configurations=len(rows), resolved_by_spc=resolved)


def observed_spcs(version, original, target):
    """没有实际失败覆盖的产品不伪造失败轨迹；保存未进入切片的产品。"""
    included, absent = [], set()
    for line in original.read_text(encoding="utf-8").splitlines(keepends=True):
        _, source = line.split(";", 1)
        variant = Path(source.strip()).parent
        if (variant / "coverage/spectrum_failed_coverage.xml").is_file():
            included.append(line)
        else:
            absent.add(variant.name)
    target.write_text("".join(included), encoding="utf-8")
    return sorted(absent)


def rank_case(job):
    from .slicing import batch, locked
    version, output = Path(job["path"]), folder(job)
    output.mkdir(parents=True, exist_ok=True)
    with locked(output / "rank.lock"):
        source = report_source(version)
        header, rows, choices, references, unavailable = backend.correction_sets(job)
        if load_report(source) != (header, rows):
            raise ValueError("原始csv与预测所用报告内容不一致")
        fingerprint, input_files = backend.input_fingerprint(version)
        original_reports = {str(p): digest(p) for p in
                            (version / "config.report.csv", version / "config.report.csv.done") if p.exists()}
        protocol = dict(name=PROTOCOL, source=sources(), input=fingerprint,
            report_source=str(source), original_reports=original_reports, references=references,
            methods={method: sorted(ids) for method, ids in choices.items()}, upstream=UPSTREAM,
            feature_slicing_sha256=digest(backend.JAR), metrics=METRICS, alpha=.5,
            normalization="ENABLE_NORMALIZATION", aggregation="AGGREGATION_ARITHMETIC_MEAN",
            product_removal=False, coverage_relabel=False, scope="only_report_readers")
        receipt = output / "complete.json"
        if receipt.exists():
            previous = read(receipt)
            if previous["protocol"] == protocol and all(digest(output/p) == sha for p, sha in previous["files"].items()):
                return "cached"
        started = time.monotonic()
        ranking, products, _ = backend.upstream()
        all_statements, failing = ranking.get_executed_stms_of_the_system(version.as_posix(), "", 0.)
        if set(all_statements) != {r[0] for r in rows}:
            raise ValueError("保留的产品数与完整配置报告不一致")
        metric_names = [getattr(ranking, metric.upper()) for metric in METRICS]
        local = ranking.local_ranking_a_suspicious_list(version.as_posix(), failing, metric_names, "")
        product_scores = ranking.product_based_assessment(version.as_posix(), all_statements, metric_names, "")
        full = {metric: ranking.global_ranking_a_suspicious_list(
            all_statements, failing, failing, local[metric], product_scores[metric], metric,
            ranking.AGGREGATION_ARITHMETIC_MEAN, ranking.NORMALIZATION_ENABLE, .5) for metric in metric_names}
        plans, details = {}, {}
        for method, ids in choices.items():
            target = output / method
            report = target / "config.report.csv"
            write_report(source, report, ids)
            info = write_spcs(version, report, target / "spc.log")
            info["without_observed_failed_coverage"] = observed_spcs(
                version, target / "spc.log", target / "spc-observed-failures.log")
            with use_report(version, report) as (_, branches):
                bases = ["Base", "Compress"] if job["system"] == "zipme" else ["Base"]
                info["variability_case"] = bool(branches.is_var_bug_by_config(version.as_posix(), bases))
            info["relabeled"] = len(ids)
            info["removed"] = 0
            info["retained"] = len(rows)
            details[method] = info
            plans[method] = (target / "spc-observed-failures.log", target / "slicing.json")
        batch(version, plans, output)
        for method, info in details.items():
            target = output / method
            isolated = backend.read_isolation(version, target / "slicing.json", set(), {r[0] for r in rows})
            ranked = {}
            for metric in metric_names:
                focus = ranking.global_ranking_a_suspicious_list(
                    all_statements, failing, isolated, local[metric], product_scores[metric], metric,
                    ranking.AGGREGATION_ARITHMETIC_MEAN, ranking.NORMALIZATION_ENABLE, .5) if isolated else []
                ranked[metric] = dict(full=full[metric], isolated=focus)
            with gzip.open(target / "rankings.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(ranked, stream, separators=(",", ":"), allow_nan=False)
            info["isolated_statements"] = len(ranking.get_set_of_stms(isolated))
        # 未修正条件必须复现上一轮完整排名，防止更换协议时混入其他改动。
        reference = backend.work_folder(job) / "varcop/original/rankings.json.gz"
        from ..evaluation.ranking import assert_same_ranking
        with gzip.open(output / "original/rankings.json.gz", "rt", encoding="utf-8") as stream:
            current_original = json.load(stream)
        if reference.exists():
            with gzip.open(reference, "rt", encoding="utf-8") as stream:
                previous_original = json.load(stream)
            for metric in metric_names:
                for kind in ("full", "isolated"):
                    assert_same_ranking(previous_original[metric][kind], current_original[metric][kind])
        if backend.input_fingerprint(version)[0] != fingerprint or any(digest(Path(p)) != sha for p, sha in original_reports.items()):
            raise ValueError("原始报告、源码或覆盖被修改")
        files = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name in {"config.report.csv", "spc.log", "spc-observed-failures.log", "slicing.json", "rankings.json.gz"}}
        proof = dict(protocol=protocol, universe=ranking.get_set_of_stms(all_statements),
                     methods=details, unavailable=unavailable, input_files=input_files,
                     original_rankings_reproduced=True if reference.exists() else None, original_inputs_unchanged=True,
                     seconds=time.monotonic()-started, files=files)
        atomic_json(receipt, proof)
        return "verified "+str({m: d["relabeled"] for m, d in details.items()})
