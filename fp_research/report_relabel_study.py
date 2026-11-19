"""修正独立配置报告后的完整定位实验，保留此前剔除协议的历史结果。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import os
from pathlib import Path
import traceback

import pandas as pd

from .inventory import atomic_json
from .localization_study import ROOT, OUTPUT as PREVIOUS_OUTPUT, read, digest
from . import varcop_report_relabel as implementation
from .localization_evaluate import SYSTEMS, NAMES, fault_positions, layered_position, locate

OUTPUT = implementation.OUTPUT
METHODS = ["original", "joint", "CLAP", "PULP", "PULP-default"]


def jobs():
    return read(PREVIOUS_OUTPUT / "manifest.json")["jobs"]


def register():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = PREVIOUS_OUTPUT / "manifest.json"
    protocol = dict(protocol=implementation.PROTOCOL, jobs=jobs(),
        previous_manifest=dict(path=str(previous), sha256=digest(previous)),
        correction="每个方法生成独立config.report.csv；只将预测假通过配置的最后一列改为__FAILED__，保留所有配置",
        coverage="原始逐测试成败及覆盖不变；不建立预测失败测试；只修改原程序读取产品报告的环节",
        consumers=["SPCsManager.load_configs", "VarBugManager.is_var_bug_by_config"],
        unaffected=["局部测试覆盖评分", "从实际失败覆盖读取的产品级评分", "完整语句排名"],
        missing_failed_coverage="新标失败但没有实际失败测试覆盖的产品不生成失败轨迹；记录切片不可用产品，保留其配置参与特征组合计算",
        predictions="引用前轮相同逐配置预测，不重训、不选阈值、不读取真值调整修正集合",
        baseline_comparison="同一份原始输入和原始VARCOP参数；五个输入条件均保留全部配置",
        validation="逐案复现未修正六公式的完整与筛选排名；原文件摘要不变；修正只作用最后一列",
        deferred=["GPL-2fault", "GPL-3fault", "ZipMe-2fault", "ZipMe-3fault"])
    target = OUTPUT / "manifest.json"
    if target.exists() and read(target) != protocol:
        raise ValueError("已登记的报告修正协议改变")
    atomic_json(target, protocol)
    return protocol


def perform(job):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    try:
        from .varcop_score_lookup import indexed_lookup
        ranking, _, _ = implementation.legacy.upstream()
        with indexed_lookup(ranking):
            value = implementation.rank_case(job)
        if value != "cached":
            receipt = implementation.folder(job) / "complete.json"
            proof = read(receipt)
            proof["runtime_optimization"] = dict(name="indexed_readonly_local_scores",
                source_sha256=digest(ROOT / "fp_research/varcop_score_lookup.py"),
                every_index_checked_against_original=True)
            atomic_json(receipt, proof)
        atomic_json(implementation.folder(job) / "status.json", dict(ok=True, value=value))
        print(job["system"], job["stratum"], job["version"], value, flush=True)
        return True
    except Exception as error:
        output = implementation.folder(job)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "status.json", dict(ok=False, reason=str(error), traceback=traceback.format_exc()))
        print("FAILED", job["system"], job["stratum"], job["version"], repr(error), flush=True)
        return False


def evaluate(selected, complete=False):
    rows, cases, reports = [], [], []
    for job in selected:
        output = implementation.folder(job)
        proof = read(output / "complete.json")
        if proof["protocol"]["source"] != implementation.sources():
            raise ValueError("计算源码与核验源码不一致")
        for name, sha in proof["files"].items():
            if digest(output / name) != sha:
                raise ValueError("修正报告或定位产物改变")
        if implementation.legacy.input_fingerprint(Path(job["path"]))[0] != proof["protocol"]["input"]:
            raise ValueError("原始源码或覆盖改变")
        original_header, original_rows = implementation.load_report(implementation.report_source(job["path"]))
        identities = {r[0] for r in original_rows}
        expected_faults = fault_positions(Path(job["path"]))
        denominator = len(proof["universe"])
        for method in METHODS:
            info = proof["methods"][method]
            report = output / method / "config.report.csv"
            header, corrected = implementation.load_report(report)
            selected_ids = set(proof["protocol"]["methods"][method])
            assert header == original_header and len(corrected) == len(original_rows)
            assert {r[0] for r in corrected} == identities
            for before, after in zip(original_rows, corrected):
                assert after[:-1] == before[:-1]
                assert after[-1] == ("__FAILED__" if before[0] in selected_ids else before[-1])
            assert info["removed"] == 0 and info["retained"] == len(identities)
            assert info["passed"]+info["failed"] == len(identities)
            assert Path(info["resolved_by_spc"]) == report
            for path, sha in proof["protocol"]["original_reports"].items():
                assert digest(Path(path)) == sha
            reports.append(dict(system=job["system"], stratum=job["stratum"], version=job["version"],
                method=method, original_source=proof["protocol"]["report_source"], corrected_report=str(report),
                report_sha256=digest(report), relabeled=len(selected_ids), removed=0, retained=len(identities),
                passed=info["passed"], failed=info["failed"], isolation=int(info["variability_case"]),
                unavailable_failed_traces=len(info["without_observed_failed_coverage"])))
            with gzip.open(output / method / "rankings.json.gz", "rt", encoding="utf-8") as stream:
                ranks = json.load(stream)
            for metric, lists in ranks.items():
                faults = []
                for statement in expected_faults:
                    full = locate(statement, lists["full"])
                    focused = layered_position(statement, lists["full"], lists["isolated"])
                    rank = focused if info["variability_case"] else full
                    penalty = rank if rank is not None else denominator+1
                    item = dict(system=job["system"], stratum=job["stratum"], version=job["version"],
                        method=method, metric=metric, statement=statement, rank=rank, penalized_rank=penalty,
                        full_rank=full, isolated_rank=focused, exam=min(penalty, denominator)/denominator,
                        found=int(rank is not None), denominator=denominator,
                        varcop_isolation=int(info["variability_case"]), relabeled=len(selected_ids), removed=0)
                    rows.append(item)
                    faults.append(item)
                cases.append(dict(system=job["system"], stratum=job["stratum"], version=job["version"],
                    method=method, metric=metric, faults=len(faults), found=sum(r["found"] for r in faults),
                    first_rank=min(r["penalized_rank"] for r in faults), all_rank=max(r["penalized_rank"] for r in faults),
                    mean_rank=sum(r["penalized_rank"] for r in faults)/len(faults), relabeled=len(selected_ids), removed=0))
    frame, case_frame = pd.DataFrame(rows), pd.DataFrame(cases)
    keys = ["system", "stratum", "version", "metric", "statement"]
    assert not frame.duplicated(keys+["method"]).any()
    ref = frame[frame.method == "original"][keys+["penalized_rank"]].rename(columns={"penalized_rank": "original_rank"})
    frame = frame.merge(ref, on=keys, validate="many_to_one")
    frame["improvement"] = frame.original_rank-frame.penalized_rank
    reference = pd.read_csv(PREVIOUS_OUTPUT / "fault_positions.csv")
    expected = reference[(reference.method == "original") & reference.version.isin({j['version'] for j in selected})]
    a = frame[frame.method == "original"].merge(expected, on=keys+["method"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(a) == len(frame[frame.method == "original"])
    assert (a.penalized_rank_new == a.penalized_rank_old).all()
    prefix = OUTPUT if complete else OUTPUT / ("partial-"+selected[0]["system"]+"-"+str(selected[0]["stratum"]))
    prefix.mkdir(parents=True, exist_ok=True)
    frame.to_csv(prefix / "fault_positions.csv", index=False)
    case_frame.to_csv(prefix / "cases.csv", index=False)
    pd.DataFrame(reports).to_csv(prefix / "report_corrections.csv", index=False)
    aggregates = []
    for fields in (["system", "stratum", "method", "metric"], ["method", "metric"]):
        summaries = []
        for identity, group in frame.groupby(fields, sort=True):
            row = dict(zip(fields, identity))
            row.update(cases=len(group[["system", "version"]].drop_duplicates()), positions=len(group),
                mean_rank=float(group.penalized_rank.mean()), mean_exam=float(group.exam.mean()),
                improved=int((group.improvement > 0).sum()), tied=int((group.improvement == 0).sum()),
                worsened=int((group.improvement < 0).sum()), unranked=int((group.found == 0).sum()))
            for budget in (1, 3, 5, 10):
                row["hit"+str(budget)] = int(((group.found == 1) & (group['rank'] <= budget)).sum())
            summaries.append(row)
        aggregates.append(pd.DataFrame(summaries))
    metrics, overall = aggregates
    metrics.to_csv(prefix / "metrics.csv", index=False)
    overall.to_csv(prefix / "overall_metrics.csv", index=False)
    human = metrics[metrics.metric == "Ochiai"].replace({"system": SYSTEMS, "method": NAMES}).rename(columns={
        "system":"系统", "stratum":"缺陷数", "method":"修正方法", "metric":"定位公式", "cases":"案例数",
        "positions":"故障位置数", "mean_rank":"平均检查语句数", "mean_exam":"平均检查比例",
        "improved":"改善位置数", "tied":"相同位置数", "worsened":"退化位置数", "unranked":"未定位位置数",
        "hit1":"前1定位位置数", "hit3":"前3定位位置数", "hit5":"前5定位位置数", "hit10":"前10定位位置数"})
    human.to_csv(prefix / "语句级定位比较_报告修正.csv", index=False, encoding="utf-8-sig")
    receipt = dict(cases=len(selected), configurations=sum(j["configurations"] for j in selected),
        candidates=sum(j["candidates"] for j in selected), rows=len(frame), methods=METHODS,
        all_products_retained=True, corrected_reports_checked=True, original_inputs_unchanged=True,
        original_rankings_reproduced=True, full_scope=complete,
        files={p.name:digest(p) for p in prefix.glob("*.csv")})
    atomic_json(prefix / "verified.json", receipt)
    print(overall[overall.metric == "Ochiai"].to_string(index=False), flush=True)
    return prefix, overall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["register", "rank", "evaluate", "all"])
    parser.add_argument("--system")
    parser.add_argument("--stratum", type=int)
    parser.add_argument("--version")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    register()
    selected = [j for j in jobs() if (not args.system or j["system"] == args.system)
                and (not args.stratum or j["stratum"] == args.stratum)
                and (not args.version or j["version"] == args.version)]
    if not selected:
        parser.error("没有匹配的案例")
    os.environ["PYTHONHASHSEED"] = "42"
    if args.stage in {"rank", "all"}:
        if not (args.system == "bank" and args.stratum == 1):
            gate = read(OUTPUT / "partial-bank-1/verified.json")
            assert gate["cases"] == 73 and gate["original_rankings_reproduced"]
            assert gate["corrected_reports_checked"] and gate["all_products_retained"]
            reference = read(OUTPUT / "pilot-reference.json")
            assert reference["cases"] == 73 and reference["corrected_reports_checked"] == 365
            assert reference["official_combinations_identical"] and reference["official_uncached_slice_identical"]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(perform, job) for job in selected]
            failed = sum(not future.result() for future in as_completed(futures))
        if failed:
            raise SystemExit(f"{failed}个案例失败，请检查各案例status.json并修复重跑")
    if args.stage in {"evaluate", "all"}:
        evaluate(selected, complete=len(selected) == len(jobs()))


if __name__ == "__main__":
    main()
