"""从逐语句排名生成定位评价，不参与识别推断。"""
import gzip
import json
from pathlib import Path
import pandas as pd
from ..localization import engine as implementation
from ..io import read,digest,atomic_json
from ..experiments.catalog import HISTORY,METHODS,SYSTEMS,NAMES
from .ranking import fault_positions,locate,layered_position

def evaluate(selected, output_root):
    output_root=Path(output_root)
    complete=read(output_root/'manifest.json').get('full_scope',False)
    rows, cases, reports = [], [], []
    for job in selected:
        output = implementation.folder(job)
        proof = read(output / "complete.json")
        if proof["protocol"]["source"] != implementation.sources():
            raise ValueError("计算源码与核验源码不一致")
        for name, sha in proof["files"].items():
            if digest(output / name) != sha:
                raise ValueError("修正报告或定位产物改变")
        if implementation.backend.input_fingerprint(Path(job["path"]))[0] != proof["protocol"]["input"]:
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
    reference = pd.read_csv(HISTORY / "fault_positions.csv")
    expected = reference[(reference.method == "original") & reference.version.isin({j['version'] for j in selected})]
    a = frame[frame.method == "original"].merge(expected, on=keys+["method"], suffixes=("_new", "_old"), validate="one_to_one")
    assert (a.penalized_rank_new == a.penalized_rank_old).all()
    prefix = output_root
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
        historical_original_rows_checked=len(a), full_scope=complete,
        files={p.name:digest(p) for p in prefix.glob("*.csv")})
    atomic_json(prefix / "verified.json", receipt)
    print(overall[overall.metric == "Ochiai"].to_string(index=False), flush=True)
    return prefix, overall
