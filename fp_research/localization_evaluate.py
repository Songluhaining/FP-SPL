"""从保存的完整语句排名生成定位明细；此阶段才读取注入位置。"""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .inventory import atomic_json
from .localization_study import OUTPUT, manifest, work_folder, digest, read

NAMES = dict(original="未修正", joint="联合贝叶斯修正", CLAP="CLAP修正",
             PULP="PULP等量抽样修正", **{"PULP-default": "PULP默认5%修正"})
SYSTEMS = dict(bank="银行账户", elevator="电梯", email="邮件", examdb="考试数据库", gpl="图处理", zipme="压缩")


def locate(statement, ranked):
    """保持官方次序与最坏的连续(score,secondary)并列，独立于官方查询函数。"""
    for start, row in enumerate(ranked):
        if row[0] == statement:
            end = start
            while end+1 < len(ranked) and tuple(ranked[end][1:]) == tuple(ranked[end+1][1:]):
                end += 1
            return end+1
    return None


def fault_positions(version):
    result = set()
    for line in (version / (version.name + ".mutant.log")).read_text(encoding="utf-8").splitlines():
        location, number, _ = line.split(":", 2)
        result.add(location.rsplit(".", 1)[0] + "." + number)
    return sorted(result)


def layered_position(statement, full, focus):
    found = locate(statement, focus)
    if found is not None:
        return found
    position = locate(statement, full)
    if position is None:
        return None
    isolated = {r[0] for r in focus}
    return len(focus) + position - sum(r[0] in isolated for r in full[:position])


def evaluate(jobs):
    rows, case_rows, summaries = [], [], []
    archive = OUTPUT / "case_receipts"
    for job in jobs:
        v = Path(job["path"])
        folder = work_folder(job) / "varcop"
        proof = read(folder / "complete.json")
        for name, expected in proof["files"].items():
            if digest(folder / name) != expected:
                raise ValueError("保存的语句排名或输入改变 " + str(folder / name))
        faults = fault_positions(v)
        universe = set(proof["universe"])
        if not faults or not universe:
            raise ValueError("故障位置或原始语句全集为空")
        rankings = {}
        for method, detail in proof["methods"].items():
            alias = detail.get("reused", method)
            if alias not in rankings:
                with gzip.open(folder / alias / "rankings.json.gz", "rt", encoding="utf-8") as stream:
                    rankings[alias] = json.load(stream)
            excluded = proof["protocol"]["methods"][method]
            for metric, data in rankings[alias].items():
                full, focus = data["full"], data["isolated"]
                if len(full) != len({r[0] for r in full}) or len(focus) != len({r[0] for r in focus}):
                    raise ValueError("重复语句排名")
                for statement in faults:
                    full_rank = locate(statement, full)
                    isolated_rank = layered_position(statement, full, focus)
                    rank = isolated_rank if proof["variability_case"] else full_rank
                    penalty = rank if rank is not None else len(universe)+1
                    rows.append(dict(system=job["system"], stratum=job["stratum"], version=job["version"],
                        method=method, metric=metric, statement=statement, rank=rank, penalized_rank=penalty,
                        exam=min(penalty, len(universe))/len(universe),
                        found=int(rank is not None), in_original_universe=int(statement in universe),
                        full_rank=full_rank, isolated_rank=isolated_rank, denominator=len(universe),
                        varcop_isolation=int(proof["variability_case"]), excluded=len(excluded),
                        insufficient_predictions=proof["unavailable"].get(method, 0)))
                chosen = rows[-len(faults):]
                ranks = [r["penalized_rank"] for r in chosen]
                case_rows.append(dict(system=job["system"], stratum=job["stratum"], version=job["version"],
                    method=method, metric=metric, faults=len(faults), found=sum(r["found"] for r in chosen),
                    first_rank=min(ranks), all_rank=max(ranks), mean_rank=float(np.mean(ranks)),
                    first_exam=min(r["exam"] for r in chosen), all_exam=max(r["exam"] for r in chosen),
                    mean_exam=float(np.mean([r["exam"] for r in chosen])), excluded=len(excluded),
                    insufficient_predictions=proof["unavailable"].get(method, 0)))
        target = archive / job["system"] / job["version"]
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "complete.json", target / "complete.json")
    frame = pd.DataFrame(rows)
    cases = pd.DataFrame(case_rows)
    key = ["system", "stratum", "version", "metric", "statement"]
    reference = frame[frame.method == "original"][key+["penalized_rank"]].rename(
        columns={"penalized_rank": "original_rank"})
    frame = frame.merge(reference, on=key, how="left", validate="many_to_one")
    frame["improvement"] = frame.original_rank - frame.penalized_rank
    for identity, group in frame.groupby(["system", "stratum", "method", "metric"], sort=True):
        system, stratum, method, metric = identity
        selected = cases[(cases.system == system) & (cases.stratum == stratum) &
                         (cases.method == method) & (cases.metric == metric)]
        summary = dict(system=system, stratum=int(stratum), method=method, metric=metric,
            cases=len(selected), fault_positions=len(group), unranked=int((group.found == 0).sum()),
            absent_from_universe=int((group.in_original_universe == 0).sum()),
            mean_rank=float(group.penalized_rank.mean()), mean_exam=float(group.exam.mean()),
            first_rank=float(selected.first_rank.mean()), all_rank=float(selected.all_rank.mean()),
            improved=int((group.improvement > 0).sum()), tied=int((group.improvement == 0).sum()),
            worsened=int((group.improvement < 0).sum()), removed_products=int(selected.excluded.sum()),
            insufficient_predictions=int(selected.insufficient_predictions.sum()))
        for budget in [1, 3, 5, 10]:
            summary["hit"+str(budget)] = int(((group.found == 1) & (group["rank"] <= budget)).sum())
            summary["cases_hit"+str(budget)] = int(((selected.first_rank <= budget) & (selected.found > 0)).sum())
            summary["cases_all"+str(budget)] = int(((selected.all_rank <= budget) &
                                                   (selected.found == selected.faults)).sum())
        summaries.append(summary)
    metrics = pd.DataFrame(summaries)
    frame.to_csv(OUTPUT / "fault_positions.csv", index=False)
    cases.to_csv(OUTPUT / "cases.csv", index=False)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    columns = dict(system="系统", stratum="缺陷数", method="修正方法", metric="定位公式", cases="案例数",
        fault_positions="故障位置数", unranked="未定位位置数", absent_from_universe="原始语句全集缺失数",
        mean_rank="平均检查语句数", mean_exam="平均检查比例", first_rank="首故障平均检查数",
        all_rank="全部故障平均检查数", improved="改善位置数", tied="相同位置数", worsened="退化位置数",
        removed_products="排除通过配置数", insufficient_predictions="无法评分配置数")
    for budget in [1, 3, 5, 10]:
        columns["hit"+str(budget)] = f"前{budget}定位位置数"
        columns["cases_hit"+str(budget)] = f"前{budget}找到任一故障案例数"
        columns["cases_all"+str(budget)] = f"前{budget}找到全部故障案例数"
    human = metrics.replace({"system": SYSTEMS, "method": NAMES}).rename(columns=columns)
    human.to_csv(OUTPUT / "语句级定位比较.csv", index=False, encoding="utf-8-sig")
    main = human[human["定位公式"] == "Ochiai"]
    main.to_csv(OUTPUT.parent / "六系统语句级缺陷定位比较.csv", index=False, encoding="utf-8-sig")
    report = [
        "# 修正假通过配置后的语句级定位结果", "",
        f"全部{len(jobs)}个案例完成。当前联合方法按案例独立推断，各基线按系统和缺陷数进行版本五折预测。",
        "修正操作排除预测为假通过的通过产品，随后重新计算VARCOP定位。主表统一使用Ochiai。",
        "所有位置采用官方最坏连续并列规则。无法进入排名的位置标为未定位，平均检查数用原始语句总数加一惩罚，检查比例计1。", "",
        "|系统|缺陷数|修正方法|案例数|故障位置数|前10定位数|平均检查数|平均检查比例|改善/相同/退化|",
        "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in main.to_dict("records"):
        report.append(f"|{r['系统']}|{r['缺陷数']}|{r['修正方法']}|{r['案例数']}|{r['故障位置数']}|"
                      f"{r['前10定位位置数']}|{r['平均检查语句数']:.2f}|{r['平均检查比例']:.2%}|"
                      f"{r['改善位置数']}/{r['相同位置数']}/{r['退化位置数']}|")
    report.extend(["", "缺分的分类预测保留原始通过配置，其定位结果反映未修正这些配置的情况。",
                   "原始系统语句全集分母固定，避免删除产品改变分母造成表观改善。基础特征即可失败的案例使用官方关闭筛选的分支。",
                   "全部定位公式及位置明细见同目录CSV。"])
    (OUTPUT / "实验报告.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    atomic_json(OUTPUT / "evaluated.json", dict(cases=len(jobs), rows=len(frame),
        files={p.name: digest(p) for p in (OUTPUT / "fault_positions.csv", OUTPUT / "cases.csv",
                                         OUTPUT / "metrics.csv", OUTPUT / "语句级定位比较.csv")}))
    print(main.to_string(index=False), flush=True)


if __name__ == "__main__":
    evaluate(manifest())
