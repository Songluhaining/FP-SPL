"""完成归档前核对双场景原始输入、版本清单与六系统统一表的一致性。"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

from fp_research.case_study import ROOT, SYSTEMS, case_versions, results
from fp_research.inventory import atomic_json


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("D:/dataset/whn"))
    parser.add_argument("--baseline-commit", default="56b3b4a")
    args = parser.parse_args()
    folder = ROOT / "experimental_results/gpl_zipme"
    status = json.loads((folder / "pipeline-status.json").read_text(encoding="utf8"))
    if status["stage"] != "主实验、补充实验与补测记录均已核验":
        raise ValueError("计算流程尚未全部完成")
    proofs = {}
    scenarios = {}
    for scenario in ("main", "masked"):
        here = folder / scenario
        proofs[scenario] = json.loads((here / "raw-input-verification.json").read_text(encoding="utf8"))
        receipt = json.loads((here / "registry-verified.json").read_text(encoding="utf8"))
        for name in ("metrics", "predictions"):
            if hashlib.sha256((here / (name + ".csv")).read_bytes()).hexdigest() != receipt[name + "_sha256"]:
                raise ValueError("统计核验后的文件发生变化")
        scenarios[scenario] = receipt
    if proofs["main"]["versions"] != proofs["masked"]["versions"]:
        raise ValueError("两个场景的原始输入摘要不同")
    saved = {(r["system"], r["version"]): r["raw_digest"] for r in proofs["main"]["versions"]}
    counts = {}
    for system in SYSTEMS:
        versions = case_versions(args.data, system)
        counts[system] = len(versions)
        for version in versions:
            for scenario in ("main", "masked"):
                here = results(version, scenario)
                complete = json.loads((here / "complete.json").read_text(encoding="utf8"))
                if complete["raw_digest"] != saved[(system, version.name)]:
                    raise ValueError("案例完成标记与复核输入不符")
                if (here / "failure.json").exists():
                    raise ValueError("案例仍留有未处理的失败标记")
            if not (results(version, "main") / "replay-complete.json").exists():
                raise ValueError("实际补测记录缺失")
    if counts != {"gpl": 105, "zipme": 55} or len(saved) != sum(counts.values()):
        raise ValueError("版本范围有遗漏或变化")
    construction = read_csv(folder / "masked/construction.csv")
    allowed = {"passed", "retained_failure", "ineligible"}
    if any(r["status"] not in allowed for r in construction):
        raise ValueError("补充样本仍有构造错误")
    path = ROOT / "experimental_results/六系统单缺陷实验比较.csv"
    rows = read_csv(path)
    if any(r["完成状态"] != "已完成并复算核验" for r in rows):
        raise ValueError("统一表仍有未完成行")
    original = subprocess.check_output([
        "git", "-c", "safe.directory=" + ROOT.as_posix(), "show",
        args.baseline_commit + ":experimental_results/六系统单缺陷实验比较.csv"], cwd=ROOT)
    old = list(csv.DictReader(io.StringIO(original.decode("utf-8-sig"))))
    keys = ("系统", "方法", "实验场景", "版本划分", "评价子集")
    def identity(row):
        return tuple(row[k] for k in keys)
    now = {identity(r): r for r in rows}
    if len(now) != len(rows):
        raise ValueError("统一表重复")
    preserved = 0
    for row in old:
        if row["完成状态"] != "已完成并复算核验":
            continue
        current = now[identity(row)]
        for key, value in row.items():
            if key not in {"更新时间", "结果归档提交"} and current[key] != value:
                raise ValueError(f"既有结果被改写: {identity(row)} {key}")
        preserved += 1
    primary = [r for r in rows if r["实验场景"] == "原本通过配置"
               and r["版本划分"] == "按缺陷版本五折" and r["评价子集"] == "全部"]
    if len(primary) != 18:
        raise ValueError("六系统三方法主表不完整")
    total = sum(int(r["候选配置数"]) for r in primary if r["方法"] == "贝叶斯诊断")
    if total != 11465:
        raise ValueError("六系统主实验候选总数不符")
    output = dict(versions=counts, verified_scenarios=scenarios,
                  identical_raw_inputs=len(saved), construction_records=len(construction),
                  converted=sum(r["status"] == "passed" for r in construction),
                  unified_rows=len(rows), preserved_previous_rows=preserved,
                  baseline_commit=args.baseline_commit, six_system_candidates=total,
                  unified_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    atomic_json(folder / "completion-verification.json", output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
