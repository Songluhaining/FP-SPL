"""核验完成后，从逐位置明细生成总体比较及修正行为说明。"""
from pathlib import Path

import pandas as pd

from .inventory import atomic_json
from .localization_study import OUTPUT, ROOT, read, digest, work_folder
from .localization_evaluate import SYSTEMS, NAMES


def main():
    verified = read(OUTPUT / "verified.json")
    assert verified["cases"] == 1054
    assert digest(OUTPUT / "fault_positions.csv") == verified["files"]["fault_positions.csv"]
    frame = pd.read_csv(OUTPUT / "fault_positions.csv")
    summaries = []
    for (metric, method), g in frame.groupby(["metric", "method"]):
        summaries.append(dict(metric=metric, method=method, positions=len(g),
            hit10=int(((g.found == 1) & (g["rank"] <= 10)).sum()),
            mean_exam=float(g.exam.mean()), mean_rank=float(g.penalized_rank.mean()),
            improved=int((g.improvement > 0).sum()), tied=int((g.improvement == 0).sum()),
            worsened=int((g.improvement < 0).sum())))
    totals = pd.DataFrame(summaries)
    totals.to_csv(OUTPUT / "overall_metrics.csv", index=False)
    groups = []
    for (system, method), g in frame[frame.metric == "Ochiai"].groupby(["system", "method"]):
        groups.append(dict(system=system, method=method, positions=len(g),
            hit10=int(((g.found == 1) & (g["rank"] <= 10)).sum()), mean_exam=float(g.exam.mean())))
    pd.DataFrame(groups).to_csv(OUTPUT / "system_overall_metrics.csv", index=False)
    main_table = totals[totals.metric == "Ochiai"].copy()
    assert set(main_table.positions) == {2047}
    human = main_table.replace({"method": NAMES}).rename(columns=dict(metric="定位公式", method="修正方法",
        positions="故障位置数", hit10="前10定位位置数", mean_exam="平均检查比例", mean_rank="平均检查语句数",
        improved="改善位置数", tied="相同位置数", worsened="退化位置数"))
    human.to_csv(OUTPUT / "总体语句级定位比较.csv", index=False, encoding="utf-8-sig")
    cases = pd.read_csv(OUTPUT / "cases.csv")
    cases = cases[cases.metric == "Ochiai"].copy()
    jobs = read(OUTPUT / "manifest.json")["jobs"]
    identities = {(j["system"], j["version"]): j for j in jobs}
    cases["original_passing"] = [identities[(r.system, r.version)]["candidates"] for r in cases.itertuples()]
    cases["remaining_passing"] = cases.original_passing-cases.excluded
    before = cases[cases.method == "original"][["system", "version", "first_rank", "all_rank"]].rename(
        columns=dict(first_rank="original_first", all_rank="original_all"))
    selected = cases[cases.method == "joint"].merge(before, on=["system", "version"])
    missing_spcs = []
    for r in selected.itertuples():
        folder = work_folder(identities[(r.system, r.version)]) / "varcop"
        proof = read(folder / "complete.json")
        method = proof["methods"]["joint"].get("reused", "joint")
        missing_spcs.append(int((folder / "original/spc.log").stat().st_size > 0 and
                                (folder / method / "spc.log").stat().st_size == 0))
    selected["lost_nonempty_spcs"] = missing_spcs
    selected.to_csv(OUTPUT / "case_correction_diagnostics.csv", index=False)
    removed_all = selected.remaining_passing == 0
    rows = main_table.set_index("method")
    original, joint = rows.loc["original"], rows.loc["joint"]
    text = ["## 本轮结果解读", "",
        f"主公式下，2047个按案例计数的注入位置中，未修正输入前10条语句定位{int(original.hit10)}个，"
        f"联合修正后定位{int(joint.hit10)}个；平均检查比例由{original.mean_exam:.2%}变为{joint.mean_exam:.2%}（越低越好）。",
        f"联合修正使{int(joint.improved)}个位置改善、{int(joint.tied)}个相同、{int(joint.worsened)}个退化。"
        "当前直接剔除规则没有带来总体定位优势，六种公式的总体表均已保存，不根据结果更换主公式或调整阈值。", "",
        "|输入修正|前10定位位置数|平均检查比例|平均检查语句数|",
        "|---|---:|---:|---:|"]
    for method in ["original", "joint", "CLAP", "PULP", "PULP-default"]:
        r = rows.loc[method]
        text.append(f"|{NAMES[method]}|{int(r.hit10)}|{r.mean_exam:.2%}|{r.mean_rank:.2f}|")
    text.extend(["", "邮件的部分多缺陷场景有所改善，考试数据库大多保持不变；银行账户多缺陷与电梯场景的整体退化需要重视。",
        f"联合修正在{int(removed_all.sum())}个案例中剔除了全部原本通过配置，其中"
        f"{int((removed_all & (selected.first_rank > selected.original_first)).sum())}个案例的首故障名次退化。"
        f"有{int(selected.lost_nonempty_spcs.sum())}个案例的可疑特征组合由非空变为空。"
        "这些是后续原因核查的线索，本轮对照尚不能区分误删、对照损失和排序归一化变化各自的贡献。",
        "识别质量与如何利用识别结果应分别评价；后续可检验保留通过轨迹并使用后验风险加权的方式，效果需另行实验。",
        "采样耗时来自首次有效推断凭据，其中406个多缺陷案例复用了既有推断；这些数值不构成统一条件下的运行速度比较。",
        "比较方法按版本五折划分，当前结果不用于宣称对未见基础变异的泛化性能。", ""])
    recognition = pd.read_csv(OUTPUT / "fp_metrics.csv")
    bank = recognition[(recognition.system == "bank") & (recognition.method == "joint")]
    assert int(bank.fp.sum()) == 0
    text.extend(["银行账户三个缺陷层中，联合识别相对基准标签的误报数均为0，但双、三缺陷的定位仍有总体退化。"
                 "这说明仅核查误删真通过配置还不够，还需研究如何保留配置中有用的通过测试证据。",
                 "考试数据库双、三缺陷组的主标签没有假通过正例，其准确率不能单独证明假通过检出能力。", ""])
    report = OUTPUT / "实验报告.md"
    report.write_text(report.read_text(encoding="utf-8").split("## 本轮结果解读")[0].rstrip()+"\n\n"+"\n".join(text), encoding="utf-8")
    software = read(OUTPUT / "software.json")
    software["analysis_source_sha256"] = {"fp_research/localization_summary.py": digest(Path(__file__))}
    atomic_json(OUTPUT / "software.json", software)
    completion_path = OUTPUT / "completion.json"
    completion = read(completion_path) if completion_path.exists() else dict(cases=verified["cases"],
        verified=digest(OUTPUT / "verified.json"), files={p.name: digest(p) for p in
            (OUTPUT / "fault_positions.csv", OUTPUT / "metrics.csv", OUTPUT / "语句级定位比较.csv")})
    completion["files"]["实验报告.md"] = digest(report)
    atomic_json(completion_path, completion)
    print(human.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
