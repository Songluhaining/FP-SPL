"""独立核验通过后发布报告修正实验主表、中文说明及版本清单。"""
from datetime import datetime
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from .inventory import atomic_json
from . import varcop_report_relabel as implementation
from .localization_study import ROOT, digest, read
from .localization_evaluate import SYSTEMS, NAMES


def main():
    output = implementation.OUTPUT
    checked = read(output / "independently_verified.json")
    assert checked["cases"] == 1054 and checked["removed_products"] == 0
    for name, sha in checked["files"].items():
        assert digest(output / name) == sha
    frame = pd.read_csv(output / "fault_positions.csv")
    overall = pd.read_csv(output / "overall_metrics.csv")
    main_table = overall[overall.metric == "Ochiai"].set_index("method")
    reports = pd.read_csv(output / "report_corrections.csv")
    system_rows = []
    for (system, method), group in frame[frame.metric == "Ochiai"].groupby(["system", "method"], sort=True):
        system_rows.append(dict(system=system, method=method, cases=len(group.version.unique()),
            positions=len(group), mean_rank=float(group.penalized_rank.mean()), mean_exam=float(group.exam.mean()),
            hit10=int(((group.found == 1) & (group['rank'] <= 10)).sum())))
    system_table = pd.DataFrame(system_rows)
    system_table.to_csv(output / "system_overall_metrics.csv", index=False)
    target = ROOT / "experimental_results/六系统语句级缺陷定位比较_报告修正.csv"
    shutil.copyfile(output / "语句级定位比较_报告修正.csv", target)
    references = {}
    for name in ["fp_predictions.csv", "fp_metrics.csv", "manifest.json", "version_manifest.json"]:
        path = implementation.PREVIOUS_OUTPUT / name
        references[name] = dict(path=str(path), sha256=digest(path))
    atomic_json(output / "prediction_sources.json", references)
    proof = read(output / "pilot-reference.json")
    lines = ["# 修正配置报告后的语句级缺陷定位实验", "",
        "本轮完成1054个缺陷案例、32494个配置，使用与前轮完全相同的16790个原本通过配置的识别结果。覆盖六系统单缺陷和银行账户、电梯、邮件、考试数据库双、三缺陷；图处理、压缩系统多缺陷仍暂缓。",
        "此前将假通过配置整体剔除的接入方式不符合用户要求，历史结果保留，仅用于追溯；本轮主结果以独立修正配置报告为准。", "",
        "每个案例、每种方法新增一份config.report.csv，仅将预测假通过配置最后一列从__PASSED__改为__FAILED__。配置数量、顺序、特征选择、原报告与原覆盖不变。定位程序的特征组合读取入口和基础特征判断入口都改为读取对应方法的新报告。",
        "本轮严格只改报告及其读取路径：固定官方VARCOP提交62b413b52bbfba1f86a244ddb9f1daed232e329e，保留逐测试覆盖评分。原版产品级覆盖评分根据实际失败覆盖文件判断产品，未额外改为依赖修正报告；完整语句分数因此不变，改变发生在特征组合、切片候选及报告驱动的基础特征判断。",
        "新标失败而没有实际失败覆盖的产品仍进入特征组合计算，但不伪造失败测试或失败覆盖，不把原通过测试全部改成失败；对应切片不可用配置逐案记录。完整组合日志spc.log与实际进入切片的spc-observed-failures.log都保留。", "",
        "先行验证：银行账户73个单缺陷案例、2482个配置全部保留，未修正六种公式完整与筛选排名全部复现；365份方法报告的特征组合与官方枚举相同；案例14联合修正的候选语句集合与未加速官方切片工具相同。其后才扩展全量运行。",
        "使用同一组预测、原始观测和定位参数。主公式Ochiai，另报五种公式；权重0.5、归一化和算术平均保持不变。两种正例学习抽样设置分别报告。排名与阈值不根据定位真值调整。", "",
        "|输入条件|前10定位位置数|平均检查语句数|平均检查比例|改善/相同/退化位置数|",
        "|---|---:|---:|---:|---:|"]
    for method in ["original", "joint", "CLAP", "PULP", "PULP-default"]:
        row = main_table.loc[method]
        lines.append(f"|{NAMES[method]}|{int(row.hit10)}|{row.mean_rank:.3f}|{row.mean_exam:.3%}|{int(row.improved)}/{int(row.tied)}/{int(row.worsened)}|")
    lines.extend(["", "各系统汇总如下。数值为平均检查比例，越低越好；四个已完成多缺陷系统合并展示该系统单、双、三缺陷，图处理、压缩仅单缺陷。各缺陷层单独结果见同目录中文CSV。", "",
        "|系统|未修正|联合贝叶斯|有监督分类|正例学习等量抽样|正例学习默认5%|",
        "|---|---:|---:|---:|---:|---:|"])
    for system in SYSTEMS:
        table = system_table[system_table.system == system].set_index("method")
        values = [f"{table.loc[method, 'mean_exam']:.3%}" for method in ["original", "joint", "CLAP", "PULP", "PULP-default"]]
        lines.append("|"+SYSTEMS[system]+"|"+"|".join(values)+"|")
    joint = main_table.loc['joint']
    baseline = main_table.loc['original']
    lines.extend(["", f"联合修正前10定位数由{int(baseline.hit10)}变为{int(joint.hit10)}，平均检查比例由{baseline.mean_exam:.6%}变为{joint.mean_exam:.6%}。该结果只说明本轮报告修正协议下的实际效果，不由识别准确率直接推导定位收益，也不据此修改模型或阈值。",
        f"全量独立核验：61410条逐故障位置评价记录全部复算通过，五种条件分母一致；共核验162470行配置报告，剔除配置数为0；未修正的12282条逐位置、逐公式名次与历史参考完全一致。联合修正状态数为{int(reports[reports.method == 'joint'].relabeled.sum())}，并逐案与保存预测核对。",
        "原始文件摘要逐案复核；完整排名计算不读取注入日志，评价阶段才读取故障位置。统计采用最坏连续并列名次及固定原始语句全集分母。无排名位置按原始语句数加一惩罚，未丢弃任何位置。",
        "识别模型未重新训练。识别明细、训练划分及识别指标通过prediction_sources.json引用前轮已归档文件；这些识别指标与前轮一致，变化的是定位接入协议。",
        "来源报告统计：365个案例有config.report.csv，689个仅有config.report.csv.done；两者同时存在时字节完全一致。本轮派生报告统一命名config.report.csv。", "",
        "复现命令见仓库fp_research/REPORT_RELABEL_PROTOCOL.md。主要文件：语句级定位比较_报告修正.csv、fault_positions.csv、report_corrections.csv、independently_verified.json；逐案修正报告保存在原案例.fp_research/statement_localization_report_relabel/方法/下。", ""])
    (output / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")
    entry = ROOT / "experimental_results/语句级实验结果入口.md"
    entry.write_text("# 当前语句级定位结果\n\n当前采用用户要求的配置报告修正协议，最新主表为[六系统语句级缺陷定位比较_报告修正.csv](六系统语句级缺陷定位比较_报告修正.csv)，说明见[实验报告](statement_localization_report_relabel/实验报告.md)。\n\n旧文件六系统语句级缺陷定位比较.csv及statement_localization目录对应整配置剔除协议，不符合本轮要求，仅保留历史追溯，不与当前结果混用。旧归档文件保持原字节及摘要。\n", encoding="utf-8")
    source = {str(p.relative_to(ROOT)):digest(p) for p in [ROOT / "fp_research/varcop_report_relabel.py",
        ROOT / "fp_research/report_relabel_study.py", ROOT / "fp_research/checks/test_report_relabel.py",
        ROOT / "fp_research/checks/report_relabel_reference.py", ROOT / "fp_research/checks/verify_report_relabel.py",
        ROOT / "fp_research/varcop_score_lookup.py", Path(__file__)]}
    version = subprocess.check_output(["git", "-c", "safe.directory="+ROOT.as_posix(), "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    atomic_json(output / "version_manifest.json", dict(execution_commit=version,
        algorithm_commit="cf5a1aa7276972bc6cbc0b1e22bd934676020322", source_sha256=source,
        protocol=implementation.PROTOCOL, cases=1054, positions=2047, sources=implementation.sources(),
        pilot_reference=proof, created=datetime.now().astimezone().isoformat()))
    atomic_json(output / "completion.json", dict(cases=1054, positions=2047,
        independent_verification_sha256=digest(output / "independently_verified.json"),
        files={str(p.relative_to(ROOT)):digest(p) for p in [output / "实验报告.md", target, entry,
            output / "fault_positions.csv", output / "metrics.csv", output / "overall_metrics.csv"]}))
    print(system_table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
