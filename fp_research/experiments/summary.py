"""只为已经完成独立核验的当前实验生成简洁中文主表。"""
from pathlib import Path
import importlib.metadata
import subprocess
import sys
import pandas as pd
from ..config import ROOT,tool
from ..io import read,digest,atomic_json
from .catalog import METHODS,NAMES,SYSTEMS


def summarize(output):
    output=Path(output)
    proof=read(output/'independently_verified.json')
    for name,sha in proof['files'].items():
        if digest(output/name)!=sha: raise ValueError('独立核验后的明细发生变化')
    metrics=pd.read_csv(output/'overall_metrics.csv')
    primary=metrics[metrics.metric=='Ochiai'].set_index('method')
    lines=['# 当前配置报告修正定位实验','',
        f"本次选择{proof['cases']}个案例、{proof['positions']}个故障位置；全部所选案例完成并独立核验。是否覆盖全量协议见manifest.json中的full_scope。",
        '每个方法保留全部配置，只把独立报告中预测假通过配置最后一列改成__FAILED__。原始报告、测试成败和覆盖不变。',
        '原程序读取报告的特征组合和基础特征判断使用修正报告；覆盖评分仍使用实际覆盖，不伪造失败测试。',
        '识别预测来源、训练折外预测来源及摘要逐案例保存；未重新选择模型阈值或定位公式。','',
        '|输入条件|前10定位位置数|平均检查语句数|平均检查比例|改善/相同/退化|',
        '|---|---:|---:|---:|---:|']
    for method in METHODS:
        row=primary.loc[method]
        lines.append(f"|{NAMES[method]}|{int(row.hit10)}|{row.mean_rank:.3f}|{row.mean_exam:.3%}|{int(row.improved)}/{int(row.tied)}/{int(row.worsened)}|")
    lines.extend(['','完整逐位置明细见fault_positions.csv；独立复算见independently_verified.json。',
        '本目录是本次新实验，不覆盖experimental_results下已有历史结果；历史复现使用其版本清单中的原始提交。'])
    report=output/'实验报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    command=['git','-c','safe.directory='+ROOT.as_posix()]
    runtime=dict(python=sys.version,execution_commit=subprocess.check_output([*command,'rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output([*command,'status','--porcelain'],cwd=ROOT)),
        packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','threadpoolctl']},
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'fp_research').rglob('*.py')},
        feature_slicing_jar_sha256=digest(tool('varcop_jar')))
    atomic_json(output/'software.json',runtime)
    atomic_json(output/'completion.json',dict(cases=proof['cases'],positions=proof['positions'],
        independent_verification_sha256=digest(output/'independently_verified.json'),report_sha256=digest(report),
        software_sha256=digest(output/'software.json')))
    return report
