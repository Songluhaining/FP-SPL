from pathlib import Path
from typing import Dict, List

import pandas as pd

from pyutils.SuspiciousStatementManager import get_suspicious_statement_varcop
from pyutils.ranking.RankingManager import global_ranking_a_suspicious_list, get_executed_stms_of_the_system, \
    product_based_assessment, local_ranking_a_suspicious_list, locate_multiple_bugs, get_set_of_stms
from pyutils.ranking.Spectrum_Expression import *
from pyutils.ranking.VarBugManager import is_var_bug_by_config

def dump_stmt_sbfl_priors_multi(suspicious_value_by_varcop: dict, mutated_project_dir):
    """
    suspicious_value_by_varcop: dict[metric_name -> dict[stmt_key -> score]]
      e.g. suspicious_value_by_varcop["Op2"]["DailyLimit.Account.33"] = 0.12

    输出：mutated_project_dir/stmt_sbfl_priors.txt
    每行：
      <stmt_key> Tarantula=0.000000 Ochiai=... Op2=... ...
    分数会被 clamp 到 [0,1]，并对每个 metric 做 min-max 归一化到 [0,1]
    （如果该 metric 全部相等，则该 metric 全部写 0.5）
    """
    mutated_project_dir = Path(mutated_project_dir)
    mutated_project_dir.mkdir(parents=True, exist_ok=True)
    out_path = mutated_project_dir / "stmt_sbfl_priors.txt"

    if not suspicious_value_by_varcop:
        out_path.write_text("", encoding="utf-8")
        print("[SBFL_PRIOR] empty metric dict, wrote empty file:", out_path)
        return

    metrics = list(suspicious_value_by_varcop.keys())

    # 收集所有 stmt_key 的全集（保证每行指标齐全）
    all_keys = set()
    for m in metrics:
        all_keys.update(suspicious_value_by_varcop.get(m, {}).keys())
    all_keys = sorted(all_keys)

    # 对每个 metric 做归一化：dict[metric][key] -> norm_score
    norm = {m: {} for m in metrics}
    for m in metrics:
        raw_map = suspicious_value_by_varcop.get(m, {}) or {}
        vals = []
        for k in all_keys:
            v = raw_map.get(k, 0.0)
            try:
                v = float(v)
            except Exception:
                v = 0.0
            if v < 0.0: v = 0.0
            if v > 1.0: v = 1.0
            vals.append(v)

        if not vals:
            continue
        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            for k in all_keys:
                norm[m][k] = 0.5
        else:
            scale = 1.0 / (vmax - vmin)
            for k, v in zip(all_keys, vals):
                norm[m][k] = (v - vmin) * scale

    # 写文件
    with out_path.open("w", encoding="utf-8") as f:
        for k in all_keys:
            tokens = [k]
            for m in metrics:
                tokens.append(f"{m}={norm[m].get(k, 0.0):.6f}")
            f.write(" ".join(tokens) + "\n")

    print(f"[SBFL_PRIOR] wrote multi-metric stmt priors to {out_path} (N={len(all_keys)}, metrics={len(metrics)})")

def get_sbfl_prio(system_name, mutated_project_dir):
    # if system_name == "ZipMe":
    #     is_a_var_bug = is_var_bug_by_config(mutated_project_dir, ["Base", "Compress"])
    # else:
    #     is_a_var_bug = is_var_bug_by_config(mutated_project_dir, ["Base"])
    sbfl_metrics = [TARANTULA, OCHIAI, OP2, BARINEL, DSTAR, KULCZYNSKI2, M2, HARMONIC_MEAN, ZOLTAR, GEOMETRIC_MEAN]

    output = {}
    for metric in sbfl_metrics:
        output[metric] = pd.DataFrame(columns=["mutated_project_name", "RANK", "EXAM"])
    all_stms_of_the_system, all_stms_in_failing_products = get_executed_stms_of_the_system(
        mutated_project_dir, "", 0.1)
    variant_level_suspiciousness = product_based_assessment(mutated_project_dir, all_stms_of_the_system,
                                                            sbfl_metrics, "")

    local_suspiciousness_of_all_the_system = local_ranking_a_suspicious_list(mutated_project_dir,
                                                                             all_stms_in_failing_products,
                                                                             sbfl_metrics,
                                                                             "")  # 能够得到每个语句的可疑分数
    # suspicious_stms_list = get_suspicious_statement_varcop(mutated_project_dir, 0.1)
    full_ranked_list = {}
    _A = {}
    _B = {}
    suspicious_value_by_varcop = {}
    for metric in sbfl_metrics:
        suspicious_value_by_varcop_metric = {}
        full_ranked_list[metric], _A[metric], _B[metric] = global_ranking_a_suspicious_list(all_stms_of_the_system,
                                                                                            all_stms_in_failing_products,
                                                                                            all_stms_in_failing_products,
                                                                                            local_suspiciousness_of_all_the_system[
                                                                                                metric],
                                                                                            variant_level_suspiciousness[
                                                                                                metric],
                                                                                            metric,
                                                                                            "AGGREGATION_ARITHMETIC_MEAN",
                                                                                            "ENABLE_NORMALIZATION", 0.5)
        for l in range(len(_A[metric])):
            suspicious_value_by_varcop_metric[_A[metric][l]] = _B[metric][l]
        suspicious_value_by_varcop[metric] = suspicious_value_by_varcop_metric

        # ✅ 一次性写入一个文件（一个 stmt_key 一行，后面 10 个 METRIC=value）
        dump_stmt_sbfl_priors_multi(suspicious_value_by_varcop, mutated_project_dir)
    return suspicious_value_by_varcop