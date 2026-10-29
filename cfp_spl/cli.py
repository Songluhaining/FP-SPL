from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import CounterfactualPipeline


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="用反事实测试与BRFL因果推理识别软件产品线假通过配置")
    value.add_argument("--dataset", type=Path, required=True, help="数据集根目录")
    value.add_argument("--output", type=Path, required=True, help="实验输出目录")
    value.add_argument("--java-home", type=Path, required=True, help="Java 8开发工具包目录")
    value.add_argument("--version", help="只处理一个缺陷版本")
    value.add_argument("--neighbors", type=int, default=1, help="每个通过配置使用的近邻未通过配置数")
    value.add_argument("--similarity-threshold", type=float, default=0.55, help="已有测试的最低内容相似度")
    value.add_argument("--top-k", type=int, default=20, help="每条失败测试检查的高可疑语句数")
    value.add_argument("--inference", choices=("tensor", "brfl"), default="tensor",
                       help="概率推理器：自适应区域张量推理或BRFL原循环信念传播")
    value.add_argument("--spectrum-metric",
                       choices=("ensemble", "ochiai", "tarantula", "barinel", "jaccard"),
                       default="ensemble", help="语句先验使用的频谱指标；默认使用多指标中位数")
    value.add_argument("--max-region-variables", type=int, default=14,
                       help="单个精确张量区域允许的最大二值变量数")
    value.add_argument("--generator-command", help="生成测试的外部命令；可用{request}和{response}占位符")
    value.add_argument("--plan-only", action="store_true", help="只生成配置与测试配对计划，不插桩执行")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    CounterfactualPipeline(
        args.dataset, args.output, args.java_home, args.generator_command,
        args.neighbors, args.similarity_threshold, args.top_k,
        args.inference, args.spectrum_metric, args.max_region_variables,
    ).run(args.version, args.plan_only)
    return 0
