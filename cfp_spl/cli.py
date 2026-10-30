"""解析运行参数并启动识别流程；算法本身在 pipeline.py 中。"""

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
    value.add_argument("--max-passing", type=int, default=0, help="过程验证时每版本只处理前若干通过配置；0表示全部")
    value.add_argument("--neighbors", type=int, default=1, help="至少选取的近邻数，边界距离并列项全部保留；默认选择所有并列最近项")
    value.add_argument("--similarity-threshold", type=float, default=0.55, help="已有测试的最低内容相似度")
    value.add_argument("--top-k", type=int, default=20, help="每条失败测试检查的高可疑语句数")
    value.add_argument("--generator-command", help="生成测试的外部命令；可用{request}和{response}占位符")
    value.add_argument("--deepseek-api-key-file", type=Path,
                       help="DeepSeek接口密钥文件；默认读取环境变量或仓库同级deepseek-api-key.txt")
    value.add_argument("--deepseek-model", default="deepseek-chat",
                       help="用于生成反事实测试的模型名称")
    value.add_argument("--deepseek-base-url", default="https://api.deepseek.com",
                       help="DeepSeek接口根地址")
    value.add_argument("--plan-only", action="store_true", help="只生成配置与测试配对计划，不插桩执行")
    return value


def main(argv: list[str] | None = None) -> int:
    """可在调试器中从这里开始，argv=None 时读取终端参数。"""
    args = parser().parse_args(argv)
    CounterfactualPipeline(
        args.dataset, args.output, args.java_home,
        generator_command=args.generator_command, neighbors=args.neighbors,
        similarity_threshold=args.similarity_threshold, top_k=args.top_k,
        deepseek_key_file=args.deepseek_api_key_file,
        deepseek_model=args.deepseek_model,
        deepseek_base_url=args.deepseek_base_url,
        max_passing=args.max_passing,
    ).run(args.version, args.plan_only)
    return 0
