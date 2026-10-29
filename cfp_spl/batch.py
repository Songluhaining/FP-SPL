from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import CounterfactualPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将完整数据集拆成互不重叠的并行实验分片")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--java-home", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--inference", choices=("tensor", "brfl"), default="tensor")
    parser.add_argument("--spectrum-metric",
                        choices=("ensemble", "ochiai", "tarantula", "barinel", "jaccard"),
                        default="ensemble")
    parser.add_argument("--max-region-variables", type=int, default=14)
    args = parser.parse_args(argv)
    if args.shards < 1 or not 0 <= args.index < args.shards:
        raise ValueError("分片编号必须处于[0, 分片总数)范围")
    pipeline = CounterfactualPipeline(
        args.dataset, args.output, args.java_home,
        inference=args.inference, spectrum_metric=args.spectrum_metric,
        max_region_variables=args.max_region_variables,
    )
    pipeline.runtime.build_brfl()
    versions = [
        path for path in sorted(args.dataset.iterdir())
        if (path / "config.report.csv.done").is_file()
    ]
    summaries = {}
    for position, version in enumerate(versions):
        if position % args.shards != args.index:
            continue
        if (args.output / version.name / "ranking.csv").is_file():
            summaries[version.name] = {"status": "already_complete"}
            continue
        summaries[version.name] = pipeline._run_version(version, plan_only=False)
    shard_dir = args.output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / f"shard-{args.index}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
