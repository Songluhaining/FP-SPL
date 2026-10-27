from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fpspl.pipeline import run_dataset
from fpspl.brfl import BrflRuntime
from fpspl.dynamic import RANKING_FIELDS, run_dynamic_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract failed-test behavior templates and rank passing SPL configurations."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Root containing buggy-version directories")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for graphs, templates and ranking")
    parser.add_argument("--method", choices=("dynamic", "legacy-slice"), default="dynamic")
    parser.add_argument("--brfl-root", type=Path, help="BRFL source tree, required for dynamic mode")
    parser.add_argument("--java-home", type=Path, help="Java 8 JDK home, defaults to JAVA_HOME")
    parser.add_argument("--limit-failed-variants", type=int)
    parser.add_argument("--limit-passing-variants", type=int)
    parser.add_argument("--max-tests-per-variant", type=int)
    parser.add_argument("--slice-dataset", type=Path, help="Optional root holding PULP backward-slice files")
    parser.add_argument("--version", action="append", help="Restrict to one version; may be repeated")
    parser.add_argument(
        "--allow-coverage-only", action="store_true",
        help="Allow missing slices; the result then has no dependency links",
    )
    args = parser.parse_args(argv)
    if args.method == "dynamic":
        if args.brfl_root is None:
            parser.error("--brfl-root is required for dynamic mode")
        runtime = BrflRuntime(args.brfl_root, args.output / "runtime", args.java_home)
        runtime.build()
        version_dirs = sorted(path for path in args.dataset.iterdir()
                              if path.is_dir() and (path / "config.report.csv.done").is_file())
        if args.version:
            version_dirs = [path for path in version_dirs if path.name in set(args.version)]
        if not version_dirs:
            parser.error("No matching dataset versions")
        summaries = []
        all_rows = []
        for version_dir in version_dirs:
            print(f"[dynamic] {version_dir.name}", flush=True)
            summary = run_dynamic_version(
                version_dir, args.output / version_dir.name, runtime,
                max_tests=args.max_tests_per_variant,
                limit_failed_variants=args.limit_failed_variants,
                limit_passing_variants=args.limit_passing_variants,
            )
            summaries.append(summary)
            with (args.output / version_dir.name / "ranking.csv").open(encoding="utf-8-sig", newline="") as handle:
                all_rows.extend(csv.DictReader(handle))
        with (args.output / "ranking.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        (args.output / "summary.json").write_text(
            json.dumps({"dataset": str(args.dataset.resolve()), "versions": len(summaries),
                        "ranked_variants": sum(item["ranked_variants"] for item in summaries),
                        "summaries": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Completed: {len(summaries)} versions, "
              f"{sum(summary['ranked_variants'] for summary in summaries)} ranked variants")
        return 0
    result = run_dataset(
        args.dataset,
        args.output,
        args.slice_dataset,
        set(args.version) if args.version else None,
        args.allow_coverage_only,
    )
    print(f"Completed: {result['versions']} versions, {result['ranked_variants']} ranked variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
