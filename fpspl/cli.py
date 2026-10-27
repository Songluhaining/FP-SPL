from __future__ import annotations

import argparse
from pathlib import Path

from fpspl.pipeline import run_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract failed-test behavior templates and rank passing SPL configurations."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Root containing buggy-version directories")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for graphs, templates and ranking")
    parser.add_argument("--slice-dataset", type=Path, help="Optional root holding PULP backward-slice files")
    parser.add_argument("--version", action="append", help="Restrict to one version; may be repeated")
    parser.add_argument(
        "--allow-coverage-only", action="store_true",
        help="Allow missing slices; the result then has no dependency links",
    )
    args = parser.parse_args(argv)
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
