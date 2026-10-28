"""Run every buggy version with isolated outputs and resumable parallel workers."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

from fpspl.brfl import BrflRuntime
from fpspl.dynamic import RANKING_FIELDS, run_dynamic_version


def _run_one(version_dir: Path, output_dir: Path, java_home: Path, model_kind: str) -> dict:
    destination = output_dir / version_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    summary_file = destination / "summary.json"
    ranking_file = destination / "ranking.csv"
    if summary_file.is_file() and ranking_file.is_file():
        previous = json.loads(summary_file.read_text(encoding="utf-8"))
        if not previous.get("errors") and previous.get("model_kind") == model_kind:
            return previous
    runtime = BrflRuntime(output_dir / "runtime", java_home)
    with (destination / "run.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        return run_dynamic_version(version_dir, destination, runtime, model_kind=model_kind)


def run_all(dataset: Path, output_dir: Path, java_home: Path, jobs: int, model_kind: str) -> dict:
    dataset, output_dir, java_home = dataset.resolve(), output_dir.resolve(), java_home.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    BrflRuntime(output_dir / "runtime", java_home).build()
    versions = sorted(path for path in dataset.iterdir()
                      if path.is_dir() and (path / "config.report.csv.done").is_file())
    if not versions:
        raise ValueError(f"No buggy versions found in {dataset}")
    summaries: dict[str, dict] = {}
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        pending = {executor.submit(_run_one, version, output_dir, java_home, model_kind): version.name
                   for version in versions}
        for future in as_completed(pending):
            name = pending[future]
            try:
                summary = future.result()
                summaries[name] = summary
                print(f"{len(summaries) + len(failures)}/{len(versions)} {name}: "
                      f"ranked={summary['ranked_variants']} errors={len(summary['errors'])}", flush=True)
            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
                print(f"{len(summaries) + len(failures)}/{len(versions)} {name}: {failures[name]}", flush=True)
            (output_dir / "progress.json").write_text(json.dumps({
                "total_versions": len(versions), "completed": len(summaries),
                "failed": failures, "summaries": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for name in sorted(summaries):
        with (output_dir / name / "ranking.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    with (output_dir / "ranking.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    result = {"dataset": str(dataset), "model_kind": model_kind, "total_versions": len(versions),
              "completed_versions": len(summaries), "failed_versions": failures,
              "ranked_variants": sum(summary["ranked_variants"] for summary in summaries.values()),
              "trace_errors": sum(len(summary["errors"]) for summary in summaries.values())}
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--java-home", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--model", choices=("graph", "linear"), default="graph")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    print(json.dumps(run_all(args.dataset, args.output, args.java_home, args.jobs, args.model),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
