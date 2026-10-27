from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from fpspl.coverage import FeatureLocationMapper, read_test_graph
from fpspl.model import FailureTemplate, TestGraph
from fpspl.ranking import ContrastWeights, rank_variants
from fpspl.slicing import SLICE_NAME, read_backward_slices


RANKING_FIELDS = (
    "version", "variant", "rank", "score", "status", "best_failure_variant",
    "best_failure_test", "best_passing_test", "matched_nodes", "matched_links",
    "node_similarity", "link_similarity", "branch_similarity",
    "feature_similarity", "semantic_similarity", "observation_similarity", "evidence",
)


def read_observed_outcomes(version_dir: Path) -> dict[str, str]:
    path = version_dir / "config.report.csv.done"
    if not path.is_file():
        path = version_dir / "config.report.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Configuration report missing: {version_dir}")
    outcomes: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "__TEST_OUTPUT__" not in reader.fieldnames:
            raise ValueError(f"Missing __TEST_OUTPUT__ in {path}")
        id_field = reader.fieldnames[0]
        for row in reader:
            variant = row[id_field].strip()
            result = row["__TEST_OUTPUT__"].strip()
            if result == "__FAILED__":
                outcomes[variant] = "failed"
            elif result == "__PASSED__":
                outcomes[variant] = "passed"
    return outcomes


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ranking(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _coverage_files(variant_dir: Path, outcome: str) -> list[Path]:
    return sorted((variant_dir / "coverage" / outcome).glob("*.coverage.xml"))


def process_version(
    version_dir: Path,
    output_dir: Path,
    slice_version_dir: Path | None = None,
    allow_coverage_only: bool = False,
) -> tuple[dict, list[dict]]:
    outcomes = read_observed_outcomes(version_dir)
    failures: list[TestGraph] = []
    passing_by_variant: dict[str, list[TestGraph]] = {}
    all_graphs: list[TestGraph] = []
    missing_variants: list[str] = []
    for variant, outcome in sorted(outcomes.items()):
        variant_dir = version_dir / "variants" / variant
        if not variant_dir.is_dir():
            missing_variants.append(variant)
            continue
        slice_dir = (slice_version_dir or version_dir) / "variants" / variant
        if not allow_coverage_only and not (slice_dir / SLICE_NAME).is_file():
            raise FileNotFoundError(
                f"No backward slices for {version_dir.name}/{variant}. "
                "Pass --slice-dataset with the prepared dataset, or explicitly "
                "use --allow-coverage-only for a weaker coverage-only run."
            )
        slices = read_backward_slices(slice_dir)
        mapper = FeatureLocationMapper(variant_dir)
        graphs = [
            read_test_graph(version_dir.name, variant_dir, path, outcome, mapper, slices)
            for path in _coverage_files(variant_dir, outcome)
        ]
        all_graphs.extend(graphs)
        if outcome == "failed":
            failures.extend(graphs)
        else:
            passing_by_variant[variant] = graphs

    templates = [
        FailureTemplate.from_graph(graph)
        for graph in failures
        if len(graph.nodes) >= 3 and (graph.links or allow_coverage_only)
    ]
    unlabeled = [graph for graphs in passing_by_variant.values() for graph in graphs if graph.nodes]
    weights = ContrastWeights.fit(failures, unlabeled)
    ranked = rank_variants(templates, passing_by_variant, weights)
    rows = []
    for row in ranked:
        components = row["components"]
        rows.append({
            "version": version_dir.name,
            "variant": row["variant"],
            "rank": row["rank"],
            "score": row["score"],
            "status": row["status"],
            "best_failure_variant": row["best_failure_variant"],
            "best_failure_test": row["best_failure_test"],
            "best_passing_test": row["best_passing_test"],
            "matched_nodes": row["matched_nodes"],
            "matched_links": row["matched_links"],
            "node_similarity": components.get("nodes"),
            "link_similarity": components.get("links"),
            "branch_similarity": components.get("branches"),
            "feature_similarity": components.get("features"),
            "semantic_similarity": components.get("semantics"),
            "observation_similarity": components.get("observations"),
            "evidence": row["evidence"],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "failure_templates.json", [template.to_dict() for template in templates])
    with gzip.open(output_dir / "test_graphs.jsonl.gz", "wt", encoding="utf-8") as handle:
        for graph in all_graphs:
            handle.write(json.dumps(graph.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    _write_ranking(output_dir / "ranking.csv", rows)
    summary = {
        "version": version_dir.name,
        "observed_outcomes": dict(Counter(outcomes.values())),
        "failure_tests": len(failures),
        "passing_tests": sum(len(graphs) for graphs in passing_by_variant.values()),
        "templates": len(templates),
        "ranked_variants": sum(row["status"] == "ranked" for row in rows),
        "insufficient_evidence_variants": sum(row["status"] != "ranked" for row in rows),
        "missing_variant_dirs": missing_variants,
        "graph_evidence": dict(Counter(graph.evidence for graph in all_graphs)),
        "mean_failure_links": round(sum(len(g.links) for g in failures) / len(failures), 2) if failures else 0,
        "mean_passing_links": round(sum(len(g.links) for g in unlabeled) / len(unlabeled), 2) if unlabeled else 0,
        "label_policy": "Only __TEST_OUTPUT__ and per-test results used; FP/TP ground truth is not read.",
        "score_meaning": "Similarity/enrichment rank, not a calibrated false-passing probability.",
    }
    _write_json(output_dir / "summary.json", summary)
    return summary, rows


def run_dataset(
    dataset_dir: Path,
    output_dir: Path,
    slice_dataset_dir: Path | None = None,
    versions: set[str] | None = None,
    allow_coverage_only: bool = False,
) -> dict:
    if not dataset_dir.is_dir():
        raise NotADirectoryError(dataset_dir)
    version_dirs = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_dir() and (path / "config.report.csv.done").is_file()
    )
    if versions is not None:
        version_dirs = [path for path in version_dirs if path.name in versions]
        unknown = versions - {path.name for path in version_dirs}
        if unknown:
            raise ValueError(f"Unknown versions: {', '.join(sorted(unknown))}")
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for index, version_dir in enumerate(version_dirs, 1):
        print(f"[{index}/{len(version_dirs)}] {version_dir.name}", flush=True)
        slice_version_dir = slice_dataset_dir / version_dir.name if slice_dataset_dir else None
        summary, rows = process_version(
            version_dir,
            output_dir / version_dir.name,
            slice_version_dir,
            allow_coverage_only,
        )
        summaries.append(summary)
        all_rows.extend(rows)
        print(f"  templates={summary['templates']} ranked={summary['ranked_variants']}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_ranking(output_dir / "ranking.csv", all_rows)
    result = {
        "dataset": str(dataset_dir.resolve()),
        "slice_dataset": str(slice_dataset_dir.resolve()) if slice_dataset_dir else str(dataset_dir.resolve()),
        "versions": len(version_dirs),
        "ranked_variants": sum(item["ranked_variants"] for item in summaries),
        "failure_tests": sum(item["failure_tests"] for item in summaries),
        "templates": sum(item["templates"] for item in summaries),
        "summaries": summaries,
    }
    _write_json(output_dir / "summary.json", result)
    return result
