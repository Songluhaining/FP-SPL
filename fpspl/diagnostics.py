"""Post-hoc single-component checks for the research benchmark.

This module reads labels only after graph extraction and is never imported by
the production ranking pipeline.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from fpspl.evaluation import average_precision


def _items(graph: dict, field: str) -> set:
    return {tuple(value) if isinstance(value, list) else value for value in graph.get(field, [])}


def _f1(a: set, b: set) -> float:
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def compare_components(graphs_root: Path, labels_root: Path) -> dict:
    modes = ("features", "nodes", "links", "typed_links", "branches")
    aps = {mode: [] for mode in modes}
    prevalences = []
    for version_dir in sorted(graphs_root.iterdir()):
        if not version_dir.is_dir():
            continue
        label_path = labels_root / version_dir.name / "variant_labels.csv"
        template_path = version_dir / "failure_templates.json"
        graph_path = version_dir / "test_graphs.jsonl.gz"
        if not label_path.is_file() or not template_path.is_file() or not graph_path.is_file():
            continue
        with label_path.open(encoding="utf-8-sig", newline="") as handle:
            labels = {row["VARIANT"]: row["LABEL"] for row in csv.DictReader(handle)}
        templates = json.loads(template_path.read_text(encoding="utf-8"))
        if not templates:
            continue
        by_variant: dict[str, list[dict]] = {}
        with gzip.open(graph_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                graph = json.loads(line)
                if graph["outcome"] == "passed" and labels.get(graph["variant"]) in {"FP", "TP"}:
                    by_variant.setdefault(graph["variant"], []).append(graph)
        if not by_variant:
            continue
        y_all = [int(labels[variant] == "FP") for variant in by_variant]
        if not 0 < sum(y_all) < len(y_all):
            continue
        prevalences.append(sum(y_all) / len(y_all))
        for mode in modes:
            template_sets = [_items(template, mode) for template in templates]
            scored = []
            for variant, graphs in by_variant.items():
                best = max(
                    (_f1(template_set, _items(graph, mode)) for template_set in template_sets for graph in graphs),
                    default=0.0,
                )
                scored.append((best, variant, int(labels[variant] == "FP")))
            scored.sort(key=lambda row: (-row[0], row[1]))
            aps[mode].append(average_precision([row[2] for row in scored]))
    return {
        "mixed_versions": len(prevalences),
        "mean_positive_fraction": sum(prevalences) / len(prevalences) if prevalences else None,
        "macro_average_precision": {
            mode: sum(values) / len(values) if values else None for mode, values in aps.items()
        },
        "method": "Unweighted best-match F1 for one graph component at a time; labels used only afterward.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether graph components beat simple feature matching")
    parser.add_argument("--graphs-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare_components(args.graphs_root, args.labels_root)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
