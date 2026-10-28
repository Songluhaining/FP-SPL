"""Retrain the graph model from saved runtime graphs without repeating Java tracing."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from fpspl.dynamic import RANKING_FIELDS
from fpspl.graph_model import train_graph_pu_model
from fpspl.model import TestGraph
from fpspl.pu_model import graph_similarity


def retrain(graphs_path: Path, output_dir: Path, epochs: int = 40) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(graphs_path, "rt", encoding="utf-8") as handle:
        graphs = [TestGraph.from_dict(json.loads(line)) for line in handle]
    positives = [graph for graph in graphs if graph.outcome == "failed"]
    unlabeled = [graph for graph in graphs if graph.outcome == "passed"]
    model = train_graph_pu_model(positives, unlabeled, epochs=epochs)
    if model is None:
        raise ValueError("At least two failed and two passing test graphs are required")
    model.save(output_dir / "graph_model.pt")
    best_by_variant: dict[str, dict] = {}
    for graph, score in zip(unlabeled, model.score_all(unlabeled)):
        template, (similarity, components) = max(
            ((positive, graph_similarity(positive, graph)) for positive in positives),
            key=lambda item: item[1][0])
        row = {"version": graph.version, "variant": graph.variant, "score": round(score, 6),
               "model_score": round(score, 6), "template_similarity": round(similarity, 6),
               "status": "ranked", "best_failure_variant": template.variant,
               "best_failure_test": template.test, "best_passing_test": graph.test,
               "matched_typed_edges": len(template.typed_links & graph.typed_links),
               **{key + "_similarity": round(value, 6) for key, value in components.items()}}
        if graph.variant not in best_by_variant or score > best_by_variant[graph.variant]["score"]:
            best_by_variant[graph.variant] = row
    rows = sorted(best_by_variant.values(), key=lambda row: (-row["score"], row["variant"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    with (output_dir / "ranking.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"graphs_file": str(graphs_path.resolve()), "positive_failed_tests": len(positives),
               "unlabeled_passing_tests": len(unlabeled), "ranked_variants": len(rows),
               "model_kind": "graph", "epochs": epochs, "no_ground_truth_used_for_training": True}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True, help="Saved test_graphs.jsonl.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args(argv)
    print(json.dumps(retrain(args.graphs, args.output, args.epochs), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
