"""Compare graph ranking with PULP on identical originally passing variants."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             accuracy_score, roc_auc_score, roc_curve)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _original_passes(dataset: Path) -> set[tuple[str, str]]:
    found = set()
    for version in dataset.iterdir():
        report = version / "config.report.csv.done"
        if not report.is_file():
            continue
        rows = _read(report)
        if not rows:
            continue
        name_column = next(iter(rows[0]))
        found.update((version.name, row[name_column]) for row in rows
                     if "PASS" in row["__TEST_OUTPUT__"])
    return found


def _ranking_metrics(rows: list[dict], field: str) -> dict:
    if not rows:
        return {"average_precision": None, "precision_at_5": None, "recall_at_5": None}
    ordered = sorted(rows, key=lambda row: (-float(row[field]), row["variant"]))
    labels = [int(row["label"] == "FP") for row in ordered]
    top = labels[:min(5, len(labels))]
    return {"average_precision": float(average_precision_score(labels, [float(row[field]) for row in ordered]))
            if any(labels) else None,
            "precision_at_5": sum(top) / len(top),
            "recall_at_5": sum(top) / sum(labels) if any(labels) else None}


def compare(dataset: Path, pulp_predictions: Path, graph_ranking: Path,
            base_labels: Path, output: Path) -> dict:
    original = _original_passes(dataset)
    pulp = {(row["version"], row["model"]): row for row in _read(pulp_predictions)}
    graph = {(row["version"], row["variant"]): row for row in _read(graph_ranking)
             if row["status"] in {"ranked", "no_trained_model"}}
    matched = sorted(original & pulp.keys() & graph.keys())
    base_versions = set(json.loads(base_labels.read_text(encoding="utf-8")))
    combined = []
    for version, variant in matched:
        baseline, proposed = pulp[version, variant], graph[version, variant]
        combined.append({"version": version, "variant": variant, "label": baseline["LABEL"],
                         "group": "base_feature" if version in base_versions else "other_feature",
                         "pulp_score": float(baseline["score"]),
                         "pulp_prediction": baseline["prediction"],
                         "graph_score": float(proposed["score"]),
                         "graph_status": proposed["status"]})
    by_version = defaultdict(list)
    for row in combined:
        by_version[row["version"]].append(row)
    per_version = []
    for version, rows in sorted(by_version.items()):
        pulp_metrics = _ranking_metrics(rows, "pulp_score")
        graph_metrics = _ranking_metrics(rows, "graph_score")
        per_version.append({"version": version,
                            "group": "base_feature" if version in base_versions else "other_feature",
                            "variants": len(rows), "FP": sum(row["label"] == "FP" for row in rows),
                            "TP": sum(row["label"] == "TP" for row in rows),
                            "graph_model_trained": all(row["graph_status"] == "ranked" for row in rows),
                            **{"pulp_" + key: value for key, value in pulp_metrics.items()},
                            **{"graph_" + key: value for key, value in graph_metrics.items()}})

    def group_summary(group: str) -> dict:
        entries = [row for row in per_version if row["group"] == group]
        subjects = [row for row in combined if row["group"] == group]
        mixed = [row for row in entries if row["FP"] > 0 and row["TP"] > 0]
        def mean(field: str, items: list[dict]):
            values = [row[field] for row in items if row[field] is not None]
            return sum(values) / len(values) if values else None
        differences = [row["graph_average_precision"] - row["pulp_average_precision"]
                       for row in mixed]
        if differences:
            rng = random.Random(17)
            draws = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                           for _ in range(5000))
            interval = [draws[124], draws[4874]]
        else:
            interval = None
        true_positive = sum(row["label"] == "FP" and row["pulp_prediction"] == "FP"
                            for row in subjects)
        false_positive = sum(row["label"] == "TP" and row["pulp_prediction"] == "FP"
                             for row in subjects)
        false_negative = sum(row["label"] == "FP" and row["pulp_prediction"] == "TP"
                             for row in subjects)
        return {"versions": len(entries), "mixed_label_versions": len(mixed),
                "all_FP_versions": sum(row["FP"] == row["variants"] for row in entries),
                "all_TP_versions": sum(row["TP"] == row["variants"] for row in entries),
                "graph_model_trained_versions": sum(row["graph_model_trained"] for row in entries),
                "variants": sum(row["variants"] for row in entries),
                "FP": sum(row["FP"] for row in entries),
                "TP": sum(row["TP"] for row in entries),
                "pulp_binary_FP_detected": true_positive,
                "pulp_binary_FP_precision": true_positive / (true_positive + false_positive)
                if true_positive + false_positive else None,
                "pulp_binary_FP_recall": true_positive / (true_positive + false_negative)
                if true_positive + false_negative else None,
                "pulp_macro_precision_at_5_all": mean("pulp_precision_at_5", entries),
                "graph_macro_precision_at_5_all": mean("graph_precision_at_5", entries),
                "pulp_macro_AP_mixed": mean("pulp_average_precision", mixed),
                "graph_macro_AP_mixed": mean("graph_average_precision", mixed),
                "graph_minus_pulp_macro_AP_mixed": sum(differences) / len(differences)
                if differences else None,
                "paired_bootstrap_95pct_interval": interval,
                "graph_AP_wins_ties_losses": [sum(value > 1e-12 for value in differences),
                                              sum(abs(value) <= 1e-12 for value in differences),
                                              sum(value < -1e-12 for value in differences)],
                "pulp_macro_precision_at_5_mixed": mean("pulp_precision_at_5", mixed),
                "graph_macro_precision_at_5_mixed": mean("graph_precision_at_5", mixed)}

    true = [int(row["label"] == "FP") for row in combined]
    pulp_binary = [int(row["pulp_prediction"] == "FP") for row in combined]
    missing = sorted((original & pulp.keys()) - graph.keys())
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, pulp_binary, average="binary", zero_division=0) if combined else (None, None, None, None)
    cross_version = [row for row in combined if row["label"] == "TP"
                     or (row["group"] == "base_feature" and row["label"] == "FP")]
    cross_labels = [int(row["label"] == "FP") for row in cross_version]
    cross_metrics = {}
    if set(cross_labels) == {0, 1}:
        for method in ("pulp", "graph"):
            scores = [row[method + "_score"] for row in cross_version]
            fpr, tpr, _ = roc_curve(cross_labels, scores)
            cross_metrics[method] = {"ROC_AUC": float(roc_auc_score(cross_labels, scores)),
                                     "recall_at_FPR_at_most_10pct": float(max(
                                         recall for false_positive_rate, recall in zip(fpr, tpr)
                                         if false_positive_rate <= 0.10))}
    result = {"dataset_versions": len([p for p in dataset.iterdir() if p.is_dir()]),
              "originally_passing_expected": len(original),
              "pulp_originally_passing": len(original & pulp.keys()),
              "graph_originally_passing": len(original & graph.keys()),
              "matched_originally_passing": len(matched),
              "missing_from_graph_count": len(missing),
              "missing_from_graph_examples": [f"{v}/{c}" for v, c in missing[:20]],
              "base_feature": group_summary("base_feature"),
              "other_feature": group_summary("other_feature"),
              "pulp_binary_on_matched": {"accuracy": accuracy_score(true, pulp_binary) if combined else None,
                                          "FP_precision": precision, "FP_recall": recall, "FP_F1": f1},
              "base_FP_vs_other_versions_TP_exploratory": cross_metrics,
              "comparison_scope": "Originally passing variants matched by version and variant; base-feature versions have no TP under PULP labeling, so ranking AP is not discriminative there."}
    output.mkdir(parents=True, exist_ok=True)
    if missing:
        with (output / "missing_from_graph.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["version", "variant"])
            writer.writerows(missing)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows in (("per_version.csv", per_version), ("matched_predictions.csv", combined)):
        if rows:
            with (output / name).open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pulp", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--base-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.dataset, args.pulp, args.graph, args.base_labels, args.output),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
