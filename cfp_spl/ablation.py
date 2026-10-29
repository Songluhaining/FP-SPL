from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .evaluation import _metrics


RULES = {
    "joint_posterior_0_50": lambda s: s.get("joint_suspiciousness", 0.0) >= 0.50,
    "joint_posterior_0_55": lambda s: s.get("joint_suspiciousness", 0.0) >= 0.55,
    "joint_posterior_0_60": lambda s: s.get("joint_suspiciousness", 0.0) >= 0.60,
    "joint_posterior_0_70": lambda s: s.get("joint_suspiciousness", 0.0) >= 0.70,
}


def run(results: Path, labels_root: Path) -> dict:
    labels = {}
    for path in labels_root.glob("*/variant_labels.csv"):
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            if row.get("FP transformed from F", "") != "1" and row["LABEL"] in {"FP", "TP"}:
                labels[(path.parent.name, row["VARIANT"])] = int(row["LABEL"] == "FP")
    evidence = {key: {rule: 0.0 for rule in RULES} for key in labels}
    for path in results.glob("*/pair-assessments.json"):
        version = path.parent.name
        for pair in json.loads(path.read_text(encoding="utf-8")):
            key = (version, pair["passing_configuration"])
            if key not in evidence:
                continue
            for rule, predicate in RULES.items():
                matches = [s for s in pair["statements"] if predicate(s)]
                if matches:
                    score = max(s["joint_suspiciousness"] for s in matches)
                    evidence[key][rule] = max(evidence[key][rule], score)
    ordered = sorted(labels)
    base_file = labels_root.parent / "base_case_labeling.json"
    base_versions = set(json.loads(base_file.read_text(encoding="utf-8"))) if base_file.is_file() else set()
    y = [labels[key] for key in ordered]
    report = {}
    for rule in RULES:
        scores = [evidence[key][rule] for key in ordered]
        predicted = [int(score > 0) for score in scores]
        by_group = {}
        for group, choose in (
            ("base_feature_versions", lambda key: key[0] in base_versions),
            ("optional_feature_versions", lambda key: key[0] not in base_versions),
        ):
            indices = [i for i, key in enumerate(ordered) if choose(key)]
            group_y = [y[i] for i in indices]
            group_predicted = [predicted[i] for i in indices]
            group_scores = [scores[i] for i in indices]
            group_ranking = sorted(zip(group_scores, group_y), reverse=True)
            budget = sum(group_y)
            budget_true_positive = sum(label for _, label in group_ranking[:budget]) if budget else 0
            by_group[group] = {
                "configurations": len(indices),
                "ground_truth_false_passing": sum(group_y),
                "predicted_positive": sum(group_predicted),
                "top_ground_truth_positive_count": {
                    "k": budget,
                    "true_positive": budget_true_positive,
                    "precision_and_recall_at_equal_budget": budget_true_positive / budget if budget else 0.0,
                },
                **_metrics(group_y, group_predicted, group_scores),
            }
        report[rule] = {"predicted_positive": sum(predicted), **_metrics(y, predicted, scores), "by_group": by_group}
    (results / "ablation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.results, args.labels), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
