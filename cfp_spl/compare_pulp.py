from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .evaluation import _metrics


def compare(results: Path, pulp_predictions: Path) -> dict:
    rows = list(csv.DictReader(pulp_predictions.open(encoding="utf-8-sig")))
    truth = {(r["version"], r["variant"]): int(r["label"] == "FP") for r in rows}
    group = {(r["version"], r["variant"]): r["group"] for r in rows}
    pulp_pred = {(r["version"], r["variant"]): int(r["pulp_prediction"] == "FP") for r in rows}
    new_score = {key: 0.0 for key in truth}
    for path in results.glob("*/pair-assessments.json"):
        version = path.parent.name
        for pair in json.loads(path.read_text(encoding="utf-8")):
            key = (version, pair["passing_configuration"])
            if key not in new_score:
                continue
            candidates = [stmt.get("joint_suspiciousness", 0.0) for stmt in pair["statements"]]
            if candidates:
                new_score[key] = max(new_score[key], max(candidates))

    keys = sorted(truth)
    budgets = defaultdict(int)
    for key, predicted in pulp_pred.items():
        budgets[key[0]] += predicted
    new_same_per_version_budget = {key: 0 for key in keys}
    by_version = defaultdict(list)
    for key in keys:
        by_version[key[0]].append(key)
    for version, version_keys in by_version.items():
        ranked = sorted(version_keys, key=lambda key: (-new_score[key], key[1]))
        for key in ranked[:budgets[version]]:
            new_same_per_version_budget[key] = 1

    total_budget = sum(pulp_pred.values())
    globally_ranked = sorted(keys, key=lambda key: (-new_score[key], key[0], key[1]))
    global_selected = set(globally_ranked[:total_budget])
    new_same_global_budget = {key: int(key in global_selected) for key in keys}
    new_same_group_budget = {key: 0 for key in keys}
    for group_name in {group[key] for key in keys}:
        group_keys = [key for key in keys if group[key] == group_name]
        group_budget = sum(pulp_pred[key] for key in group_keys)
        ranked = sorted(group_keys, key=lambda key: (-new_score[key], key[0], key[1]))
        for key in ranked[:group_budget]:
            new_same_group_budget[key] = 1
    new_fixed = {key: int(new_score[key] > 0) for key in keys}

    report = {"scope": {"configurations": len(keys), "false_passing": sum(truth.values()),
                         "pulp_predicted_positive_budget": total_budget}}
    methods = {
        "pulp_binary": pulp_pred,
        "new_fixed_neutral_threshold": new_fixed,
        "new_same_per_version_budget_as_pulp": new_same_per_version_budget,
        "new_same_feature_group_budget_as_pulp": new_same_group_budget,
        "new_same_global_budget_as_pulp": new_same_global_budget,
    }
    for method, predictions in methods.items():
        report[method] = {}
        for group_name, selector in (
            ("all", lambda key: True),
            ("base_feature", lambda key: group[key] == "base_feature"),
            ("optional_feature", lambda key: group[key] == "other_feature"),
        ):
            selected = [key for key in keys if selector(key)]
            y = [truth[key] for key in selected]
            pred = [predictions[key] for key in selected]
            scores = [new_score[key] if method.startswith("new") else float(pulp_pred[key]) for key in selected]
            report[method][group_name] = {
                "configurations": len(selected), "ground_truth_false_passing": sum(y),
                "predicted_positive": sum(pred), **_metrics(y, pred, scores),
            }
    (results / "comparison-with-pulp.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--pulp-predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.results, args.pulp_predictions), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
