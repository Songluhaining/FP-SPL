"""Compare PULP, CLAP, and the restored causal ranker on all candidates.

The PULP benchmark relabels some originally failing configurations as FP.  The
causal ranker only scores configurations whose observed tests pass, so this
script reports both a strict missing-score treatment and an observable-failure
routing treatment for those synthetic configurations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


FEATURES = (
    "DDU", "code_coverage", "incorrectness_verifiability",
    "correctness_reflectability", "bscp", "bug_involving_statements",
)


def metrics(rows: list[dict]) -> dict:
    tp = sum(row["label"] and row["prediction"] for row in rows)
    fp = sum(not row["label"] and row["prediction"] for row in rows)
    tn = sum(not row["label"] and not row["prediction"] for row in rows)
    fn = sum(row["label"] and not row["prediction"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "configurations": len(rows),
        "false_passing": tp + fn,
        "predicted_positive": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(rows),
        "false_passing_precision": precision,
        "false_passing_recall": recall,
        "false_passing_f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
    }


def fold_of(version: str, count: int = 5) -> int:
    digest = hashlib.sha256(version.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def causal_scores(results: Path, keys: set[tuple[str, str]]) -> dict:
    scores = {key: 0.0 for key in keys}
    for path in results.glob("*/pair-assessments.json"):
        version = path.parent.name
        for pair in json.loads(path.read_text(encoding="utf-8")):
            key = (version, pair["passing_configuration"])
            if key not in scores:
                continue
            candidates = [
                statement["failing_suspiciousness"]
                for statement in pair["statements"]
                if statement["present_in_a"]
                and statement["executed_in_a"]
                and statement["causal_in_a"]
                and statement["failing_suspiciousness"] >= 0.50
            ]
            if candidates:
                scores[key] = max(scores[key], max(candidates))
    return scores


def select_group_budgets(rows: list[dict], scores: dict, budgets: dict) -> list[dict]:
    selected = set()
    for group, budget in budgets.items():
        candidates = [row for row in rows if row["group"] == group]
        candidates.sort(key=lambda row: (
            -scores[(row["version"], row["configuration"])],
            row["version"], row["configuration"],
        ))
        selected.update(
            (row["version"], row["configuration"])
            for row in candidates[:budget]
        )
    return [
        {**row, "prediction": int((row["version"], row["configuration"]) in selected)}
        for row in rows
    ]


def run(args: argparse.Namespace) -> dict:
    import pandas as pd
    from sklearn.svm import SVC

    base_versions = set(json.loads(args.base_versions.read_text(encoding="utf-8")))
    origin = {}
    for version in sorted(path for path in args.dataset.iterdir() if path.is_dir()):
        with (version / "variant_labels.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["LABEL"] in ("FP", "TP"):
                    key = (version.name, row["VARIANT"])
                    origin[key] = {
                        "label": int(row["LABEL"] == "FP"),
                        "converted": row["FP transformed from F"] == "1",
                    }

    pulp_rows = list(csv.DictReader(args.pulp_predictions.open(encoding="utf-8-sig")))
    pulp = {}
    for row in pulp_rows:
        key = (row["version"], row["model"])
        if key not in origin:
            continue
        pulp[key] = int(row["prediction"] == "FP")
    if set(pulp) != set(origin):
        raise ValueError("PULP predictions and full candidate labels do not match")

    rows = [{
        "version": key[0], "configuration": key[1],
        "label": value["label"], "converted": value["converted"],
        "group": "base_feature" if key[0] in base_versions else "other_feature",
    } for key, value in sorted(origin.items())]
    budgets = {
        group: sum(pulp[(row["version"], row["configuration"])]
                   for row in rows if row["group"] == group)
        for group in ("base_feature", "other_feature")
    }

    pulp_eval = metrics([
        {**row, "prediction": pulp[(row["version"], row["configuration"])]}
        for row in rows
    ])

    frames = []
    for version in sorted(path for path in args.dataset.iterdir() if path.is_dir()):
        frame = pd.read_csv(version / "attributes-clap.csv")
        frame = frame[frame["LABEL"].isin(("FP", "TP"))].copy()
        frame["version"] = version.name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    clap_predictions = {}
    clap_scores = {}
    for fold in range(5):
        mask = data["version"].map(lambda name: fold_of(name) == fold)
        train, test = data[~mask], data[mask]
        model = SVC(kernel="linear")
        model.fit(train[list(FEATURES)].to_numpy(),
                  (train["LABEL"] == "FP").astype(int).to_numpy())
        prediction = model.predict(test[list(FEATURES)].to_numpy())
        score = model.decision_function(test[list(FEATURES)].to_numpy())
        for (_, row), value, strength in zip(test.iterrows(), prediction, score):
            key = (row["version"], row["VARIANT"])
            clap_predictions[key] = int(value)
            clap_scores[key] = float(strength)
    clap_binary = metrics([
        {**row, "prediction": clap_predictions[(row["version"], row["configuration"])]}
        for row in rows
    ])
    clap_budget = metrics(select_group_budgets(rows, clap_scores, budgets))

    original_keys = {
        (row["version"], row["configuration"])
        for row in rows if not row["converted"]
    }
    restored = causal_scores(args.causal_results, original_keys)
    missing_scores = {
        (row["version"], row["configuration"]): (
            restored.get((row["version"], row["configuration"]), float("-inf"))
        ) for row in rows
    }
    routed_scores = {
        (row["version"], row["configuration"]): (
            float("inf") if row["converted"]
            else restored[(row["version"], row["configuration"])]
        ) for row in rows
    }
    restored_fixed = metrics([
        {**row, "prediction": int(
            not row["converted"]
            and restored[(row["version"], row["configuration"])] > 0
        )} for row in rows
    ])
    restored_missing_budget = metrics(
        select_group_budgets(rows, missing_scores, budgets)
    )
    restored_routed_budget = metrics(
        select_group_budgets(rows, routed_scores, budgets)
    )

    report = {
        "scope": {
            "configurations": len(rows),
            "false_passing": sum(row["label"] for row in rows),
            "true_passing": sum(not row["label"] for row in rows),
            "converted_from_failing": sum(row["converted"] for row in rows),
            "pulp_group_budgets": budgets,
            "pulp_total_budget": sum(budgets.values()),
        },
        "pulp_binary": pulp_eval,
        "clap_binary": clap_binary,
        "clap_same_group_budgets_as_pulp": clap_budget,
        "restored_fixed_threshold_converted_unscored": restored_fixed,
        "restored_same_group_budgets_converted_unscored": restored_missing_budget,
        "restored_same_group_budgets_observed_failure_routed_first": restored_routed_budget,
        "validity": {
            "converted_configurations_still_fail_in_raw_dataset": True,
            "restored_ranker_requires_observed_passing_configuration": True,
            "observed_failure_routing_is_not_causal_ranker_detection": True,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pulp-predictions", type=Path, required=True)
    parser.add_argument("--causal-results", type=Path, required=True)
    parser.add_argument("--base-versions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
