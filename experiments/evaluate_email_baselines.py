"""Evaluate PULP and CLAP on originally passing Email configurations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


CLAP_FEATURES = (
    "DDU", "code_coverage", "incorrectness_verifiability",
    "correctness_reflectability", "bscp", "bug_involving_statements",
)


def fold_of(version: str) -> int:
    return int.from_bytes(hashlib.sha256(version.encode()).digest()[:8], "big") % 5


def metrics(rows: list[dict], field: str) -> dict:
    labels = [row["label"] for row in rows]
    predictions = [row[field] for row in rows]
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "configurations": len(rows), "false_passing": sum(labels),
        "predicted_positive": sum(predictions), "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn), "accuracy": float((tp + tn) / len(rows)),
        "precision": float(precision), "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0),
    }


def original_passed(dataset: Path) -> set[tuple[str, str]]:
    result = set()
    for version in dataset.iterdir():
        if not version.is_dir():
            continue
        with (version / "config.report.csv.done").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            result.update(
                (version.name, row[reader.fieldnames[0]])
                for row in reader if "PASS" in row["__TEST_OUTPUT__"]
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pulp-repository", type=Path, required=True)
    parser.add_argument("--labeling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.pulp_repository.resolve()))
    from fp_detection.PUBaggingLearning import PUBaggingLearning

    labeling = json.loads(args.labeling.read_text(encoding="utf-8"))
    base_versions = {name for name, value in labeling.items() if value["base_feature"]}
    original = original_passed(args.dataset)

    frames = []
    for version in sorted(path for path in args.dataset.iterdir() if path.is_dir()):
        frame = pd.read_csv(version / "attributes.csv")
        frame.insert(0, "version", version.name)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    failing_count = int((data["LABEL"] == "F").sum())
    unlabeled_count = int((data["LABEL"] != "F").sum())
    ratio = min(1.0, failing_count / unlabeled_count)
    pulp = PUBaggingLearning(sampling_ratio=ratio)
    pulp.prepare_data(data.iloc[:, 3:].to_numpy(), data["LABEL"].to_numpy(), None)
    pulp_prediction = pulp.predict()
    pulp_score = sum(
        weight * estimator.predict_proba(pulp.passing_feature)[:, 1]
        for estimator, weight in zip(pulp.estimators, pulp.estimator_weights)
    )
    passing = data[data["LABEL"] != "F"].copy()
    passing["pulp_prediction"] = pulp_prediction
    passing["pulp_score"] = pulp_score
    passing = passing[passing.apply(
        lambda row: (row["version"], row["model"]) in original, axis=1
    )]

    clap_frames = []
    for version in sorted(path for path in args.dataset.iterdir() if path.is_dir()):
        frame = pd.read_csv(version / "attributes-clap.csv")
        frame = frame[frame.apply(
            lambda row: (version.name, row["VARIANT"]) in original, axis=1
        )].copy()
        frame["version"] = version.name
        clap_frames.append(frame)
    clap = pd.concat(clap_frames, ignore_index=True)
    clap_prediction, clap_score = {}, {}
    for fold in range(5):
        test_mask = clap["version"].map(lambda name: fold_of(name) == fold)
        train, test = clap[~test_mask], clap[test_mask]
        model = SVC(kernel="linear")
        model.fit(train[list(CLAP_FEATURES)].to_numpy(),
                  (train["LABEL"] == "FP").astype(int).to_numpy())
        predictions = model.predict(test[list(CLAP_FEATURES)].to_numpy())
        scores = model.decision_function(test[list(CLAP_FEATURES)].to_numpy())
        for (_, row), prediction, score in zip(test.iterrows(), predictions, scores):
            key = (row["version"], row["VARIANT"])
            clap_prediction[key] = int(prediction)
            clap_score[key] = float(score)

    rows = []
    for _, row in passing.iterrows():
        key = (row["version"], row["model"])
        rows.append({
            "version": key[0], "configuration": key[1],
            "label": int(row["LABEL"] == "FP"),
            "group": "base_feature" if key[0] in base_versions else "other_feature",
            "pulp_prediction": int(row["pulp_prediction"]),
            "pulp_score": float(row["pulp_score"]),
            "clap_prediction": clap_prediction[key], "clap_score": clap_score[key],
        })
    budgets = {
        group: sum(row["pulp_prediction"] for row in rows if row["group"] == group)
        for group in ("base_feature", "other_feature")
    }
    selected = set()
    for group, budget in budgets.items():
        ranked = sorted(
            (row for row in rows if row["group"] == group),
            key=lambda row: (-row["clap_score"], row["version"], row["configuration"]),
        )
        selected.update((row["version"], row["configuration"])
                        for row in ranked[:budget])
    for row in rows:
        row["clap_budget_prediction"] = int(
            (row["version"], row["configuration"]) in selected
        )

    report = {
        "scope": {
            "versions": len({row["version"] for row in rows}),
            "configurations": len(rows), "false_passing": sum(row["label"] for row in rows),
            "true_passing": sum(not row["label"] for row in rows),
            "base_versions": len(base_versions), "group_budgets": budgets,
        },
        "pulp_training": {
            "failing_positive_samples": failing_count,
            "unlabeled_samples": unlabeled_count,
            "sampling_ratio": ratio,
            "proxy_negative_samples_per_estimator": int(unlabeled_count * ratio),
        },
        "pulp": metrics(rows, "pulp_prediction"),
        "clap_binary": metrics(rows, "clap_prediction"),
        "clap_same_group_budgets_as_pulp": metrics(rows, "clap_budget_prediction"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation-balanced.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(
        args.output / "predictions-balanced.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
