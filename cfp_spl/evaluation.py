from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


LEVEL_SCORE = {
    "insufficient": 0.0,
    "weak_suspect": 1.0,
    "strong_suspect": 2.0,
    "confirmed_false_passing": 3.0,
}


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if not positives:
        return 0.0
    groups = {}
    for label, score in zip(labels, scores):
        counts = groups.setdefault(score, [0, 0])
        counts[0] += label
        counts[1] += 1 - label
    tp = fp = 0
    previous_recall = 0.0
    result = 0.0
    for score in sorted(groups, reverse=True):
        tp += groups[score][0]
        fp += groups[score][1]
        recall = tp / positives
        result += (recall - previous_recall) * _safe_div(tp, tp + fp)
        previous_recall = recall
    return result


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    positive_scores = [s for y, s in zip(labels, scores) if y]
    negative_scores = [s for y, s in zip(labels, scores) if not y]
    if not positive_scores or not negative_scores:
        return 0.0
    wins = sum((p > n) + 0.5 * (p == n) for p in positive_scores for n in negative_scores)
    return wins / (len(positive_scores) * len(negative_scores))


def _metrics(labels: list[int], predicted: list[int], scores: list[float]) -> dict:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "confusion": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "accuracy": _safe_div(tp + tn, len(labels)),
        "false_passing_precision": precision,
        "false_passing_recall": recall,
        "false_passing_f1": _safe_div(2 * precision * recall, precision + recall),
        "specificity": _safe_div(tn, tn + fp),
        "average_precision": _average_precision(labels, scores),
        "roc_auc": _roc_auc(labels, scores),
    }


def evaluate(results_root: Path, labels_root: Path) -> dict:
    records = []
    per_version = {}
    transformed_total = 0
    for label_file in sorted(labels_root.glob("*/variant_labels.csv")):
        version = label_file.parent.name
        labels = list(csv.DictReader(label_file.open(encoding="utf-8-sig")))
        transformed_total += sum(row.get("FP transformed from F", "") == "1" for row in labels)
        label_by_name = {
            row["VARIANT"]: row["LABEL"] for row in labels
            if row.get("FP transformed from F", "") != "1" and row["LABEL"] in {"FP", "TP"}
        }
        ranking = results_root / version / "ranking.csv"
        if not ranking.is_file():
            per_version[version] = {"status": "missing_result", "expected_originally_passing": len(label_by_name)}
            continue
        rows = list(csv.DictReader(ranking.open(encoding="utf-8-sig")))
        version_records = []
        for row in rows:
            if row["configuration"] not in label_by_name:
                continue
            level = row["evidence_level"]
            score = LEVEL_SCORE[level] + float(row["score"])
            item = {
                "version": version,
                "configuration": row["configuration"],
                "label": label_by_name[row["configuration"]],
                "level": level,
                "score": score,
            }
            records.append(item)
            version_records.append(item)
        expected = set(label_by_name)
        observed = {row["configuration"] for row in version_records}
        per_version[version] = {
            "status": "complete" if expected == observed else "incomplete",
            "expected": len(expected), "evaluated": len(observed),
            "missing": sorted(expected - observed),
            "evidence_levels": dict(Counter(row["level"] for row in version_records)),
        }
    labels = [1 if row["label"] == "FP" else 0 for row in records]
    scores = [row["score"] for row in records]
    any_evidence = [row["level"] != "insufficient" for row in records]
    strong_evidence = [row["level"] in {"confirmed_false_passing", "strong_suspect"} for row in records]
    report = {
        "scope": {
            "evaluated_originally_passing": len(records),
            "ground_truth_false_passing": sum(labels),
            "ground_truth_true_passing": len(labels) - sum(labels),
            "transformed_from_failing_excluded": transformed_total,
            "versions_complete": sum(v.get("status") == "complete" for v in per_version.values()),
            "versions_total": len(per_version),
        },
        "coverage": {
            "configurations_with_any_positive_evidence": sum(any_evidence),
            "rate": _safe_div(sum(any_evidence), len(records)),
            "levels": dict(Counter(row["level"] for row in records)),
        },
        "any_evidence_as_positive": _metrics(labels, [int(x) for x in any_evidence], scores),
        "strong_or_confirmed_as_positive": _metrics(labels, [int(x) for x in strong_evidence], scores),
        "per_version": per_version,
    }
    (results_root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评价反事实因果方法对原本通过配置的识别效果")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.results, args.labels), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
