from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def diagnose(results: Path, labels_root: Path) -> dict:
    plans = assessments = errors = 0
    provenance_plan = Counter()
    provenance_executed = Counter()
    error_messages = Counter()
    statement_total = present = executed = causal = 0
    high_failing = large_gap = qualifying = 0
    pair_with_graph = pair_with_aligned_causal = pair_with_qualifying = 0
    aligned_gaps = []
    aligned_failing = []
    for version in sorted(path for path in results.iterdir() if path.is_dir() and path.name.startswith("_")):
        plan_path = version / "pair-plan.json"
        assessment_path = version / "pair-assessments.json"
        error_path = version / "errors.json"
        if plan_path.is_file():
            rows = json.loads(plan_path.read_text(encoding="utf-8"))
            plans += len(rows)
            provenance_plan.update(row["test_provenance"] for row in rows)
        if error_path.is_file():
            rows = json.loads(error_path.read_text(encoding="utf-8"))
            errors += len(rows)
            error_messages.update(row.get("error", "") for row in rows)
        if not assessment_path.is_file():
            continue
        rows = json.loads(assessment_path.read_text(encoding="utf-8"))
        assessments += len(rows)
        provenance_executed.update(row["test_provenance"] for row in rows)
        for row in rows:
            statements = row["statements"]
            if statements:
                pair_with_graph += 1
            has_aligned = has_qualifying = False
            for stmt in statements:
                statement_total += 1
                present += bool(stmt["present_in_a"])
                executed += bool(stmt["executed_in_a"])
                causal += bool(stmt["causal_in_a"])
                high_failing += stmt["failing_suspiciousness"] >= 0.55
                large_gap += stmt["posterior_gap"] >= 0.15
                aligned = stmt["present_in_a"] and stmt["executed_in_a"] and stmt["causal_in_a"]
                accepted = aligned and stmt["failing_suspiciousness"] >= 0.55 and stmt["posterior_gap"] >= 0.15
                qualifying += accepted
                if aligned:
                    has_aligned = True
                    aligned_gaps.append(stmt["posterior_gap"])
                    aligned_failing.append(stmt["failing_suspiciousness"])
                has_qualifying = has_qualifying or accepted
            pair_with_aligned_causal += has_aligned
            pair_with_qualifying += has_qualifying

    labels = {}
    for path in labels_root.glob("*/variant_labels.csv"):
        version = path.parent.name
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            if row.get("FP transformed from F", "") != "1" and row["LABEL"] in {"FP", "TP"}:
                labels[(version, row["VARIANT"])] = 1 if row["LABEL"] == "FP" else 0
    ranked = []
    for path in results.glob("*/ranking.csv"):
        version = path.parent.name
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            key = (version, row["configuration"])
            if key in labels:
                ranked.append((float(row["score"]), labels[key], version, row["configuration"]))
    ranked.sort(reverse=True)
    positive_scores = [score for score, label, _, _ in ranked if label]
    negative_scores = [score for score, label, _, _ in ranked if not label]
    thresholds = sorted({score for score, _, _, _ in ranked}, reverse=True)
    best = {"f1": -1.0}
    for threshold in thresholds:
        tp = sum(label and score >= threshold for score, label, _, _ in ranked)
        fp = sum(not label and score >= threshold for score, label, _, _ in ranked)
        fn = sum(label and score < threshold for score, label, _, _ in ranked)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {"threshold": threshold, "precision": precision, "recall": recall,
                    "f1": f1, "predicted_positive": tp + fp, "true_positive": tp, "false_positive": fp}
    precision_at = {}
    for name, k in (("top_10_percent", max(1, len(ranked) // 10)),
                    ("top_20_percent", max(1, len(ranked) // 5)),
                    ("top_ground_truth_positive_count", sum(label for _, label, _, _ in ranked))):
        precision_at[name] = {
            "k": k, "true_positive": sum(label for _, label, _, _ in ranked[:k]),
            "precision": sum(label for _, label, _, _ in ranked[:k]) / k,
        }
    report = {
        "execution": {
            "planned_pairs": plans, "assessed_pairs": assessments, "errors": errors,
            "plan_provenance": dict(provenance_plan),
            "assessed_provenance": dict(provenance_executed),
            "top_error_messages": error_messages.most_common(10),
        },
        "causal_filters": {
            "pairs_with_graph": pair_with_graph,
            "pairs_with_aligned_causal_statement": pair_with_aligned_causal,
            "pairs_with_qualifying_statement": pair_with_qualifying,
            "statements_examined": statement_total,
            "present_in_a": present, "executed_in_a": executed, "causal_in_a": causal,
            "failing_suspiciousness_at_least_0_55": high_failing,
            "posterior_gap_at_least_0_15": large_gap,
            "qualifying_all_conditions": qualifying,
            "aligned_gap_mean": statistics.fmean(aligned_gaps) if aligned_gaps else None,
            "aligned_gap_median": statistics.median(aligned_gaps) if aligned_gaps else None,
            "aligned_failing_suspiciousness_mean": statistics.fmean(aligned_failing) if aligned_failing else None,
        },
        "ranking_diagnostics": {
            "false_passing_score_mean": statistics.fmean(positive_scores),
            "true_passing_score_mean": statistics.fmean(negative_scores),
            "false_passing_score_median": statistics.median(positive_scores),
            "true_passing_score_median": statistics.median(negative_scores),
            "precision_at": precision_at,
            "oracle_tuned_best_f1_not_valid_for_final_evaluation": best,
        },
    }
    (results / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(diagnose(args.results, args.labels), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
