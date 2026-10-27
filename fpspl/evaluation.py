"""Post-hoc evaluation; this module is never imported by the ranker."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def average_precision(ranked_labels: list[int]) -> float | None:
    positives = sum(ranked_labels)
    if positives == 0:
        return None
    seen = 0
    precision_sum = 0.0
    for index, label in enumerate(ranked_labels, 1):
        if label:
            seen += 1
            precision_sum += seen / index
    return precision_sum / positives


def evaluate(ranking_file: Path, labels_root: Path, top_k: int = 5) -> dict:
    by_version: dict[str, list[dict]] = {}
    with ranking_file.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "ranked":
                by_version.setdefault(row["version"], []).append(row)

    version_results = []
    for version, rows in sorted(by_version.items()):
        label_path = labels_root / version / "variant_labels.csv"
        if not label_path.is_file():
            continue
        with label_path.open(encoding="utf-8-sig", newline="") as handle:
            labels = {row["VARIANT"]: row["LABEL"] for row in csv.DictReader(handle)}
        # The ranker only processes originally passing configurations. Thus
        # converted-failing FP rows cannot enter this evaluation by accident.
        comparable = [row for row in rows if labels.get(row["variant"]) in {"TP", "FP"}]
        comparable.sort(key=lambda row: (-float(row["score"]), row["variant"]))
        y = [int(labels[row["variant"]] == "FP") for row in comparable]
        if not y:
            continue
        k = min(top_k, len(y))
        version_results.append({
            "version": version,
            "ranked": len(y),
            "false_passing": sum(y),
            "average_precision": average_precision(y),
            "top_k_precision": sum(y[:k]) / k,
            "top_k_recall": sum(y[:k]) / sum(y) if sum(y) else None,
        })
    defined_ap = [row["average_precision"] for row in version_results if row["average_precision"] is not None]
    return {
        "evaluated_versions": len(version_results),
        "evaluated_originally_passing_variants": sum(row["ranked"] for row in version_results),
        "false_passing_variants": sum(row["false_passing"] for row in version_results),
        "macro_average_precision": sum(defined_ap) / len(defined_ap) if defined_ap else None,
        "macro_top_k_precision": sum(row["top_k_precision"] for row in version_results) / len(version_results) if version_results else None,
        "top_k": top_k,
        "scope": "Originally passing configurations only; labels read after ranking.",
        "versions": version_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ranking against held-out research labels")
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    result = evaluate(args.ranking, args.labels_root, args.top_k)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
