from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .evaluation import _metrics
from .version_split import fold_of


FEATURES = (
    "DDU", "code_coverage", "incorrectness_verifiability",
    "correctness_reflectability", "bscp", "bug_involving_statements",
)


def _group_metrics(rows: list[dict]) -> dict:
    labels = [int(row["label"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    return {"configurations": len(rows), "false_passing": sum(labels),
            "predicted_positive": sum(predictions),
            **_metrics(labels, predictions, scores)}


def _top_budget(rows: list[dict], budget: int) -> list[dict]:
    selected = {(row["version"], row["configuration"]) for row in sorted(
        rows, key=lambda row: (-row["score"], row["version"], row["configuration"])
    )[:budget]}
    return [{**row, "prediction": int(
        (row["version"], row["configuration"]) in selected
    )} for row in rows]


def run(dataset: Path, reference: Path, output: Path, fold_count: int = 5) -> dict:
    try:
        import pandas as pd
        from sklearn.svm import SVC
    except ImportError as exc:
        raise RuntimeError("运行CLAP基准需要pandas和scikit-learn") from exc

    reference_rows = list(csv.DictReader(reference.open(encoding="utf-8-sig")))
    groups = {(row["version"], row["variant"]): row["group"]
              for row in reference_rows}
    pulp_budget = sum(row["pulp_prediction"] == "FP" for row in reference_rows)
    pulp_group_budget = {
        group: sum(row["pulp_prediction"] == "FP" and row["group"] == group
                   for row in reference_rows)
        for group in ("base_feature", "other_feature")
    }
    frames = []
    for version in sorted(path for path in dataset.iterdir() if path.is_dir()):
        path = version / "attributes-clap.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame["version"] = version.name
        frame = frame[frame["LABEL"].isin(("FP", "TP"))]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    reference_keys = set(groups)
    data = data[data.apply(
        lambda row: (row["version"], row["VARIANT"]) in reference_keys, axis=1,
    )].copy()
    if len(data) != len(reference_rows):
        raise ValueError(f"CLAP样本数{len(data)}与统一评价样本数{len(reference_rows)}不一致")

    predictions = []
    fold_reports = []
    for fold in range(fold_count):
        mask = data["version"].map(lambda value: fold_of(value, fold_count) == fold)
        train, test = data[~mask], data[mask]
        model = SVC(kernel="linear")
        model.fit(train[list(FEATURES)].to_numpy(),
                  (train["LABEL"] == "FP").astype(int).to_numpy())
        labels = (test["LABEL"] == "FP").astype(int).to_numpy()
        binary = model.predict(test[list(FEATURES)].to_numpy())
        scores = model.decision_function(test[list(FEATURES)].to_numpy())
        fold_rows = []
        for (_, row), label, prediction, score in zip(
            test.iterrows(), labels, binary, scores,
        ):
            key = (row["version"], row["VARIANT"])
            item = {"version": key[0], "configuration": key[1],
                    "fold": fold, "label": int(label),
                    "prediction": int(prediction), "score": float(score),
                    "group": groups[key]}
            fold_rows.append(item)
            predictions.append(item)
        fold_reports.append({"fold": fold, **_group_metrics(fold_rows)})

    predictions.sort(key=lambda row: (row["version"], row["configuration"]))
    budget_rows = _top_budget(predictions, pulp_budget)
    base_rows = [row for row in predictions if row["group"] == "base_feature"]
    optional_rows = [row for row in predictions if row["group"] != "base_feature"]
    grouped_budget_rows = (
        _top_budget(base_rows, pulp_group_budget["base_feature"]) +
        _top_budget(optional_rows, pulp_group_budget["other_feature"])
    )
    report = {
        "method": "CLAP线性支持向量分类器，按缺陷版本固定五折",
        "features": list(FEATURES), "fold_count": fold_count,
        "folds": fold_reports,
        "binary": {
            "all": _group_metrics(predictions),
            "base_feature": _group_metrics(
                [row for row in predictions if row["group"] == "base_feature"]),
            "optional_feature": _group_metrics(
                [row for row in predictions if row["group"] != "base_feature"]),
        },
        "same_global_budget_as_pulp": {
            "budget": pulp_budget,
            "all": _group_metrics(budget_rows),
            "base_feature": _group_metrics(
                [row for row in budget_rows if row["group"] == "base_feature"]),
            "optional_feature": _group_metrics(
                [row for row in budget_rows if row["group"] != "base_feature"]),
        },
        "same_group_budgets_as_pulp": {
            "base_budget": pulp_group_budget["base_feature"],
            "optional_budget": pulp_group_budget["other_feature"],
            "all": _group_metrics(grouped_budget_rows),
            "base_feature": _group_metrics(
                [row for row in grouped_budget_rows if row["group"] == "base_feature"]),
            "optional_feature": _group_metrics(
                [row for row in grouped_budget_rows if row["group"] != "base_feature"]),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在统一BankAccountTP样本上五折评价CLAP")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    args = parser.parse_args(argv)
    report = run(args.dataset, args.reference, args.output, args.fold_count)
    print(json.dumps(report["binary"]["all"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
