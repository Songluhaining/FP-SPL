"""Reproduce a system-level PULP trial on a writable dataset copy."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from pulp_ranking_compat import REPO, features  # Applies the 10-formula ranking return-shape fix.
from pulp_include_base import main as include_base_cases
from consistent_testing_manager.LabelData import label_data
from fp_detection.PUBaggingLearning import PUBaggingLearning

BASE = Path(__file__).resolve().parents[2] / "pulp_trial"
DATA = BASE / "dataset"
LOGS = BASE / "slice_logs"
LOGS.mkdir(exist_ok=True)
JAVA = Path(r"E:\Program Files\Java\jdk-1.8\bin\java.exe")
SLICER = REPO / "plugins" / "all-statements-slicing.jar"


def slice_version(version):
    variants = [p for p in (version / "variants").iterdir() if p.is_dir()]
    needed = [
        p for p in variants
        if not (p / "slice.all_statements.backward_slicing.log").is_file()
        or not (p / "slice.all_statements.forward_slicing.log").is_file()
    ]
    if needed:
        command = [str(JAVA), "-Xmx512m", f"-Dvariants_dir={version / 'variants'}", "-jar", str(SLICER)]
        with (LOGS / f"{version.name}.log").open("wb") as log:
            result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            return version.name, f"java returned {result.returncode}"
    missing = sum(
        not (p / "slice.all_statements.backward_slicing.log").is_file()
        or not (p / "slice.all_statements.forward_slicing.log").is_file()
        for p in variants
    )
    return version.name, "ok" if missing == 0 else f"{missing} missing slicing outputs"


def main():
    started = time.time()
    versions = sorted(p for p in DATA.iterdir() if p.is_dir())
    print(f"versions={len(versions)}", flush=True)

    slicing_results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(slice_version, version): version for version in versions}
        for index, future in enumerate(as_completed(tasks), 1):
            name, status = future.result()
            slicing_results[name] = status
            if index % 10 == 0 or status != "ok" or index == len(versions):
                print(f"slicing {index}/{len(versions)}; latest={name}: {status}", flush=True)

    label_data({"BankAccountTP": {"1Bug": str(DATA)}})
    include_base_cases()
    labeled = [v for v in versions if (v / "variant_labels.csv").is_file()]
    print(f"labeled={len(labeled)}", flush=True)

    successful = []
    errors = {}
    for index, version in enumerate(labeled, 1):
        try:
            if not (version / "attributes.csv").is_file():
                features.calculate_attributes3(
                    str(version), "variant_labels.csv", "attributes_temp.csv", "attributes.csv"
                )
            df = pd.read_csv(version / "attributes.csv")
            if df.shape[1] != 52 or df.iloc[:, 2:].isna().any().any():
                raise ValueError(f"unexpected features: shape={df.shape}")
            df.insert(0, "version", version.name)
            successful.append(df)
        except Exception as exc:
            errors[version.name] = f"{type(exc).__name__}: {exc}"
        if index % 10 == 0 or index == len(labeled):
            print(f"features {index}/{len(labeled)}; successful={len(successful)}", flush=True)

    if not successful:
        raise RuntimeError("No versions produced features")
    data = pd.concat(successful, ignore_index=True)
    labels = data["LABEL"].to_numpy()
    matrix = data.iloc[:, 3:].to_numpy()
    classifier = PUBaggingLearning()
    classifier.prepare_data(matrix, labels, None)
    prediction = classifier.predict()
    score = sum(
        weight * estimator.predict_proba(classifier.passing_feature)[:, 1]
        for estimator, weight in zip(classifier.estimators, classifier.estimator_weights)
    )

    passing = data[data["LABEL"] != "F"].copy()
    passing["prediction"] = np.where(prediction == 1, "FP", "TP")
    passing["score"] = score
    passing[["version", "model", "LABEL", "prediction", "score"]].to_csv(
        BASE / "predictions_system.csv", index=False
    )
    y_true = (passing["LABEL"] == "FP").astype(int).to_numpy()
    fp_origins = {}
    for version in labeled:
        with (version / "variant_labels.csv").open(newline="") as source:
            for row in csv.DictReader(source):
                if row["LABEL"] == "FP":
                    fp_origins[(version.name, row["VARIANT"])] = (
                        "converted_from_failing"
                        if row["FP transformed from F"] == "1"
                        else "originally_passing"
                    )
    fp_breakdown = {}
    for origin in ("converted_from_failing", "originally_passing"):
        subset = passing[
            passing.apply(
                lambda row: fp_origins.get((row["version"], row["model"])) == origin,
                axis=1,
            )
        ]
        fp_breakdown[origin] = {
            "total": len(subset),
            "detected": int((subset["prediction"] == "FP").sum()),
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=[0, 1], zero_division=0
    )
    results = {
        "source": str(DATA),
        "versions_found": len(versions),
        "versions_labeled": len(labeled),
        "versions_with_features": len(successful),
        "versions_excluded_for_base_feature_fault": len(versions) - len(labeled),
        "slicing_errors": {k: v for k, v in slicing_results.items() if v != "ok"},
        "feature_errors": errors,
        "fp_origin_breakdown": fp_breakdown,
        "samples": {k: int((data["LABEL"] == k).sum()) for k in ("F", "FP", "TP")},
        "feature_count": matrix.shape[1],
        "confusion_matrix_rows_true_TP_FP_columns_predicted_TP_FP": confusion_matrix(
            y_true, prediction, labels=[0, 1]
        ).tolist(),
        "accuracy": accuracy_score(y_true, prediction),
        "TP": {"precision": precision[0], "recall": recall[0], "f1": f1[0], "support": support[0]},
        "FP": {"precision": precision[1], "recall": recall[1], "f1": f1[1], "support": support[1]},
        "seconds": time.time() - started,
    }
    (BASE / "metrics_system.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=lambda x: x.item()),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=lambda x: x.item()), flush=True)


if __name__ == "__main__":
    main()
