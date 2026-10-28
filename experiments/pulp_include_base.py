"""Extend PULP's labeling to base-feature faults without modifying PULP itself.

The original PULP label_data skips these versions before calling label(). For
an always-present base feature, every originally passing configuration is a
candidate; the original buggy-statement coverage check then assigns FP/TP.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "pulp_trial"
PULP = ROOT.parent / "PULP"
sys.dont_write_bytecode = True  # Keep imports from creating files in the reference repository.
sys.path.insert(0, str(PULP))

from consistent_testing_manager.LabelData import (  # noqa: E402
    base_is_buggy_features, passing_product_has_buggy_statements, verify_failing_variants,
)
from suspicious_statements_manager.SuspiciousStatementManager import (  # noqa: E402
    get_multiple_buggy_statements, get_mutated_features,
)
from FileManager import get_model_configs_report_path, get_failing_variants  # noqa: E402
import csv  # noqa: E402


def originally_passing(report: Path) -> list[str]:
    with report.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        output_column = "__TEST_OUTPUT__"
        return [row[rows.fieldnames[0]] for row in rows if "PASS" in row[output_column]]


def label_without_class_requirement(version: Path, covered: list[str]) -> dict[str, int]:
    """Apply PULP's label() rules but retain one-class base-fault versions."""
    variants = sorted(path.name for path in (version / "variants").iterdir() if path.is_dir())
    failing = get_failing_variants(str(version))
    unconverted = verify_failing_variants(str(version))
    if "-1" in unconverted:
        raise ValueError(f"Buggy statement absent from failing executions: {version.name}")
    failing_index = {name: index for index, name in enumerate(failing)}
    rows = []
    counts = {"F": 0, "FP": 0, "TP": 0}
    for variant in variants:
        origin = ""
        if variant in failing_index:
            if failing_index[variant] % 2 == 1 and variant not in unconverted:
                label_value, origin = "FP", "1"
            else:
                label_value = "F"
        elif variant in covered:
            label_value = "FP"
        else:
            label_value = "TP"
        rows.append({"VARIANT": variant, "LABEL": label_value, "FP transformed from F": origin})
        counts[label_value] += 1
    with (version / "variant_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["VARIANT", "LABEL", "FP transformed from F"])
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main() -> None:
    statuses = {}
    for version in sorted(path for path in (ROOT / "dataset").iterdir() if path.is_dir()):
        buggy_features = get_mutated_features(str(version))
        if not base_is_buggy_features(["BankAccount", "Base", "ExamDB", "Compress"], buggy_features):
            continue
        label_file = version / "variant_labels.csv"
        buggy_statements = get_multiple_buggy_statements(version.name, str(version))
        candidates = originally_passing(Path(get_model_configs_report_path(str(version))))
        covered = passing_product_has_buggy_statements(str(version), candidates, buggy_statements)
        counts = label_without_class_requirement(version, covered)
        statuses[version.name] = {
            "buggy_features": sorted(buggy_features),
            "originally_passing": len(candidates),
            "executing_buggy_statement": len(set(covered)),
            "label_counts": counts,
            "label_created": label_file.is_file(),
        }
        print(version.name, statuses[version.name], flush=True)
    (ROOT / "base_case_labeling.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8")
    print("base_versions", len(statuses), "labeled", sum(isinstance(v, dict) and v["label_created"] for v in statuses.values()))


if __name__ == "__main__":
    main()
