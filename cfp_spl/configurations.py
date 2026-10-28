from __future__ import annotations

import csv
import math
from pathlib import Path

from .domain import Configuration


TRUE_VALUES = {"1", "t", "true", "yes", "selected"}
FALSE_VALUES = {"0", "f", "false", "no", "unselected"}


def _bit(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return 1
    if normalized in FALSE_VALUES:
        return 0
    raise ValueError(f"无法识别的特征取值：{value!r}")


def _outcome(value: str) -> str:
    normalized = value.strip().lower().strip("_")
    if normalized in {"passed", "pass"}:
        return "passed"
    if normalized in {"failed", "fail"}:
        return "failed"
    raise ValueError(f"无法识别的测试结果：{value!r}")


def load_configurations(version_dir: Path) -> list[Configuration]:
    report = version_dir / "config.report.csv.done"
    with report.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) < 3:
            raise ValueError(f"配置报告表头不完整：{report}")
        name_col = reader.fieldnames[0]
        result_col = next(
            (c for c in reader.fieldnames if c.upper() == "__TEST_OUTPUT__"),
            reader.fieldnames[-1],
        )
        features = tuple(c for c in reader.fieldnames if c not in {name_col, result_col})
        rows = []
        for row in reader:
            name = row[name_col].strip()
            rows.append(
                Configuration(
                    name=name,
                    features=tuple(_bit(row[f]) for f in features),
                    feature_names=features,
                    outcome=_outcome(row[result_col]),
                    variant_dir=version_dir / "variants" / name,
                )
            )
    if not rows:
        raise ValueError(f"配置报告为空：{report}")
    return rows


def euclidean_distance(a: Configuration, b: Configuration) -> float:
    if a.feature_names != b.feature_names:
        raise ValueError("两个配置的特征顺序不一致")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a.features, b.features)))


def feature_delta(a: Configuration, b: Configuration) -> list[dict]:
    return [
        {"feature": name, "a": x, "b": y}
        for name, x, y in zip(a.feature_names, a.features, b.features)
        if x != y
    ]


def nearest_failing(
    passing: Configuration, all_configs: list[Configuration], count: int = 1
) -> list[tuple[Configuration, float]]:
    failures = [c for c in all_configs if c.outcome == "failed"]
    if not failures:
        raise ValueError("当前版本没有未通过配置，无法建立反事实参照")
    ranked = sorted(
        ((c, euclidean_distance(passing, c)) for c in failures),
        key=lambda item: (item[1], item[0].name),
    )
    return ranked[: max(1, count)]
