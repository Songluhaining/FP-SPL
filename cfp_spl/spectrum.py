from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class SpectrumEvidence:
    failed_covered: int
    passed_covered: int
    total_failed: int
    total_passed: int
    ochiai: float
    tarantula: float
    barinel: float
    jaccard: float
    suspiciousness: float
    error_prior: float


def _coverage(path: Path) -> tuple[int, dict[str, int]]:
    if not path.is_file():
        return 0, {}
    root = ET.parse(path).getroot()
    tests = root.find("tests")
    total = int(tests.get("count", "0")) if tests is not None else 0
    counts: dict[str, int] = {}
    for line in root.findall(".//line"):
        feature_class = line.get("featureClass")
        feature_line = line.get("featureLineNum")
        if not feature_class or not feature_line:
            continue
        key = f"{feature_class}:{feature_line}"
        counts[key] = max(counts.get(key, 0), int(line.get("count", "0")))
    return total, counts


def _scores(ef: int, ep: int, total_failed: int, total_passed: int) -> tuple[float, ...]:
    if ef <= 0:
        return 0.0, 0.0, 0.0, 0.0
    ochiai = ef / math.sqrt(max(1, total_failed) * max(1, ef + ep))
    failed_rate = ef / max(1, total_failed)
    passed_rate = ep / max(1, total_passed)
    tarantula = failed_rate / (failed_rate + passed_rate) if failed_rate + passed_rate else 0.0
    barinel = ef / (ef + ep)
    jaccard = ef / (max(1, total_failed) + ep)
    return tuple(max(0.0, min(1.0, value)) for value in (ochiai, tarantula, barinel, jaccard))


def load_spectrum_priors(
    variant_dir: Path,
    metric: str = "ensemble",
    minimum_error_prior: float = 0.01,
    maximum_error_prior: float = 0.90,
) -> dict[str, SpectrumEvidence]:
    """从聚合覆盖报告得到语句错误先验。

    覆盖文件中的count沿用BRFL实践，解释为覆盖该语句的测试数；若覆盖工具
    给出的是执行次数，则将其截断到测试总数，以保持频谱公式有效。
    默认对四个有界频谱指标取中位数，避免重新依赖单一指标选择。
    """
    if metric not in {"ensemble", "ochiai", "tarantula", "barinel", "jaccard"}:
        raise ValueError(f"不支持的频谱指标：{metric}")
    coverage = variant_dir / "coverage"
    total_failed, failed = _coverage(coverage / "spectrum_failed_coverage.xml")
    total_passed, passed = _coverage(coverage / "spectrum_passed_coverage.xml")
    result: dict[str, SpectrumEvidence] = {}
    for key in sorted(set(failed) | set(passed)):
        ef = min(total_failed, failed.get(key, 0)) if total_failed else 0
        ep = min(total_passed, passed.get(key, 0)) if total_passed else 0
        values = _scores(ef, ep, total_failed, total_passed)
        by_name = dict(zip(("ochiai", "tarantula", "barinel", "jaccard"), values))
        suspiciousness = median(values) if metric == "ensemble" else by_name[metric]
        error_prior = minimum_error_prior + (
            maximum_error_prior - minimum_error_prior
        ) * suspiciousness
        result[key] = SpectrumEvidence(
            ef, ep, total_failed, total_passed,
            by_name["ochiai"], by_name["tarantula"], by_name["barinel"],
            by_name["jaccard"], suspiciousness, error_prior,
        )
    return result


def serialize_spectrum(values: dict[str, SpectrumEvidence]) -> dict[str, dict]:
    return {key: asdict(value) for key, value in values.items()}
