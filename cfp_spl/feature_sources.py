from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


FeatureMap = dict[tuple[str, int], str]


def load_feature_map(variant_dir: Path) -> FeatureMap:
    """读取组合后源码行到原始特征源码行的映射。"""
    candidates = [
        variant_dir / "coverage" / "spectrum_failed_coverage.xml",
        variant_dir / "coverage" / "spectrum_passed_coverage.xml",
    ]
    mapping: FeatureMap = {}
    for path in candidates:
        if not path.is_file():
            continue
        root = ET.parse(path).getroot()
        for package in root.findall(".//package"):
            package_name = package.get("name", "")
            for file_node in package.findall("file"):
                class_name = Path(file_node.get("name", "")).stem
                runtime_class = f"{package_name}.{class_name}" if package_name else class_name
                for line in file_node.findall("line"):
                    number = line.get("num")
                    feature_class = line.get("featureClass")
                    feature_line = line.get("featureLineNum")
                    if number and feature_class and feature_line:
                        mapping[(runtime_class, int(number))] = f"{feature_class}:{feature_line}"
    return mapping


def runtime_location(stable_key: str) -> tuple[str, int] | None:
    parts = stable_key.rsplit("#", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    return parts[0].split(":", 1)[0], int(parts[1])


def source_identity(stable_key: str, mapping: FeatureMap) -> str:
    location = runtime_location(stable_key)
    return mapping.get(location, stable_key) if location else stable_key
