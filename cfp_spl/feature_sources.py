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
                # Clover的name可能是EmailSystem.Client（不是文件扩展名）。
                # 对它调用Path.stem会错误截成EmailSystem并覆盖不同类的行映射。
                file_path = file_node.get("path", "").replace("\\", "/")
                name = file_node.get("name", "")
                if file_path.endswith(".java"):
                    class_name = Path(file_path).stem
                    runtime_class = f"{package_name}.{class_name}" if package_name else class_name
                elif name.endswith(".java"):
                    class_name = Path(name.replace("\\", "/")).stem
                    runtime_class = f"{package_name}.{class_name}" if package_name else class_name
                else:
                    runtime_class = name if "." in name else f"{package_name}.{name}" if package_name else name
                for line in file_node.findall("line"):
                    number = line.get("num")
                    feature_class = line.get("featureClass")
                    feature_line = line.get("featureLineNum")
                    if number and feature_class and feature_line:
                        key, value = (runtime_class, int(number)), f"{feature_class}:{feature_line}"
                        if key in mapping and mapping[key] != value:
                            raise ValueError(f"冲突的特征源码映射：{key}，{mapping[key]} / {value}")
                        mapping[key] = value
    return mapping


def runtime_location(stable_key: str) -> tuple[str, int] | None:
    parts = stable_key.rsplit("#", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    return parts[0].split(":", 1)[0], int(parts[1])


def source_identity(stable_key: str, mapping: FeatureMap) -> str:
    location = runtime_location(stable_key)
    return mapping.get(location, stable_key) if location else stable_key
