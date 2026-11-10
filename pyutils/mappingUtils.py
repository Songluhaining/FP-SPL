import re
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, Tuple

def build_feature_mapping(coverage_dir: Path) -> Dict[Tuple[str, int], Tuple[str, int]]:
    """
    返回：
      (原始 className, 源码行号) -> (featureClass, featureLineNum)
    """
    mapping = {}

    def _process_xml(xml_path: Path):
        if not xml_path.exists():
            return
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 遍历 <package>/<file>/<line>
        for pkg in root.findall(".//package"):
            for f in pkg.findall("file"):
                # 例：name="ElevatorSystem.Elevator"
                cls_name = f.get("name")
                if not cls_name:
                    continue
                # 遍历 line
                for line in f.findall("line"):
                    if line.get("type") != "stmt":
                        # 只关心语句行；如果你也想区分 method，可以改成不过滤
                        continue
                    num = line.get("num")
                    feature_class = line.get("featureClass")
                    feature_line = line.get("featureLineNum")
                    if not (num and feature_class and feature_line):
                        continue
                    try:
                        num_i = int(num)
                        feature_line_i = int(feature_line)
                    except ValueError:
                        continue

                    key = (cls_name, num_i)
                    mapping[key] = (feature_class, feature_line_i)

    _process_xml(coverage_dir / "spectrum_failed_coverage.xml")
    _process_xml(coverage_dir / "spectrum_passed_coverage.xml")

    return mapping

LINEINFO_RE = re.compile(r"(lineinfo=)([^,]+)")

def rewrite_lineinfo(line: str,
                     mapping: Dict[Tuple[str, int], Tuple[str, int]]) -> str:
    """
    把一行里的 lineinfo=... 按照 mapping 替换成 featureClass + featureLineNum
    """
    m = LINEINFO_RE.search(line)
    if not m:
        return line

    value = m.group(2)
    # 例：ElevatorSystem.Environment#<init>#(I)V#29#16
    parts = value.split("#")
    if len(parts) < 5:
        return line

    cls_name, method, desc, src_line_str, offset_str = parts[:5]

    try:
        src_line = int(src_line_str)
    except ValueError:
        return line

    key = (cls_name, src_line)
    if key not in mapping:
        # 找不到映射，就保持原状
        return line

    feature_class, feature_line = mapping[key]

    # 用特征类 + 特征行号替换
    parts[0] = feature_class       # Base.ElevatorSystem.Elevator / Overloaded.ElevatorSystem.Elevator 等
    parts[3] = str(feature_line)   # featureLineNum

    new_value = "#".join(parts)
    # 拼回整行
    return line[:m.start(2)] + new_value + line[m.end(2):]


def rewrite_logs_for_variant(variant_root: Path):
    coverage_dir = variant_root / "coverage"
    mapping = build_feature_mapping(coverage_dir)
    if not mapping:
        print(f"[rewrite_logs] no mapping built for {variant_root}, skip")
        return

    logs_root = variant_root / "trace" / "logs"
    if not logs_root.exists():
        print(f"[rewrite_logs] no logs dir: {logs_root}")
        return

    print(f"[rewrite_logs] mapping size = {len(mapping)}, logs root = {logs_root}")

    for log_path in logs_root.rglob("*.log"):
        # 包括 run/*.log, mytrace/*.source.log 等，全都改一遍
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        new_lines = []
        changed = False
        for line in text.splitlines():
            new_line = rewrite_lineinfo(line, mapping)
            if new_line != line:
                changed = True
            new_lines.append(new_line)
        if changed:
            log_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            print(f"[rewrite_logs] updated {log_path}")
        else:
            print(f"[rewrite_logs] no change in {log_path}")