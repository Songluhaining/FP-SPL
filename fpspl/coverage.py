"""Turn per-test Clover coverage into test-local execution evidence.

The feature mapping follows BRFL's FeatureLocationMapper: a generated source
line is aligned to its originating feature class and feature line. Clover
coverage records execution counts, but does not record values or event order.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from fpspl.model import NodeInfo, TestGraph
from fpspl.slicing import observed_reachability_links


def _integer(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


class FeatureLocationMapper:
    def __init__(self, variant_dir: Path):
        self.variant_dir = variant_dir
        self.mapping: dict[tuple[str, str, int], NodeInfo] = {}
        self._source_cache: dict[str, list[str]] = {}
        for label in ("failed", "passed"):
            path = variant_dir / "coverage" / f"spectrum_{label}_coverage.xml"
            if path.is_file():
                self._read_spectrum(path)

    def _source_line(self, package: str, filename: str, line: int) -> str:
        key = f"{package}/{filename}"
        if key not in self._source_cache:
            source_root = self.variant_dir / "src"
            expected = source_root / Path(*package.split(".")) / filename if package else source_root / filename
            matches = [expected] if expected.is_file() else list(source_root.rglob(filename))
            self._source_cache[key] = (
                matches[0].read_text(encoding="utf-8", errors="replace").splitlines()
                if matches else []
            )
        lines = self._source_cache[key]
        return lines[line - 1].strip() if 0 < line <= len(lines) else ""

    def _read_spectrum(self, path: Path) -> None:
        root = ET.parse(path).getroot()
        for package_element in root.findall(".//package"):
            package = package_element.get("name") or ""
            for file_element in package_element.findall("file"):
                filename = Path(file_element.get("path") or file_element.get("name") or "").name
                if not filename:
                    continue
                for line_element in file_element.findall("line"):
                    self._add_spectrum_line(package, filename, line_element)

    def _add_spectrum_line(self, package: str, filename: str, line_element: ET.Element) -> None:
        line = _integer(line_element.get("num"))
        if line < 1:
            return
        feature_class = line_element.get("featureClass") or ""
        feature_line = _integer(line_element.get("featureLineNum")) or line
        feature = feature_class.split(".", 1)[0] if feature_class else "UNKNOWN"
        node_id = f"{feature_class}.{feature_line}" if feature_class else f"{filename}:{line}"
        kind = line_element.get("type") or "stmt"
        if kind == "cond":
            kind = "condition"
        info = NodeInfo(
            node_id=node_id,
            feature=feature,
            source_file=f"{package.replace('.', '/')}/{filename}" if package else filename,
            source_line=line,
            feature_line=feature_line,
            kind=kind,
            code=self._source_line(package, filename, line),
            signature=line_element.get("signature") or "",
        )
        key = (package, filename, line)
        old = self.mapping.get(key)
        if old is None or (old.kind != "condition" and kind == "condition"):
            self.mapping[key] = info

    def get(self, package: str, filename: str, line: int, kind: str = "stmt") -> NodeInfo:
        known = self.mapping.get((package, filename, line))
        if known is not None:
            return known
        stem = Path(filename).stem
        return NodeInfo(
            node_id=f"{package}.{stem}:{line}" if package else f"{stem}:{line}",
            feature="UNKNOWN",
            source_file=f"{package.replace('.', '/')}/{filename}" if package else filename,
            source_line=line,
            feature_line=line,
            kind=kind,
            code=self._source_line(package, filename, line),
        )


def _method_body(source: str, method_name: str) -> str:
    match = re.search(r"\bvoid\s+" + re.escape(method_name) + r"\s*\([^)]*\)\s*(?:throws\s+\w+\s*)?\{", source)
    if not match:
        return ""
    start = match.end()
    depth = 1
    for end in range(start, len(source)):
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
            if depth == 0:
                return source[start:end]
    return source[start:]


def read_observations(variant_dir: Path, test_name: str) -> tuple[str, ...]:
    """Read assertion source. Expected values are available; actual values are not."""
    parts = test_name.split(".")
    if len(parts) < 2:
        return ()
    java_file = variant_dir / "test" / Path(*parts[:-1]).with_suffix(".java")
    if not java_file.is_file():
        return ()
    source = java_file.read_text(encoding="utf-8", errors="replace")
    body = _method_body(source, parts[-1])
    return tuple(
        re.sub(r"\s+", " ", line.strip())
        for line in body.splitlines()
        if re.search(r"\b(?:assert\w+|fail)\s*\(", line)
    )


def read_test_graph(
    version: str,
    variant_dir: Path,
    coverage_file: Path,
    outcome: str,
    mapper: FeatureLocationMapper,
    backward_slices: dict[str, set[str]],
) -> TestGraph:
    if outcome not in {"failed", "passed"}:
        raise ValueError(f"Unknown test outcome: {outcome}")
    if not coverage_file.name.endswith(".coverage.xml"):
        raise ValueError(f"Unexpected coverage filename: {coverage_file}")
    test_name = coverage_file.name.removesuffix(".coverage.xml")
    graph = TestGraph(
        version=version,
        variant=variant_dir.name,
        test=test_name,
        outcome=outcome,
        observations=read_observations(variant_dir, test_name),
        evidence="coverage+slice-reachability" if backward_slices else "coverage-only",
    )
    root = ET.parse(coverage_file).getroot()
    for package_element in root.findall(".//package"):
        package = package_element.get("name") or ""
        for file_element in package_element.findall("file"):
            filename = Path(file_element.get("path") or file_element.get("name") or "").name
            if not filename:
                continue
            for line_element in file_element.findall("line"):
                line = _integer(line_element.get("num"))
                if line < 1:
                    continue
                kind = line_element.get("type") or "stmt"
                true_count = _integer(line_element.get("truecount"))
                false_count = _integer(line_element.get("falsecount"))
                executed = _integer(line_element.get("count")) > 0 or true_count > 0 or false_count > 0
                if not executed:
                    continue
                info = mapper.get(package, filename, line, kind)
                graph.nodes.add(info.node_id)
                graph.node_info[info.node_id] = info
                if true_count:
                    graph.branches.add(info.node_id + ":T")
                if false_count:
                    graph.branches.add(info.node_id + ":F")
    graph.links = observed_reachability_links(graph.nodes, backward_slices)
    return graph
