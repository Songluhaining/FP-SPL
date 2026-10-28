"""Build test-local BRFL runtime graphs and train the weakly supervised ranker."""

from __future__ import annotations

import base64
import csv
import gzip
import json
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from fpspl.brfl import BrflRuntime
from fpspl.coverage import FeatureLocationMapper, read_observations
from fpspl.model import NodeInfo, TestGraph
from fpspl.pipeline import read_observed_outcomes
from fpspl.pu_model import graph_similarity, train_pu_model


RANKING_FIELDS = ["version", "variant", "rank", "score", "model_score", "template_similarity", "status",
                  "best_failure_variant", "best_failure_test", "best_passing_test", "matched_typed_edges",
                  "typed_edges_similarity", "nodes_similarity", "branches_similarity", "observations_similarity"]


def _trace_location(lineinfo: str, variant_dir: Path, mapper: FeatureLocationMapper) -> NodeInfo:
    # BRFL statement names are Class:method:descriptor#line#bytecode-index.
    prefix, line, bytecode = lineinfo.rsplit("#", 2)
    class_name, method, *_ = prefix.split(":")
    source_line = int(line)
    package, short_name = (class_name.rsplit(".", 1) if "." in class_name else ("", class_name))
    if "_ESTest" in short_name:
        source_file = variant_dir / "test" / Path(*package.split(".")) / (short_name + ".java")
        lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines() if source_file.is_file() else []
        code = lines[source_line - 1].strip() if 0 < source_line <= len(lines) else ""
        return NodeInfo(f"TEST:{class_name}:{source_line if source_line > 0 else method + ':' + bytecode}",
                        "TEST", str(source_file), source_line, source_line, "test", code, method)
    mapped = mapper.get(package, short_name + ".java", source_line)
    if source_line < 1:
        return replace(mapped, node_id=f"{class_name}:{method}:bytecode:{bytecode}", feature="UNKNOWN")
    return mapped


def _branches(coverage_file: Path, mapper: FeatureLocationMapper) -> set[str]:
    found: set[str] = set()
    root = ET.parse(coverage_file).getroot()
    for package in root.findall(".//package"):
        package_name = package.get("name") or ""
        for source in package.findall("file"):
            filename = Path(source.get("path") or source.get("name") or "").name
            for line in source.findall("line"):
                if line.get("type") != "cond":
                    continue
                number = int(line.get("num") or 0)
                if number < 1:
                    continue
                node = mapper.get(package_name, filename, number).node_id
                if int(line.get("truecount") or 0):
                    found.add(node + ":T")
                if int(line.get("falsecount") or 0):
                    found.add(node + ":F")
    return found


def _localize(graph: TestGraph) -> None:
    """Keep the backward dependency cone of a reached test assertion when available."""
    anchors = {node for node in graph.nodes if (info := graph.node_info.get(node))
               and info.feature == "TEST" and re.search(r"\b(?:assert\w+|fail)\s*\(", info.code)}
    if not anchors:
        return
    predecessors: dict[str, set[str]] = defaultdict(set)
    for source, target, _ in graph.typed_links:
        predecessors[target].add(source)
    cone = set(anchors)
    pending = list(anchors)
    while pending:
        for source in predecessors[pending.pop()]:
            if source not in cone:
                cone.add(source)
                pending.append(source)
    if not any(graph.node_info.get(node) and graph.node_info[node].feature not in {"TEST", "UNKNOWN"}
               for node in cone):
        return
    graph.nodes.intersection_update(cone)
    graph.typed_links = {edge for edge in graph.typed_links if edge[0] in cone and edge[1] in cone}
    graph.links = {(source, target) for source, target, _ in graph.typed_links}
    graph.node_info = {node: info for node, info in graph.node_info.items() if node in cone}
    graph.runtime_values = {node: values for node, values in graph.runtime_values.items() if node in cone}
    graph.branches = {branch for branch in graph.branches if branch.rsplit(":", 1)[0] in cone}


def read_dynamic_graph(
    version: str, variant_dir: Path, coverage_file: Path, outcome: str,
    tsv_file: Path, result: dict, mapper: FeatureLocationMapper,
) -> TestGraph:
    test_name = coverage_file.name.removesuffix(".coverage.xml")
    graph = TestGraph(version, variant_dir.name, test_name, outcome,
                      observations=read_observations(variant_dir, test_name),
                      evidence="brfl-runtime-factor-v1", observed_result=result)
    graph.branches = _branches(coverage_file, mapper)
    location_cache: dict[str, NodeInfo] = {}
    def locate(lineinfo: str) -> NodeInfo:
        if lineinfo not in location_cache:
            location_cache[lineinfo] = _trace_location(lineinfo, variant_dir, mapper)
        return location_cache[lineinfo]
    for raw in tsv_file.read_text(encoding="utf-8").splitlines():
        fields = raw.split("\t")
        if fields[0] == "NODE" and len(fields) == 3:
            info = locate(fields[2])
            graph.nodes.add(info.node_id)
            graph.node_info[info.node_id] = info
        elif fields[0] == "EDGE" and len(fields) == 5:
            source, target = locate(fields[2]), locate(fields[3])
            if source.node_id != target.node_id:
                graph.nodes.update((source.node_id, target.node_id))
                graph.node_info[source.node_id] = source
                graph.node_info[target.node_id] = target
                graph.typed_links.add((source.node_id, target.node_id, fields[4]))
                graph.links.add((source.node_id, target.node_id))
        elif fields[0] == "VALUE" and len(fields) == 4:
            info = locate(fields[2])
            value = base64.b64decode(fields[3]).decode("utf-8", errors="replace")
            graph.runtime_values.setdefault(info.node_id, []).append(value)
        elif fields[0] == "SUMMARY" and len(fields) >= 3:
            graph.event_count = int(fields[1])
    if graph.event_count == 0:
        raise ValueError(f"No BRFL runtime factors: {tsv_file}")
    _localize(graph)
    graph.observed_result["assertion_runtime_values"] = {
        node: values for node, values in graph.runtime_values.items()
        if (info := graph.node_info.get(node)) and info.feature == "TEST"
        and re.search(r"\b(?:assert\w+|fail)\s*\(", info.code)
    }
    return graph


def run_dynamic_version(
    version_dir: Path, output_dir: Path, runtime: BrflRuntime,
    max_variants: int | None = None, max_tests: int | None = None,
    limit_failed_variants: int | None = None, limit_passing_variants: int | None = None,
    model_kind: str = "graph",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = read_observed_outcomes(version_dir)
    positive: list[TestGraph] = []
    passing: dict[str, list[TestGraph]] = {}
    errors: list[dict[str, str]] = []
    processed_variants = 0
    outcome_counts = {"failed": 0, "passed": 0}
    for variant, variant_outcome in sorted(outcomes.items(), key=lambda item: (item[1] != "failed", item[0])):
        if max_variants is not None and processed_variants >= max_variants:
            break
        limit = limit_failed_variants if variant_outcome == "failed" else limit_passing_variants
        if limit is not None and outcome_counts[variant_outcome] >= limit:
            continue
        variant_dir = version_dir / "variants" / variant
        coverage_files = sorted((variant_dir / "coverage" / variant_outcome).glob("*.coverage.xml"))
        if max_tests is not None:
            coverage_files = coverage_files[:max_tests]
        if not coverage_files:
            errors.append({"variant": variant, "error": "no per-test coverage files"})
            continue
        variant_work = output_dir / "instrumented" / variant
        variant_work.mkdir(parents=True, exist_ok=True)
        print(f"  [{processed_variants + 1}] {variant} ({variant_outcome}), tests={len(coverage_files)}", flush=True)
        try:
            built = runtime.recompile_variant(variant_dir, variant_work)
            mapper = FeatureLocationMapper(variant_dir)
            graphs = []
            for coverage_file in coverage_files:
                try:
                    tsv, result = runtime.trace_test(variant_dir, variant_work, coverage_file, variant_outcome, built)
                    graphs.append(read_dynamic_graph(version_dir.name, variant_dir, coverage_file,
                                                     variant_outcome, tsv, result, mapper))
                except Exception as exc:
                    errors.append({"variant": variant, "test": coverage_file.name, "error": str(exc)})
            if variant_outcome == "failed":
                positive.extend(graphs)
            else:
                passing[variant] = graphs
            processed_variants += 1
            outcome_counts[variant_outcome] += 1
        except Exception as exc:
            errors.append({"variant": variant, "error": str(exc)})
    unlabeled = [graph for graphs in passing.values() for graph in graphs]
    unlabeled_index = {id(graph): index for index, graph in enumerate(unlabeled)}
    if model_kind == "graph":
        from fpspl.graph_model import train_graph_pu_model
        model = train_graph_pu_model(positive, unlabeled)
        model_scores = model.score_all(unlabeled) if model is not None else []
        if model is not None:
            model.save(output_dir / "graph_model.pt")
    elif model_kind == "linear":
        model = train_pu_model(positive, unlabeled)
        model_scores = [model.score(graph, index) for index, graph in enumerate(unlabeled)] if model else []
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")
    if model_kind == "linear" and model is not None:
        with (output_dir / "pu_model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
    rows = []
    for variant, graphs in passing.items():
        comparisons = []
        for graph in graphs:
            if not positive:
                continue
            best_template, (similarity, components) = max(
                ((template, graph_similarity(template, graph)) for template in positive),
                key=lambda pair: pair[1][0],
            )
            model_score = model_scores[unlabeled_index[id(graph)]] if model is not None else None
            score = (model_score if model_kind == "graph" else 0.65 * model_score + 0.35 * similarity) \
                if model_score is not None else similarity
            comparisons.append((score, graph, best_template, model_score, similarity, components))
        if comparisons:
            score, graph, template, model_score, similarity, components = max(comparisons, key=lambda item: item[0])
            rows.append({"version": version_dir.name, "variant": variant, "score": round(score, 6),
                         "model_score": round(model_score, 6) if model_score is not None else "",
                         "template_similarity": round(similarity, 6), "rank": "",
                         "status": "ranked" if model is not None else "no_trained_model",
                         "best_failure_variant": template.variant, "best_failure_test": template.test,
                         "best_passing_test": graph.test, "matched_typed_edges": len(template.typed_links & graph.typed_links),
                         **{key + "_similarity": round(value, 6) for key, value in components.items()}})
        else:
            rows.append({"version": version_dir.name, "variant": variant, "status": "insufficient_evidence"})
    rows.sort(key=lambda row: (-row.get("score", -1), row["variant"]))
    for index, row in enumerate(rows, 1):
        if "score" in row:
            row["rank"] = index
    with (output_dir / "ranking.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with gzip.open(output_dir / "test_graphs.jsonl.gz", "wt", encoding="utf-8") as handle:
        for graph in positive + unlabeled:
            handle.write(json.dumps(graph.to_dict(), ensure_ascii=False) + "\n")
    (output_dir / "failure_templates.json").write_text(
        json.dumps([graph.to_dict() for graph in positive], ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"version": version_dir.name, "processed_variants": processed_variants,
               "positive_failed_tests": len(positive), "unlabeled_passing_tests": len(unlabeled),
               "trained_model": model is not None, "ranked_variants": sum("score" in row for row in rows),
               "model_kind": model_kind,
               "trace_evidence": "BRFL bytecode instrumentation and runtime factor edges",
               "no_ground_truth_used_for_training": True, "errors": errors}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
