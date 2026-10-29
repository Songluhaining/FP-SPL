from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .domain import FactorGraph, StatementPosterior
from .factor_graph import causal_statement_keys
from .feature_sources import FeatureMap, source_identity


@dataclass(frozen=True)
class JointCounterfactualGraph:
    """一张联合图：两侧运行值独立，对齐语句共享正确性变量。"""

    graph: FactorGraph
    statement_source: dict[str, str]
    statement_sides: dict[str, frozenset[str]]
    failing_causal_sources: frozenset[str]
    passing_causal_sources: frozenset[str]

    @property
    def shared_statements(self) -> frozenset[str]:
        return frozenset(
            node for node, sides in self.statement_sides.items() if sides == {"A", "B"}
        )


def _short(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _groups(graph: FactorGraph, feature_map: FeatureMap) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for node, statement in graph.statements.items():
        source = source_identity(statement.stable_key, feature_map)
        grouped.setdefault(source, []).append(node)
    for nodes in grouped.values():
        nodes.sort(key=lambda node: (graph.statements[node].stable_key, node))
    return grouped


def build_joint_counterfactual_graph(
    failing: FactorGraph,
    passing: FactorGraph,
    failing_feature_map: FeatureMap,
    passing_feature_map: FeatureMap,
) -> JointCounterfactualGraph:
    """合并B失败轨迹与A通过轨迹，并按源码行内顺序共享语句变量。"""
    groups_b = _groups(failing, failing_feature_map)
    groups_a = _groups(passing, passing_feature_map)
    map_b: dict[str, str] = {}
    map_a: dict[str, str] = {}
    sources: dict[str, str] = {}
    sides: dict[str, set[str]] = {}

    for source in sorted(set(groups_a) | set(groups_b)):
        nodes_a, nodes_b = groups_a.get(source, []), groups_b.get(source, [])
        shared = min(len(nodes_a), len(nodes_b))
        for index in range(shared):
            joint = f"s:shared:{_short(f'{source}|{index}')}"
            map_a[nodes_a[index]] = joint
            map_b[nodes_b[index]] = joint
            sources[joint] = source
            sides[joint] = {"A", "B"}
        for side, nodes, mapping in (("A", nodes_a[shared:], map_a), ("B", nodes_b[shared:], map_b)):
            for index, node in enumerate(nodes, shared):
                joint = f"s:{side}:{_short(f'{source}|{index}|{node}')}"
                mapping[node] = joint
                sources[joint] = source
                sides[joint] = {side}

    joint = FactorGraph()
    reverse_b = {mapped: old for old, mapped in map_b.items()}
    reverse_a = {mapped: old for old, mapped in map_a.items()}
    for node, source in sources.items():
        original_id = reverse_b.get(node)
        original = (failing.statements[original_id] if original_id is not None
                    else passing.statements[reverse_a[node]])
        joint.statements[node] = StatementPosterior(
            node, source, source, 0.5,
            original.observed, original.reduced, original.observed_value,
        )

    def copy_side(side: str, graph: FactorGraph, statement_map: dict[str, str]) -> None:
        node_map = dict(statement_map)
        for node, value in graph.values.items():
            mapped = f"{side}:v:{node}"
            node_map[node] = mapped
            joint.values[mapped] = dict(value)
        for factor, data in graph.factors.items():
            joint.factors[f"{side}:f:{factor}"] = dict(data)
        for factor, node, role in graph.incidences:
            mapped = node_map.get(node)
            if mapped is not None:
                joint.incidences.append((f"{side}:f:{factor}", mapped, role))

    copy_side("B", failing, map_b)
    copy_side("A", passing, map_a)
    causal_b = frozenset(
        source_identity(key, failing_feature_map) for key in causal_statement_keys(failing)
    )
    causal_a = frozenset(
        source_identity(key, passing_feature_map) for key in causal_statement_keys(passing)
    )
    return JointCounterfactualGraph(
        joint, sources, {node: frozenset(value) for node, value in sides.items()},
        causal_b, causal_a,
    )
