from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .domain import FactorGraph, StatementPosterior
from .factor_graph import causal_statement_keys
from .feature_sources import FeatureMap, source_identity
from .joint_graph import _groups


@dataclass(frozen=True)
class HierarchicalCounterfactualGraph:
    """共享源码缺陷变量与A/B局部表现变量组成的分层反事实图。"""

    graph: FactorGraph
    global_source: dict[str, str]
    global_to_local: dict[str, dict[str, str]]
    failing_causal_sources: frozenset[str]
    passing_causal_sources: frozenset[str]
    local_statement_nodes: frozenset[str]

    @property
    def shared_defects(self) -> frozenset[str]:
        return frozenset(
            node for node, local in self.global_to_local.items()
            if set(local) == {"A", "B"}
        )


def _short(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _manifestation_tensor(probability_if_defect: float, noise: float) -> list[float]:
    """P(局部错误|源码缺陷)；变量的真状态均表示正确。"""
    if not 0.0 < noise < probability_if_defect < 1.0:
        raise ValueError("表现概率必须满足 0 < 噪声 < 缺陷表现概率 < 1")
    return [
        probability_if_defect, noise,
        1.0 - probability_if_defect, 1.0 - noise,
    ]


def build_hierarchical_counterfactual_graph(
    failing: FactorGraph,
    passing: FactorGraph,
    failing_feature_map: FeatureMap,
    passing_feature_map: FeatureMap,
    failing_manifestation: float = 0.95,
    passing_manifestation: float = 0.50,
    noise: float = 0.01,
) -> HierarchicalCounterfactualGraph:
    """保留两侧局部执行图，用软因子连接到共享的源码缺陷变量。"""
    groups = {
        "B": _groups(failing, failing_feature_map),
        "A": _groups(passing, passing_feature_map),
    }
    graph = FactorGraph()
    global_source: dict[str, str] = {}
    global_to_local: dict[str, dict[str, str]] = {}
    local_nodes: set[str] = set()
    statement_maps: dict[str, dict[str, str]] = {"A": {}, "B": {}}

    for source in sorted(set(groups["A"]) | set(groups["B"])):
        count = max(len(groups["A"].get(source, [])), len(groups["B"].get(source, [])))
        for index in range(count):
            global_node = f"d:{_short(f'{source}|{index}')}"
            graph.statements[global_node] = StatementPosterior(
                global_node, source, source, 0.5, False, False,
            )
            global_source[global_node] = source
            global_to_local[global_node] = {}
            for side, original_graph in (("A", passing), ("B", failing)):
                nodes = groups[side].get(source, [])
                if index >= len(nodes):
                    continue
                original_node = nodes[index]
                original = original_graph.statements[original_node]
                local = f"{side}:s:{original_node}"
                statement_maps[side][original_node] = local
                local_nodes.add(local)
                global_to_local[global_node][side] = local
                graph.statements[local] = StatementPosterior(
                    local, original.raw_name, original.stable_key, original.correctness,
                    original.observed, original.reduced, original.observed_value,
                )

                factor = f"link:{side}:{global_node}"
                manifestation = failing_manifestation if side == "B" else passing_manifestation
                graph.factors[factor] = {
                    "statement": source,
                    "operators": ["manifestation"],
                    "tensor": _manifestation_tensor(manifestation, noise),
                }
                graph.incidences.extend([
                    (factor, global_node, "DEFECT"),
                    (factor, local, "MANIFESTATION"),
                ])

    def copy_execution(side: str, original: FactorGraph) -> None:
        node_map = dict(statement_maps[side])
        for node, value in original.values.items():
            mapped = f"{side}:v:{node}"
            node_map[node] = mapped
            graph.values[mapped] = dict(value)
        for factor, data in original.factors.items():
            graph.factors[f"{side}:f:{factor}"] = dict(data)
        for factor, node, role in original.incidences:
            mapped = node_map.get(node)
            if mapped is not None:
                graph.incidences.append((f"{side}:f:{factor}", mapped, role))

    copy_execution("B", failing)
    copy_execution("A", passing)
    causal_b = frozenset(
        source_identity(key, failing_feature_map) for key in causal_statement_keys(failing)
    )
    causal_a = frozenset(
        source_identity(key, passing_feature_map) for key in causal_statement_keys(passing)
    )
    return HierarchicalCounterfactualGraph(
        graph, global_source, global_to_local, causal_b, causal_a,
        frozenset(local_nodes),
    )
