from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import FactorGraph
from .factor_graph import causal_statement_keys
from .feature_sources import FeatureMap, source_identity
from .joint_graph import _groups


NODE_KINDS = ("statement", "value", "factor", "feature", "test")
RELATIONS = (
    "a_incidence", "b_incidence", "aligned_statement",
    "feature_source", "test_context",
)


@dataclass(frozen=True)
class PairedExecutionGraph:
    """供学习模型使用的一张A/B配对动态差异图。"""

    node_ids: tuple[str, ...]
    node_features: np.ndarray
    edges: np.ndarray
    edge_types: np.ndarray
    relation_names: tuple[str, ...]
    aligned_statements: tuple[tuple[str, str, str], ...]
    test_similarity: float
    oracle_similarity: float

    def validate(self) -> None:
        if self.node_features.shape != (len(self.node_ids), 18):
            raise ValueError("配对图节点特征维度错误")
        if self.edges.ndim != 2 or self.edges.shape[0] != 2:
            raise ValueError("配对图边索引必须为2×E")
        if self.edges.shape[1] != self.edge_types.size:
            raise ValueError("边与关系类型数量不一致")
        if self.edges.size and (self.edges.min() < 0 or self.edges.max() >= len(self.node_ids)):
            raise ValueError("配对图边引用了不存在的节点")


def _base_features(kind: str, side: str) -> np.ndarray:
    values = np.zeros(18, dtype=np.float32)
    values[NODE_KINDS.index(kind)] = 1.0
    values[5] = float(side == "A")
    values[6] = float(side == "B")
    values[7] = float(side == "shared")
    return values


def build_paired_execution_graph(
    failing: FactorGraph,
    passing: FactorGraph,
    failing_feature_map: FeatureMap,
    passing_feature_map: FeatureMap,
    feature_delta: list[dict],
    test_similarity: float,
    oracle_similarity: float,
    test_provenance: str,
) -> PairedExecutionGraph:
    """构造保留两侧局部结构、源码对齐和特征来源的差异图。"""
    ids: list[str] = []
    features: list[np.ndarray] = []
    index: dict[str, int] = {}
    edges: list[tuple[int, int]] = []
    edge_types: list[int] = []
    aligned: list[tuple[str, str, str]] = []
    source_by_node: dict[str, str] = {}

    causal = {
        "A": {source_identity(key, passing_feature_map)
              for key in causal_statement_keys(passing)},
        "B": {source_identity(key, failing_feature_map)
              for key in causal_statement_keys(failing)},
    }

    def add(node_id: str, values: np.ndarray) -> int:
        index[node_id] = len(ids)
        ids.append(node_id)
        features.append(values)
        return index[node_id]

    def connect(left: str, right: str, relation: str, bidirectional: bool = True) -> None:
        relation_id = RELATIONS.index(relation)
        edges.append((index[left], index[right]))
        edge_types.append(relation_id)
        if bidirectional:
            edges.append((index[right], index[left]))
            edge_types.append(relation_id)

    for side, graph, mapping in (
        ("A", passing, passing_feature_map), ("B", failing, failing_feature_map),
    ):
        for node, statement in graph.statements.items():
            identity = f"{side}:s:{node}"
            source = source_identity(statement.stable_key, mapping)
            values = _base_features("statement", side)
            values[8] = float(statement.observed)
            values[9] = float(statement.observed_value is True)
            values[10] = float(statement.reduced)
            values[11] = float(source in causal[side])
            add(identity, values)
            source_by_node[identity] = source
        for node, value in graph.values.items():
            values = _base_features("value", side)
            values[8] = float(value.get("observed", False))
            values[9] = float(value.get("observed_value") is True)
            values[12] = min(1.0, len(value.get("stack_values", ())) / 8.0)
            values[13] = float(bool(value.get("origin")))
            add(f"{side}:v:{node}", values)
        for node, factor in graph.factors.items():
            values = _base_features("factor", side)
            values[12] = min(1.0, len(factor.get("operators", ())) / 8.0)
            values[13] = min(1.0, len(factor.get("tensor", ())) / 16.0)
            add(f"{side}:f:{node}", values)
        for factor, node, _role in graph.incidences:
            left = f"{side}:f:{factor}"
            prefix = "s" if node in graph.statements else "v"
            right = f"{side}:{prefix}:{node}"
            if left in index and right in index:
                connect(left, right, f"{side.lower()}_incidence")

    groups_a = _groups(passing, passing_feature_map)
    groups_b = _groups(failing, failing_feature_map)
    for source in sorted(set(groups_a) & set(groups_b)):
        for node_a, node_b in zip(groups_a[source], groups_b[source]):
            left, right = f"A:s:{node_a}", f"B:s:{node_b}"
            connect(left, right, "aligned_statement")
            features[index[left]][14] = 1.0
            features[index[right]][14] = 1.0
            aligned.append((left, right, source))

    for position, delta in enumerate(feature_delta):
        name = str(delta["feature"])
        node_id = f"feature:{position}:{name}"
        values = _base_features("feature", "shared")
        values[15] = float(delta.get("a", 0))
        values[16] = float(delta.get("b", 0))
        add(node_id, values)
        prefix = f"{name}."
        for statement, source in source_by_node.items():
            if source.startswith(prefix) or source.startswith(f"{name}:"):
                connect(node_id, statement, "feature_source")

    test_node = "test:pair"
    values = _base_features("test", "shared")
    values[12] = float(test_similarity)
    values[13] = float(oracle_similarity)
    values[15] = float(test_provenance == "exact")
    values[16] = float(test_provenance == "generated")
    values[17] = min(1.0, len(aligned) / max(1, max(len(groups_a), len(groups_b))))
    add(test_node, values)
    for statement in source_by_node:
        if features[index[statement]][11] > 0:
            connect(test_node, statement, "test_context")

    edge_array = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    result = PairedExecutionGraph(
        tuple(ids), np.stack(features), edge_array, np.asarray(edge_types, dtype=np.int64),
        RELATIONS, tuple(aligned), float(test_similarity), float(oracle_similarity),
    )
    result.validate()
    return result
