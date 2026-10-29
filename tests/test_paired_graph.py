import numpy as np

from cfp_spl.paired_graph import build_paired_execution_graph

from test_joint_graph import execution_graph


def test_paired_graph_keeps_sides_and_adds_alignment_feature_and_test_nodes():
    graph = build_paired_execution_graph(
        execution_graph(False), execution_graph(True), {}, {},
        [{"feature": "Account", "a": 1, "b": 0}],
        0.8, 1.0, "similar",
    )
    graph.validate()
    assert graph.node_features.dtype == np.float32
    assert len(graph.aligned_statements) == 1
    assert "A:s:s0" in graph.node_ids
    assert "B:s:s0" in graph.node_ids
    assert "test:pair" in graph.node_ids
    assert any(name.startswith("feature:") for name in graph.node_ids)
    assert graph.edges.shape[1] == graph.edge_types.size
