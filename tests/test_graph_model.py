from __future__ import annotations

import unittest

from fpspl.graph_model import graph_arrays, train_graph_pu_model
from fpspl.model import NodeInfo, TestGraph


def make_graph(variant: str, outcome: str, dependency: str) -> TestGraph:
    graph = TestGraph("version", variant, "test", outcome)
    graph.nodes = {"feature.A.1", "feature.B.2", "TEST:assert:3"}
    graph.node_info = {
        "feature.A.1": NodeInfo("feature.A.1", "Base", "A.java", 1, 1, "stmt", "int a = 1;"),
        "feature.B.2": NodeInfo("feature.B.2", "Optional", "B.java", 2, 2, "stmt", "int b = a;"),
        "TEST:assert:3": NodeInfo("TEST:assert:3", "TEST", "Test.java", 3, 3, "test", "assertEquals(1, b);")}
    graph.typed_links = {("feature.A.1", "feature.B.2", dependency),
                         ("feature.B.2", "TEST:assert:3", "DATA")}
    graph.branches = {"feature.A.1:T"}
    graph.runtime_values = {"feature.B.2": ["I|1"]}
    return graph


class GraphModelTests(unittest.TestCase):
    def test_directed_typed_edges_remain_distinct(self):
        graph = make_graph("v1", "failed", "CONTROL")
        arrays = graph_arrays(graph)
        self.assertEqual(len(arrays[0]), 3)
        self.assertEqual(len(arrays[-1]), 4)
        self.assertIn(1, arrays[-1])
        self.assertIn(3, arrays[-1])

    def test_training_and_out_of_variant_scoring(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed in this Python environment")
        positive = [make_graph("failed1", "failed", "CONTROL"),
                    make_graph("failed2", "failed", "CONTROL")]
        unlabeled = [make_graph("pass1", "passed", "DATA"),
                     make_graph("pass2", "passed", "DATA")]
        model = train_graph_pu_model(positive, unlabeled, ensembles=2, epochs=3)
        scores = model.score_all(unlabeled)
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(0 <= score <= 1 for score in scores))
        self.assertTrue(all(model.sampled_variants))


if __name__ == "__main__":
    unittest.main()
