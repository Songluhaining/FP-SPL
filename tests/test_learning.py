import pytest

torch = pytest.importorskip("torch")

from cfp_spl.learning import CounterfactualGraphNetwork, TestEquivalenceNetwork
from cfp_spl.paired_graph import build_paired_execution_graph

from test_joint_graph import execution_graph


def test_learning_models_produce_finite_logits():
    equivalence = TestEquivalenceNetwork()
    logits = equivalence(torch.zeros(3, 10))
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()

    graph = build_paired_execution_graph(
        execution_graph(False), execution_graph(True), {}, {}, [], 0.8, 1.0, "similar",
    )
    model = CounterfactualGraphNetwork(relations=len(graph.relation_names))
    output = model(graph, "cpu")
    assert torch.isfinite(output.test_equivalence_logit)
    assert torch.isfinite(output.masking_logit)
    assert torch.isfinite(output.defect_logit)
    assert torch.isfinite(output.pair_false_passing_logit)
