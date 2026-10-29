from cfp_spl.learned_factor_graph import build_learned_counterfactual_graph
from cfp_spl.tensor_inference import infer_factor_graph

from test_joint_graph import execution_graph


def probability(equivalence: float, masking: float) -> float:
    learned = build_learned_counterfactual_graph(
        execution_graph(False), execution_graph(True), {}, {}, {}, equivalence, masking,
    )
    diagnostics = infer_factor_graph(
        learned.graph, {}, {}, statement_error_priors=learned.statement_error_priors,
    )
    assert diagnostics.exact
    return learned.false_passing_probability


def test_false_passing_requires_equivalent_test_and_masking():
    strong = probability(0.95, 0.95)
    unrelated = probability(0.05, 0.95)
    unmasked = probability(0.95, 0.05)
    assert strong > unrelated
    assert strong > unmasked
    assert 0.0 < strong < 1.0
