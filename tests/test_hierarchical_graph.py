from cfp_spl.hierarchical_graph import build_hierarchical_counterfactual_graph
from cfp_spl.tensor_inference import infer_factor_graph

from test_joint_graph import execution_graph


def _defect_posterior(passing_manifestation: float) -> float:
    hierarchy = build_hierarchical_counterfactual_graph(
        execution_graph(False), execution_graph(True), {}, {},
        failing_manifestation=0.95,
        passing_manifestation=passing_manifestation,
    )
    priors = {node: 0.2 for node in hierarchy.shared_defects}
    priors.update({node: 0.5 for node in hierarchy.local_statement_nodes})
    diagnostics = infer_factor_graph(
        hierarchy.graph, {}, {}, statement_error_priors=priors,
    )
    assert diagnostics.exact
    defect = next(iter(hierarchy.shared_defects))
    return hierarchy.graph.statements[defect].suspiciousness


def test_hierarchy_keeps_local_manifestations_separate():
    hierarchy = build_hierarchical_counterfactual_graph(
        execution_graph(False), execution_graph(True), {}, {},
    )
    assert len(hierarchy.shared_defects) == 1
    defect = next(iter(hierarchy.shared_defects))
    assert set(hierarchy.global_to_local[defect]) == {"A", "B"}
    assert len(hierarchy.local_statement_nodes) == 2


def test_masking_prevents_passing_execution_from_erasing_defect_evidence():
    masked = _defect_posterior(0.2)
    usually_manifested = _defect_posterior(0.8)
    assert masked > usually_manifested
    assert masked > 0.5
