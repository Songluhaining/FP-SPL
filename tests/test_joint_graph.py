from cfp_spl.domain import FactorGraph, StatementPosterior
from cfp_spl.joint_graph import build_joint_counterfactual_graph
from cfp_spl.tensor_inference import infer_factor_graph


def execution_graph(observed_value: bool) -> FactorGraph:
    graph = FactorGraph()
    graph.statements["s0"] = StatementPosterior(
        "s0", "main.Account:m:()V#7#0", "main.Account:m:()V#7#0",
        0.5, False, False,
    )
    graph.values["v0"] = {
        "name": "v0", "origin": "", "stack_values": [], "correctness": 0.5,
        "observed": True, "observed_value": observed_value,
    }
    graph.factors["f0"] = {
        "statement": "", "operators": [], "tensor": [1.0, 0.0, 0.0, 1.0],
    }
    graph.incidences = [("f0", "s0", "STMT"), ("f0", "v0", "DEF")]
    return graph


def test_joint_graph_shares_statement_but_keeps_execution_values_separate():
    joint = build_joint_counterfactual_graph(
        execution_graph(False), execution_graph(True), {}, {},
    )
    assert len(joint.graph.statements) == 1
    assert len(joint.shared_statements) == 1
    assert set(joint.graph.values) == {"A:v:v0", "B:v:v0"}
    shared = next(iter(joint.shared_statements))
    attached_factors = {
        factor for factor, node, role in joint.graph.incidences
        if node == shared and role == "STMT"
    }
    assert attached_factors == {"A:f:f0", "B:f:f0"}
    diagnostics = infer_factor_graph(joint.graph, {}, {})
    assert diagnostics.exact


def test_unmatched_statement_is_not_falsely_shared():
    passing = execution_graph(True)
    passing.statements["s0"] = StatementPosterior(
        "s0", "main.Account:m:()V#8#0", "main.Account:m:()V#8#0",
        0.5, False, False,
    )
    joint = build_joint_counterfactual_graph(execution_graph(False), passing, {}, {})
    assert not joint.shared_statements
    assert {side for sides in joint.statement_sides.values() for side in sides} == {"A", "B"}
