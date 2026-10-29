import numpy as np

from cfp_spl.domain import FactorGraph, StatementPosterior
from cfp_spl.spectrum import SpectrumEvidence
from cfp_spl.tensor_inference import infer_factor_graph


def _evidence(error_prior: float) -> SpectrumEvidence:
    return SpectrumEvidence(1, 0, 1, 1, 1, 1, 1, 1, 1, error_prior)


def test_exact_tensor_inference_propagates_observed_failure():
    graph = FactorGraph()
    graph.statements["s"] = StatementPosterior(
        "s", "main.Account:m:()V#7#0", "main.Account:m:()V#7#0", 0.5, False, False,
    )
    graph.values["v"] = {
        "name": "v", "origin": "", "stack_values": [], "correctness": 0.5,
        "observed": True, "observed_value": False,
    }
    graph.factors["f"] = {
        "statement": "", "operators": [], "tensor": [1.0, 0.0, 0.0, 1.0],
    }
    graph.incidences.extend((("f", "s", "STMT"), ("f", "v", "DEF")))
    diagnostics = infer_factor_graph(
        graph, {"BankAccount.Account:7": _evidence(0.8)},
        {("main.Account", 7): "BankAccount.Account:7"},
    )
    assert diagnostics.exact
    assert diagnostics.algorithm == "exact-junction-tree-tensor"
    assert graph.statements["s"].correctness == 0.0


def test_tensor_layout_matches_brfl_bit_order():
    graph = FactorGraph()
    graph.statements["s"] = StatementPosterior(
        "s", "main.Account:m:()V#7#0", "main.Account:m:()V#7#0", 0.5, False, False,
    )
    graph.values["v"] = {
        "name": "v", "origin": "", "stack_values": [], "correctness": 0.5,
        "observed": True, "observed_value": True,
    }
    # BRFL下标：bit0=s，bit1=v；只有(s=true,v=true)具有权重。
    graph.factors["f"] = {"statement": "", "operators": [], "tensor": [0, 0, 0, 1]}
    graph.incidences.extend((("f", "s", "STMT"), ("f", "v", "DEF")))
    infer_factor_graph(graph, {}, {})
    assert np.isclose(graph.statements["s"].correctness, 1.0)


def test_exact_region_tensor_matches_bruteforce_on_cycle():
    graph = FactorGraph()
    for name, line in (("a", 7), ("b", 8), ("c", 9)):
        graph.statements[name] = StatementPosterior(
            name, f"main.Account:m:()V#{line}#0", f"main.Account:m:()V#{line}#0",
            0.5, False, False,
        )
    # 三个相等偏好因子形成环，并给a一个偏向错误的一元频谱先验。
    equality = [3.0, 1.0, 1.0, 3.0]
    for factor, left, right in (("f1", "a", "b"), ("f2", "b", "c"), ("f3", "c", "a")):
        graph.factors[factor] = {"statement": "", "operators": [], "tensor": equality}
        graph.incidences.extend(((factor, left, "STMT"), (factor, right, "USE")))
    priors = {"main.Account:m:()V#7#0": _evidence(0.8)}
    diagnostics = infer_factor_graph(graph, priors, {}, default_error_prior=0.2)

    weights = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                prior = (0.8 if a == 0 else 0.2) * (0.2 if b == 0 else 0.8) * (0.2 if c == 0 else 0.8)
                pair = lambda x, y: 3.0 if x == y else 1.0
                weights[(a, b, c)] = prior * pair(a, b) * pair(b, c) * pair(c, a)
    total = sum(weights.values())
    expected = sum(weight for state, weight in weights.items() if state[0] == 1) / total
    assert diagnostics.exact
    assert np.isclose(graph.statements["a"].correctness, expected)
