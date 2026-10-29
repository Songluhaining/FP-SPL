from __future__ import annotations

from dataclasses import dataclass

from .domain import FactorGraph, StatementPosterior
from .feature_sources import FeatureMap
from .hierarchical_graph import (
    HierarchicalCounterfactualGraph,
    build_hierarchical_counterfactual_graph,
)
from .spectrum import SpectrumEvidence


@dataclass(frozen=True)
class LearnedCounterfactualGraph:
    hierarchy: HierarchicalCounterfactualGraph
    statement_error_priors: dict[str, float]
    false_passing_node: str | None
    test_equivalence_node: str
    masking_node: str

    @property
    def graph(self) -> FactorGraph:
        return self.hierarchy.graph

    @property
    def false_passing_probability(self) -> float:
        if self.false_passing_node is None:
            return 0.0
        return float(self.graph.values[self.false_passing_node].get("correctness", 0.5))


def _conditional_a_tensor(manifestation: float, noise: float = 0.01,
                          masking_success: float = 0.95) -> list[float]:
    """P(A局部正确性 | 源码正确性, 测试等价, 屏蔽发生)。"""
    values = []
    # 因子张量按Fortran顺序展开，第一个变量变化最快。
    for masking in (False, True):
        for equivalent in (False, True):
            for local_correct in (False, True):
                for source_correct in (False, True):
                    if source_correct:
                        probability_correct = 1.0 - noise
                    elif not equivalent:
                        probability_correct = 0.5
                    elif masking:
                        probability_correct = masking_success
                    else:
                        probability_correct = 1.0 - manifestation
                    values.append(probability_correct if local_correct
                                  else 1.0 - probability_correct)
    return values


def _event_tensor() -> list[float]:
    values = []
    for event in (False, True):
        for masking in (False, True):
            for equivalent in (False, True):
                for source_correct in (False, True):
                    expected = (not source_correct) and equivalent and masking
                    values.append(float(event == expected))
    return values


def _or_tensor() -> list[float]:
    values = []
    for next_value in (False, True):
        for event in (False, True):
            for previous in (False, True):
                values.append(float(next_value == (previous or event)))
    return values


def build_learned_counterfactual_graph(
    failing: FactorGraph,
    passing: FactorGraph,
    failing_feature_map: FeatureMap,
    passing_feature_map: FeatureMap,
    spectrum: dict[str, SpectrumEvidence],
    test_equivalence_probability: float,
    masking_probability: float,
    failing_manifestation: float = 0.95,
) -> LearnedCounterfactualGraph:
    """将学习到的等价与屏蔽概率作为统一因子图的势函数。"""
    hierarchy = build_hierarchical_counterfactual_graph(
        failing, passing, failing_feature_map, passing_feature_map,
        failing_manifestation=failing_manifestation, passing_manifestation=0.5,
    )
    graph = hierarchy.graph
    q_node, m_node = "latent:test-equivalence", "latent:masking"
    graph.statements[q_node] = StatementPosterior(q_node, q_node, q_node, 0.5, False, False)
    graph.statements[m_node] = StatementPosterior(m_node, m_node, m_node, 0.5, False, False)

    # 删除固定A侧表现因子，换成同时受测试等价性和屏蔽机制控制的条件因子。
    removed = {factor for factor in graph.factors if factor.startswith("link:A:")}
    for factor in removed:
        del graph.factors[factor]
    graph.incidences = [item for item in graph.incidences if item[0] not in removed]
    for defect, local in hierarchy.global_to_local.items():
        if "A" not in local:
            continue
        factor = f"learned:A:{defect}"
        graph.factors[factor] = {
            "statement": hierarchy.global_source[defect],
            "operators": ["conditional-manifestation"],
            "tensor": _conditional_a_tensor(failing_manifestation),
        }
        graph.incidences.extend([
            (factor, defect, "DEFECT"),
            (factor, local["A"], "MANIFESTATION"),
            (factor, q_node, "EQUIVALENCE"),
            (factor, m_node, "MASKING"),
        ])

    candidates = sorted(
        node for node in hierarchy.shared_defects
        if hierarchy.global_source[node] in hierarchy.failing_causal_sources
    )
    previous = "latent:false-passing:0"
    graph.values[previous] = {
        "name": previous, "origin": "", "stack_values": [], "correctness": 0.0,
        "observed": True, "observed_value": False,
    }
    for index, defect in enumerate(candidates, 1):
        event = f"latent:event:{index}"
        current = f"latent:false-passing:{index}"
        graph.values[event] = {
            "name": event, "origin": "", "stack_values": [], "correctness": 0.5,
            "observed": False, "observed_value": None,
        }
        graph.values[current] = {
            "name": current, "origin": "", "stack_values": [], "correctness": 0.5,
            "observed": False, "observed_value": None,
        }
        event_factor = f"learned:event:{index}"
        graph.factors[event_factor] = {"statement": "", "operators": ["and"],
                                       "tensor": _event_tensor()}
        graph.incidences.extend([
            (event_factor, defect, "DEFECT"),
            (event_factor, q_node, "EQUIVALENCE"),
            (event_factor, m_node, "MASKING"),
            (event_factor, event, "EVENT"),
        ])
        or_factor = f"learned:or:{index}"
        graph.factors[or_factor] = {"statement": "", "operators": ["or"],
                                    "tensor": _or_tensor()}
        graph.incidences.extend([
            (or_factor, previous, "PREVIOUS"),
            (or_factor, event, "EVENT"),
            (or_factor, current, "NEXT"),
        ])
        previous = current

    priors = {node: 0.5 for node in hierarchy.local_statement_nodes}
    priors.update({
        node: spectrum[source].error_prior if source in spectrum else 0.01
        for node, source in hierarchy.global_source.items()
    })
    priors[q_node] = 1.0 - min(0.999, max(0.001, test_equivalence_probability))
    priors[m_node] = 1.0 - min(0.999, max(0.001, masking_probability))
    return LearnedCounterfactualGraph(
        hierarchy, priors, previous if candidates else None, q_node, m_node,
    )
