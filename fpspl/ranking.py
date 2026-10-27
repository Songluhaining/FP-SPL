"""Transparent positive-versus-unlabeled graph-template ranking.

This is an initial evidence ranker, not a calibrated PU probability model.
Unlabeled graphs are used to estimate background frequency, never assigned
negative ground-truth labels.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from fpspl.model import FailureTemplate, TestGraph


def _tokens(graph: TestGraph | FailureTemplate, group: str) -> set[str]:
    if group == "nodes":
        return set(graph.nodes)
    if group == "links":
        return {f"{source}>{target}" for source, target in graph.links}
    if group == "branches":
        return set(graph.branches)
    if group == "features":
        return set(graph.features)
    if group == "semantics":
        return {
            f"{info.feature}:{' '.join(info.code.split())}"
            for node in graph.nodes
            if (info := graph.node_info.get(node)) is not None and info.code
        }
    raise ValueError(group)


@dataclass
class ContrastWeights:
    weights: dict[str, dict[str, float]]
    positive_graphs: int
    unlabeled_graphs: int

    @classmethod
    def fit(cls, positives: list[TestGraph], unlabeled: list[TestGraph]) -> "ContrastWeights":
        groups = ("nodes", "links", "branches", "features", "semantics")
        weights: dict[str, dict[str, float]] = {}
        for group in groups:
            positive_frequency: Counter[str] = Counter()
            unlabeled_frequency: Counter[str] = Counter()
            for graph in positives:
                positive_frequency.update(_tokens(graph, group))
            for graph in unlabeled:
                unlabeled_frequency.update(_tokens(graph, group))
            group_weights: dict[str, float] = {}
            for token in positive_frequency.keys() | unlabeled_frequency.keys():
                p = (positive_frequency[token] + 1) / (len(positives) + 2)
                u = (unlabeled_frequency[token] + 1) / (len(unlabeled) + 2)
                # The floor retains common dependencies; the cap avoids one
                # rare token dominating a small buggy version.
                group_weights[token] = min(3.0, max(0.25, 1.0 + math.log(p / u)))
            weights[group] = group_weights
        return cls(weights, len(positives), len(unlabeled))

    def get(self, group: str, token: str) -> float:
        return self.weights[group].get(token, 1.0)


def _weighted_f1(a: set[str], b: set[str], group: str, weights: ContrastWeights) -> float | None:
    if not a or not b:
        return None
    shared = a & b
    shared_weight = sum(weights.get(group, token) for token in shared)
    if shared_weight == 0:
        return 0.0
    a_weight = sum(weights.get(group, token) for token in a)
    b_weight = sum(weights.get(group, token) for token in b)
    return 2.0 * shared_weight / (a_weight + b_weight)


def _observation_similarity(template: FailureTemplate, graph: TestGraph) -> float | None:
    if not template.observations or not graph.observations:
        return None
    # Generated method names can refer to different tests across variants.
    # Compare assertion source instead of assuming matching names imply matching behavior.
    left, right = set(template.observations), set(graph.observations)
    return len(left & right) / len(left | right) if left | right else 0.0


def compare_graphs(
    template: FailureTemplate, graph: TestGraph, weights: ContrastWeights
) -> tuple[float, dict[str, float | None]]:
    components: dict[str, float | None] = {
        group: _weighted_f1(_tokens(template, group), _tokens(graph, group), group, weights)
        for group in ("nodes", "links", "branches", "features", "semantics")
    }
    components["observations"] = _observation_similarity(template, graph)
    component_weights = {"nodes": 0.30, "links": 0.25, "branches": 0.20, "features": 0.10, "semantics": 0.10, "observations": 0.05}
    # Missing evidence contributes zero rather than granting a tiny graph a
    # perfect score by renormalizing over its only available component.
    score = sum(weight * (components[name] or 0.0) for name, weight in component_weights.items())
    return score, components


def rank_variants(
    templates: list[FailureTemplate],
    passing_by_variant: dict[str, list[TestGraph]],
    weights: ContrastWeights,
) -> list[dict]:
    rows: list[dict] = []
    for variant, graphs in passing_by_variant.items():
        comparisons: list[tuple[float, FailureTemplate, TestGraph, dict[str, float | None]]] = []
        for template in templates:
            for graph in graphs:
                score, components = compare_graphs(template, graph, weights)
                comparisons.append((score, template, graph, components))
        if not comparisons:
            rows.append({"variant": variant, "rank": None, "score": None, "status": "insufficient_evidence",
                         "best_failure_variant": "", "best_failure_test": "", "best_passing_test": "",
                         "matched_nodes": 0, "matched_links": 0, "components": {}, "evidence": ""})
            continue
        comparisons.sort(key=lambda item: (-item[0], item[1].source_variant, item[1].source_test, item[2].test))
        best, template, graph, components = comparisons[0]
        rows.append({
            "variant": variant,
            "rank": None,
            "score": round(best, 6),
            "status": "ranked",
            "best_failure_variant": template.source_variant,
            "best_failure_test": template.source_test,
            "best_passing_test": graph.test,
            "matched_nodes": len(template.nodes & graph.nodes),
            "matched_links": len(template.links & graph.links),
            "components": components,
            "evidence": graph.evidence,
        })
    ranked = sorted((row for row in rows if row["status"] == "ranked"), key=lambda row: (-row["score"], row["variant"]))
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
    return ranked + sorted((row for row in rows if row["status"] != "ranked"), key=lambda row: row["variant"])
