"""Weakly supervised positive-versus-unlabeled graph ranking model."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from fpspl.model import TestGraph


def _code_terms(code: str) -> set[str]:
    return {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9_]+", code)
            if term not in {"public", "private", "static", "return", "void"}}


def graph_tokens(graph: TestGraph) -> dict[str, float]:
    """Represent actual runtime edges, branch outcomes and source context."""
    tokens: dict[str, float] = {}
    def add(token: str, weight: float = 1.0) -> None:
        tokens[token] = max(tokens.get(token, 0.0), weight)
    for node in graph.nodes:
        info = graph.node_info.get(node)
        if not info or info.feature == "TEST":
            continue
        add("node:" + node)
        add("feature:" + info.feature, 0.4)
        for term in _code_terms(info.code):
            add("code:" + info.feature + ":" + term, 0.25)
    for source, target, kind in graph.typed_links:
        add("edge:" + kind + ":" + source + ">" + target, 1.5)
        a = graph.node_info.get(source)
        b = graph.node_info.get(target)
        if a and b and a.feature != "TEST" and b.feature != "TEST":
            add("feature-edge:" + kind + ":" + a.feature + ">" + b.feature, 0.6)
    for branch in graph.branches:
        add("branch:" + branch, 0.8)
    for node, values in graph.runtime_values.items():
        info = graph.node_info.get(node)
        if not info or info.feature == "TEST":
            continue
        for value in values:
            kind, _, raw = value.replace("|", ":", 1).partition(":")
            if kind not in {"I", "S", "J", "F", "D", "Z"}:
                continue
            try:
                number = float(raw)
            except ValueError:
                continue
            state = "zero" if number == 0 else "negative" if number < 0 else "positive"
            magnitude = int(math.log10(abs(number) + 1))
            add(f"value:{node}:{kind}:{state}:{min(magnitude, 8)}", 0.3)
    return tokens


def _f1(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def graph_similarity(left: TestGraph, right: TestGraph) -> tuple[float, dict[str, float]]:
    components = {
        "typed_edges": _f1(left.typed_links, right.typed_links),
        "nodes": _f1(left.nodes - {n for n in left.nodes if left.node_info.get(n) and left.node_info[n].feature == "TEST"},
                     right.nodes - {n for n in right.nodes if right.node_info.get(n) and right.node_info[n].feature == "TEST"}),
        "branches": _f1(left.branches, right.branches),
        "observations": _f1(set(left.observations), set(right.observations)),
    }
    score = (0.40 * components["typed_edges"] + 0.30 * components["nodes"]
             + 0.20 * components["branches"] + 0.10 * components["observations"])
    return score, components


@dataclass
class TrainedPUModel:
    vectorizer: object
    classifiers: list[object]
    sampled_unlabeled: list[set[int]]
    positives: int
    unlabeled: int

    def score(self, graph: TestGraph, unlabeled_index: int | None = None) -> float:
        matrix = self.vectorizer.transform([graph_tokens(graph)])
        eligible = [model for model, sampled in zip(self.classifiers, self.sampled_unlabeled)
                    if unlabeled_index is None or unlabeled_index not in sampled]
        if not eligible:
            eligible = self.classifiers
        return sum(float(model.predict_proba(matrix)[0, 1]) for model in eligible) / len(eligible)


def train_pu_model(positives: list[TestGraph], unlabeled: list[TestGraph], seed: int = 17) -> TrainedPUModel | None:
    """Fit repeated biased PU discriminators; unlabeled is a noisy background, not ground truth."""
    if len(positives) < 2 or len(unlabeled) < 2:
        return None
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = DictVectorizer(sparse=True)
    all_graphs = positives + unlabeled
    matrix = vectorizer.fit_transform([graph_tokens(graph) for graph in all_graphs])
    if matrix.shape[1] == 0:
        return None
    rng = np.random.default_rng(seed)
    classifiers = []
    sampled_unlabeled = []
    positive_count = len(positives)
    for _ in range(8):
        take = min(len(unlabeled), max(2, positive_count))
        sampled = rng.choice(len(unlabeled), size=take, replace=len(unlabeled) < take)
        rows = np.r_[np.arange(positive_count), positive_count + sampled]
        labels = np.r_[np.ones(positive_count), np.zeros(take)]
        # Background examples receive low weight because they may contain false passes.
        sample_weight = np.r_[np.ones(positive_count), np.full(take, 0.5 * positive_count / take)]
        classifier = LogisticRegression(C=0.5, max_iter=500, random_state=seed)
        classifier.fit(matrix[rows], labels, sample_weight=sample_weight)
        classifiers.append(classifier)
        sampled_unlabeled.append(set(int(index) for index in sampled))
    return TrainedPUModel(vectorizer, classifiers, sampled_unlabeled, len(positives), len(unlabeled))
