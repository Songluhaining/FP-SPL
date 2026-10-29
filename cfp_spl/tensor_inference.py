from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Iterable

import numpy as np

from .domain import FactorGraph
from .feature_sources import FeatureMap, source_identity
from .spectrum import SpectrumEvidence


@dataclass(frozen=True)
class TensorFactor:
    scope: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class InferenceDiagnostics:
    algorithm: str
    exact: bool
    induced_width: int
    regions: int
    iterations: int
    converged: bool
    max_delta: float
    spectrum_statements: int
    default_prior_statements: int

    def to_dict(self) -> dict:
        return asdict(self)


def _normal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(values.shape, 1.0 / values.size)
    return values / total


def _expand(factor: TensorFactor, target: tuple[str, ...]) -> np.ndarray:
    if not factor.scope:
        return factor.values
    positions = [target.index(name) for name in factor.scope]
    order = sorted(range(len(positions)), key=positions.__getitem__)
    values = factor.values.transpose(order) if order != list(range(len(order))) else factor.values
    sorted_positions = [positions[index] for index in order]
    shape = [1] * len(target)
    for position in sorted_positions:
        shape[position] = 2
    return values.reshape(shape)


def _product(factors: Iterable[TensorFactor], scope: tuple[str, ...]) -> TensorFactor:
    values = np.ones((2,) * len(scope), dtype=float)
    for factor in factors:
        values *= _expand(factor, scope)
    return TensorFactor(scope, values)


def _marginal(factor: TensorFactor, keep: tuple[str, ...]) -> TensorFactor:
    axes = tuple(index for index, name in enumerate(factor.scope) if name not in keep)
    values = factor.values.sum(axis=axes) if axes else factor.values.copy()
    remaining = tuple(name for name in factor.scope if name in keep)
    if remaining != keep and keep:
        values = values.transpose([remaining.index(name) for name in keep])
    return TensorFactor(keep, _normal(values))


def _raw_factors(
    graph: FactorGraph,
    spectrum: dict[str, SpectrumEvidence],
    feature_map: FeatureMap,
    default_error_prior: float,
    statement_error_priors: dict[str, float] | None = None,
) -> tuple[list[TensorFactor], int, int]:
    by_factor: dict[str, list[str]] = {}
    for factor, node, _role in graph.incidences:
        by_factor.setdefault(factor, []).append(node)
    factors: list[TensorFactor] = []
    for identity, data in graph.factors.items():
        scope = tuple(by_factor.get(identity, ()))
        flat = np.asarray(data["tensor"], dtype=float)
        if scope and flat.size == 1 << len(scope):
            factors.append(TensorFactor(scope, flat.reshape((2,) * len(scope), order="F")))

    matched = missing = 0
    for identity, statement in graph.statements.items():
        if statement.observed:
            value = True if statement.observed_value is None else statement.observed_value
            factors.append(TensorFactor((identity,), np.asarray([not value, value], dtype=float)))
            continue
        explicit = (statement_error_priors or {}).get(identity)
        key = source_identity(statement.stable_key, feature_map)
        evidence = spectrum.get(key) if explicit is None else None
        error = explicit if explicit is not None else (
            evidence.error_prior if evidence else default_error_prior
        )
        matched += explicit is not None or evidence is not None
        missing += explicit is None and evidence is None
        factors.append(TensorFactor((identity,), np.asarray([error, 1.0 - error])))
    for identity, value in graph.values.items():
        if value.get("observed"):
            observed_value = value.get("observed_value")
            if observed_value is None:
                observed_value = value.get("correctness", 0.5) >= 0.5
            factors.append(TensorFactor(
                (identity,), np.asarray([not observed_value, observed_value], dtype=float)
            ))
    return factors, matched, missing


def _triangulate(factors: list[TensorFactor]) -> tuple[list[frozenset[str]], int]:
    adjacency: dict[str, set[str]] = {
        name: set() for factor in factors for name in factor.scope
    }
    for factor in factors:
        for left, right in combinations(factor.scope, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
    cliques: list[frozenset[str]] = []
    width = 0
    while adjacency:
        def priority(name: str) -> tuple[int, int, str]:
            neighbors = sorted(adjacency[name])
            fill = sum(
                right not in adjacency[left]
                for index, left in enumerate(neighbors)
                for right in neighbors[index + 1:]
            )
            return fill, len(neighbors), name

        node = min(adjacency, key=priority)
        neighbors = sorted(adjacency[node])
        cliques.append(frozenset((node, *neighbors)))
        width = max(width, len(neighbors))
        for left, right in combinations(neighbors, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
        for neighbor in neighbors:
            adjacency[neighbor].discard(node)
        del adjacency[node]
    maximal = [
        clique for index, clique in enumerate(cliques)
        if not any(clique < other for j, other in enumerate(cliques) if index != j)
    ]
    unique: list[frozenset[str]] = []
    for clique in maximal:
        if clique not in unique:
            unique.append(clique)
    return unique, width


def _junction_forest(cliques: list[frozenset[str]]) -> dict[int, set[int]]:
    parent = list(range(len(cliques)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    edges = sorted(
        ((len(left & right), i, j) for i, left in enumerate(cliques)
         for j, right in enumerate(cliques[i + 1:], i + 1) if left & right),
        reverse=True,
    )
    tree = {index: set() for index in range(len(cliques))}
    for _weight, left, right in edges:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        parent[root_right] = root_left
        tree[left].add(right)
        tree[right].add(left)
    return tree


def _exact_junction_tree(
    factors: list[TensorFactor], cliques: list[frozenset[str]], width: int,
) -> tuple[dict[str, float], InferenceDiagnostics]:
    tree = _junction_forest(cliques)
    clique_scopes = [tuple(sorted(clique)) for clique in cliques]
    assigned: list[list[TensorFactor]] = [[] for _ in cliques]
    for factor in factors:
        candidates = [
            index for index, clique in enumerate(cliques) if set(factor.scope) <= clique
        ]
        if not candidates:
            raise ValueError(f"没有团能够容纳因子：{factor.scope}")
        target = min(candidates, key=lambda index: len(cliques[index]))
        assigned[target].append(factor)
    potentials = []
    for scope, values in zip(clique_scopes, assigned):
        potential = _product(values, scope)
        potentials.append(TensorFactor(scope, _normal(potential.values)))

    messages: dict[tuple[int, int], TensorFactor] = {}

    def send(source: int, target: int) -> None:
        incoming = [messages[(neighbor, source)] for neighbor in tree[source]
                    if neighbor != target and (neighbor, source) in messages]
        combined = _product((potentials[source], *incoming), clique_scopes[source])
        separator = tuple(sorted(cliques[source] & cliques[target]))
        messages[(source, target)] = _marginal(combined, separator)

    visited: set[int] = set()
    for root in range(len(cliques)):
        if root in visited:
            continue
        order, parents = [], {root: -1}
        stack = [root]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            order.append(current)
            for neighbor in tree[current]:
                if neighbor != parents[current]:
                    parents[neighbor] = current
                    stack.append(neighbor)
        for current in reversed(order[1:]):
            send(current, parents[current])
        for current in order:
            for neighbor in tree[current]:
                if parents.get(neighbor) == current:
                    send(current, neighbor)

    marginals: dict[str, float] = {}
    for index, scope in enumerate(clique_scopes):
        incoming = [messages[(neighbor, index)] for neighbor in tree[index]]
        belief = _product((potentials[index], *incoming), scope)
        for variable in scope:
            if variable not in marginals:
                marginals[variable] = float(_marginal(belief, (variable,)).values[1])
    diagnostics = InferenceDiagnostics(
        "exact-junction-tree-tensor", True, width, len(cliques), 2, True, 0.0, 0, 0,
    )
    return marginals, diagnostics


def _merge_regions(factors: list[TensorFactor], limit: int) -> list[list[TensorFactor]]:
    non_unary = [[factor] for factor in factors if len(factor.scope) > 1]
    unary = [factor for factor in factors if len(factor.scope) <= 1]
    if not non_unary:
        return [[factor] for factor in unary]
    while True:
        best = None
        scopes = [set(name for factor in region for name in factor.scope) for region in non_unary]
        for left, right in combinations(range(len(non_unary)), 2):
            overlap = scopes[left] & scopes[right]
            union = scopes[left] | scopes[right]
            if len(overlap) < 2 or len(union) > limit:
                continue
            candidate = (len(overlap), -len(union), -left, -right)
            if best is None or candidate > best[0]:
                best = candidate, left, right
        if best is None:
            break
        _, left, right = best
        non_unary[left].extend(non_unary[right])
        del non_unary[right]
    for factor in unary:
        variable = factor.scope[0] if factor.scope else None
        candidates = [index for index, region in enumerate(non_unary)
                      if variable in {name for item in region for name in item.scope}]
        if candidates:
            non_unary[min(candidates, key=lambda index: len(non_unary[index]))].append(factor)
        else:
            non_unary.append([factor])
    return non_unary


def _bounded_region_messages(
    factors: list[TensorFactor], width: int, limit: int,
    max_iterations: int, damping: float, tolerance: float,
) -> tuple[dict[str, float], InferenceDiagnostics]:
    grouped = _merge_regions(factors, limit)
    scopes = [tuple(sorted({name for factor in region for name in factor.scope})) for region in grouped]
    potentials = [TensorFactor(scope, _normal(_product(region, scope).values))
                  for region, scope in zip(grouped, scopes)]
    by_variable: dict[str, list[int]] = {}
    for index, scope in enumerate(scopes):
        for variable in scope:
            by_variable.setdefault(variable, []).append(index)
    r_to_v = {(index, variable): np.asarray([0.5, 0.5])
              for index, scope in enumerate(scopes) for variable in scope}
    v_to_r = {(variable, index): np.asarray([0.5, 0.5])
              for variable, regions in by_variable.items() for index in regions}
    converged, max_delta = False, float("inf")
    for iteration in range(1, max_iterations + 1):
        new_r_to_v = {}
        max_delta = 0.0
        for index, scope in enumerate(scopes):
            for variable in scope:
                incoming = [TensorFactor((other,), v_to_r[(other, index)])
                            for other in scope if other != variable]
                combined = _product((potentials[index], *incoming), scope)
                proposal = _marginal(combined, (variable,)).values
                updated = _normal(damping * r_to_v[(index, variable)] + (1.0 - damping) * proposal)
                new_r_to_v[(index, variable)] = updated
                max_delta = max(max_delta, float(np.max(np.abs(updated - r_to_v[(index, variable)]))))
        new_v_to_r = {}
        for variable, regions in by_variable.items():
            for index in regions:
                value = np.ones(2)
                for other in regions:
                    if other != index:
                        value *= new_r_to_v[(other, variable)]
                proposal = _normal(value)
                updated = _normal(damping * v_to_r[(variable, index)] + (1.0 - damping) * proposal)
                new_v_to_r[(variable, index)] = updated
                max_delta = max(max_delta, float(np.max(np.abs(updated - v_to_r[(variable, index)]))))
        r_to_v, v_to_r = new_r_to_v, new_v_to_r
        if max_delta < tolerance:
            converged = True
            break
    marginals = {}
    for variable, regions in by_variable.items():
        value = np.ones(2)
        for index in regions:
            value *= r_to_v[(index, variable)]
        marginals[variable] = float(_normal(value)[1])
    diagnostics = InferenceDiagnostics(
        "bounded-region-tensor", False, width, len(grouped), iteration,
        converged, max_delta, 0, 0,
    )
    return marginals, diagnostics


def infer_factor_graph(
    graph: FactorGraph,
    spectrum: dict[str, SpectrumEvidence],
    feature_map: FeatureMap,
    max_region_variables: int = 14,
    default_error_prior: float = 0.01,
    max_iterations: int = 100,
    damping: float = 0.35,
    tolerance: float = 1e-8,
    statement_error_priors: dict[str, float] | None = None,
) -> InferenceDiagnostics:
    """用自适应区域张量推理更新图中语句和值的正确性后验。"""
    factors, matched, missing = _raw_factors(
        graph, spectrum, feature_map, default_error_prior, statement_error_priors,
    )
    cliques, width = _triangulate(factors)
    if width < max_region_variables:
        marginals, diagnostics = _exact_junction_tree(factors, cliques, width)
    else:
        marginals, diagnostics = _bounded_region_messages(
            factors, width, max_region_variables, max_iterations, damping, tolerance,
        )
    for identity, statement in tuple(graph.statements.items()):
        if identity in marginals:
            graph.statements[identity] = replace(statement, correctness=marginals[identity])
    for identity, value in graph.values.items():
        if identity in marginals:
            value["correctness"] = marginals[identity]
    return replace(
        diagnostics,
        spectrum_statements=matched,
        default_prior_statements=missing,
    )
