from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .domain import JavaTestCase
from .java_tests import CALL, canonical_tokens, oracle_signature


NUMBER = re.compile(r"(?<![\w$])[-+]?\d+(?:\.\d+)?")


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _ratio(left: int, right: int) -> float:
    return min(left, right) / max(1, max(left, right))


def test_pair_vector(left: JavaTestCase, right: JavaTestCase) -> np.ndarray:
    """不读取真假通过标签的测试行为静态表示。"""
    tokens_left, tokens_right = canonical_tokens(left.body), canonical_tokens(right.body)
    calls_left, calls_right = CALL.findall(left.body), CALL.findall(right.body)
    oracle_left, oracle_right = oracle_signature(left), oracle_signature(right)
    numbers_left, numbers_right = set(NUMBER.findall(left.body)), set(NUMBER.findall(right.body))
    class_left, class_right = left.class_name.rsplit(".", 1)[-1], right.class_name.rsplit(".", 1)[-1]
    return np.asarray([
        _jaccard(set(tokens_left), set(tokens_right)),
        _jaccard(set(calls_left), set(calls_right)),
        _jaccard(oracle_left, oracle_right),
        _jaccard(numbers_left, numbers_right),
        float(class_left == class_right),
        float(left.method_name == right.method_name),
        _ratio(len(tokens_left), len(tokens_right)),
        _ratio(len(calls_left), len(calls_right)),
        _ratio(len(oracle_left), len(oracle_right)),
        float(tokens_left == tokens_right),
    ], dtype=np.float32)


@dataclass(frozen=True)
class EquivalenceExample:
    features: np.ndarray
    label: int
    version: str
    left: str
    right: str


def make_weak_examples(version: str, tests: list[JavaTestCase]) -> list[EquivalenceExample]:
    """完全相同内容作可靠正例，不同判定目标作可靠负例。"""
    examples: list[EquivalenceExample] = []
    by_tokens: dict[tuple[str, ...], list[JavaTestCase]] = {}
    for test in tests:
        by_tokens.setdefault(canonical_tokens(test.body), []).append(test)
    for group in by_tokens.values():
        if len(group) >= 2:
            left, right = group[0], group[-1]
            if left.source_path != right.source_path:
                examples.append(EquivalenceExample(
                    test_pair_vector(left, right), 1, version, left.identity, right.identity,
                ))

    unique = sorted(tests, key=lambda item: (item.identity, str(item.source_path)))
    negative_target = len(examples)
    for i, left in enumerate(unique):
        if len(examples) >= 2 * negative_target:
            break
        for right in unique[i + 1:]:
            if left.source_path == right.source_path:
                continue
            if oracle_signature(left) & oracle_signature(right):
                continue
            examples.append(EquivalenceExample(
                test_pair_vector(left, right), 0, version, left.identity, right.identity,
            ))
            break
    return examples


def make_hard_examples(version: str, tests: list[JavaTestCase]) -> list[EquivalenceExample]:
    """构造内容不完全相同的近等价正例和词法相近但判定不同的困难负例。"""
    unique_by_body: dict[tuple[str, ...], JavaTestCase] = {}
    for test in tests:
        unique_by_body.setdefault(canonical_tokens(test.body), test)
    unique = sorted(unique_by_body.values(), key=lambda item: (item.identity, str(item.source_path)))
    positives: list[EquivalenceExample] = []
    negatives: list[EquivalenceExample] = []
    for position, left in enumerate(unique):
        calls_left = set(CALL.findall(left.body))
        oracle_left = oracle_signature(left)
        tokens_left = set(canonical_tokens(left.body))
        for right in unique[position + 1:]:
            if left.class_name.rsplit(".", 1)[-1] != right.class_name.rsplit(".", 1)[-1]:
                continue
            calls_right = set(CALL.findall(right.body))
            oracle_right = oracle_signature(right)
            call_overlap = _jaccard(calls_left, calls_right)
            token_overlap = _jaccard(tokens_left, set(canonical_tokens(right.body)))
            if oracle_left and oracle_left == oracle_right and call_overlap >= 0.5:
                positives.append(EquivalenceExample(
                    test_pair_vector(left, right), 1, version, left.identity, right.identity,
                ))
            elif (not oracle_left & oracle_right and call_overlap >= 0.3
                  and token_overlap >= 0.35):
                negatives.append(EquivalenceExample(
                    test_pair_vector(left, right), 0, version, left.identity, right.identity,
                ))
    count = min(len(positives), len(negatives))
    return positives[:count] + negatives[:count]
