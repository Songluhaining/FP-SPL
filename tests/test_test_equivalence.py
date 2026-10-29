from pathlib import Path

from cfp_spl.domain import JavaTestCase
from cfp_spl.test_equivalence import make_hard_examples, test_pair_vector as pair_vector


def case(method: str, body: str, source: str = "A.java") -> JavaTestCase:
    return JavaTestCase("main.AccountTest", method, Path(source), body, body)


def test_identical_behavior_has_stronger_vector_than_unrelated_test():
    left = case("testA", "account.deposit(1); assertEquals(1, account.balance());")
    equivalent = case("testB", "account.deposit(1); assertEquals(1, account.balance());", "B.java")
    unrelated = case("testC", "account.lock(); assertTrue(account.isLocked());", "C.java")
    same = pair_vector(left, equivalent)
    different = pair_vector(left, unrelated)
    assert same.shape == (10,)
    assert same[0] > different[0]
    assert same[1] > different[1]
    assert same[2] > different[2]


def test_hard_examples_exclude_identical_body_shortcut():
    tests = [
        case("testA", "account.deposit(1); assertEquals(1, account.balance());", "A.java"),
        case("testB", "account.deposit(2); assertEquals(1, account.balance());", "B.java"),
        case("testC", "account.deposit(1); assertTrue(account.isLocked());", "C.java"),
    ]
    examples = make_hard_examples("v1", tests)
    assert examples
    assert {item.label for item in examples} == {0, 1}
    assert all(item.features[9] == 0.0 for item in examples)
