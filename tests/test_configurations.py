from pathlib import Path

from cfp_spl.configurations import euclidean_distance, nearest_failing
from cfp_spl.domain import Configuration


def config(name, bits, outcome):
    return Configuration(name, tuple(bits), ("A", "B", "C"), outcome, Path(name))


def test_binary_euclidean_is_sqrt_hamming_and_ties_are_deterministic():
    passing = config("P", [1, 0, 1], "passed")
    f2 = config("F2", [0, 0, 1], "failed")
    f1 = config("F1", [1, 1, 1], "failed")
    assert euclidean_distance(passing, f1) == 1.0
    assert [x.name for x, _ in nearest_failing(passing, [passing, f2, f1], 2)] == ["F1", "F2"]
    farther = config("F3", [0, 1, 1], "failed")
    assert [x.name for x, _ in nearest_failing(passing, [farther, f2, f1])] == ["F1", "F2"]
