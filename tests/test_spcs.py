"""将等价SPC搜索与原始穷举及手工样例比较。"""
from itertools import product
from pathlib import Path
import random
import tempfile
import unittest

from fp_research.localization.common import upstream
from fp_research.localization.spcs import detect, minimal_hitting_sets


def normalized(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        combination, source = line.split(";", 1)
        rows.append((tuple(sorted(combination.split(","))), source.strip().replace("\\", "/")))
    return set(rows)


class SPCChecks(unittest.TestCase):
    def test_small_switches(self):
        self.assertEqual(set(minimal_hitting_sets([{1, 2}, {2, 3}])),
                         {frozenset({2}), frozenset({1, 3})})
        self.assertEqual(minimal_hitting_sets([set(), {1}]), [])
        self.assertEqual(minimal_hitting_sets([{i} for i in range(8)]), [])

    def test_random_configurations_match_original_exhaustive_search(self):
        _, _, original = upstream()
        rng = random.Random(20261118)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for _ in range(80):
                features = ["F"+str(i) for i in range(rng.randrange(2, 8))]
                configs = list(product([False, True], repeat=len(features)))
                rng.shuffle(configs)
                configs = configs[:rng.randrange(3, min(len(configs), 22)+1)]
                split = rng.randrange(1, len(configs))
                passed, failed = list(map(list, configs[:split])), list(map(list, configs[split:]))
                names = {tuple(c): "model_"+str(i) for i, c in enumerate(configs)}
                left, right = folder / "original.log", folder / "optimized.log"
                original.detect_SPCs(features, passed, failed, names, folder.as_posix(), str(left))
                detect(features, passed, failed, names, folder.as_posix(), right)
                self.assertEqual(normalized(left), normalized(right))
                self.assertEqual(left.read_text(), right.read_text())


if __name__ == "__main__":
    unittest.main()
