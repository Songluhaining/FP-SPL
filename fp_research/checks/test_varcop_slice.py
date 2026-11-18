"""手算静态关系重算以及并列排序核验的边界。"""
from pathlib import Path
import tempfile
import unittest

from fp_research.varcop_slice import from_basis
from fp_research.checks.localization_smoke import assert_same_ranking


class SliceChecks(unittest.TestCase):
    def test_intersection_backward_disabled_and_duplicate_combinations(self):
        basis = ({"A": {"x", "y"}, "B": {"y"}, "Empty": set()},
                 {"y": {"z"}}, {"x": "A.1", "y": "B.2", "z": "C.3"},
                 {"A.1", "B.2", "C.3"}, set())
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "spc.log"
            log.write_text("A,B; /v/product/src\nA,B; /v/product/src\nA,#Off; /v/product/src\n"
                           "A,Empty; /v/product/src\n", encoding="utf-8")
            result = from_basis(basis, log, "/v/product/src")
            self.assertEqual(result, {"product": {"A.1": {"num_interactions": 1},
                                                   "B.2": {"num_interactions": 2},
                                                   "C.3": {"num_interactions": 2}}})

    def test_missing_later_feature_preserves_official_break_behavior(self):
        basis = ({"A": {"x"}, "B": set()}, {}, {"x": "A.1"}, {"A.1"}, set())
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "spc.log"
            log.write_text("A,Missing,B; /v/product/src\nMissing,A; /v/product/src\n", encoding="utf-8")
            self.assertEqual(from_basis(basis, log, "/v/product/src"),
                             {"product": {"A.1": {"num_interactions": 1}}})

    def test_tie_permutation_is_equivalent_but_score_change_is_not(self):
        a = [("a", .9, 2), ("b", .9, 2), ("c", .1, 1)]
        assert_same_ranking(a, [a[1], a[0], a[2]])
        with self.assertRaises(AssertionError):
            assert_same_ranking(a, [("a", .8, 2), a[1], a[2]])


if __name__ == "__main__":
    unittest.main()
