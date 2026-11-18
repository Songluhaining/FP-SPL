"""手算并列、筛选回退、缺失故障及只读配置排除。"""
from types import SimpleNamespace
import unittest

from fp_research.localization_evaluate import locate, layered_position
from fp_research.varcop_localization import excluded_products, variability_case


class LocalizationChecks(unittest.TestCase):
    def test_worst_ties_and_isolation_fallback(self):
        full = [("a", .9, 1), ("b", .8, 2), ("c", .8, 2), ("d", .1, 1)]
        focus = [("d", .9, 1)]
        self.assertEqual(locate("b", full), 3)
        self.assertEqual(layered_position("b", full, focus), 4)
        self.assertEqual(layered_position("d", full, focus), 1)
        self.assertIsNone(layered_position("absent", full, focus))

    def test_exclusion_changes_all_product_readers_and_restores_them(self):
        dirs = lambda path: ["/case/variants/pass", "/case/variants/fail"]
        listing = lambda path, *a, **k: ["pass", "fail"]
        ranking = SimpleNamespace(get_all_variant_dirs=dirs, list_dir=listing)
        products = SimpleNamespace(get_all_variant_dirs=dirs)
        with excluded_products(ranking, products, {"pass"}):
            self.assertEqual(ranking.get_all_variant_dirs("/case"), ["/case/variants/fail"])
            self.assertEqual(products.get_all_variant_dirs("/case"), ["/case/variants/fail"])
            self.assertEqual(ranking.list_dir("/case/variants"), ["fail"])
        self.assertIs(ranking.get_all_variant_dirs, dirs)
        self.assertIs(products.get_all_variant_dirs, dirs)
        self.assertIs(ranking.list_dir, listing)

    def test_official_base_failure_branch_uses_visible_configurations(self):
        header = ["variant", "Base", "Optional", "__TEST_OUTPUT__"]
        self.assertFalse(variability_case([["a", "T", "F", "__FAILED__"]], header, "elevator"))
        self.assertTrue(variability_case([["a", "T", "T", "__FAILED__"]], header, "elevator"))


if __name__ == "__main__":
    unittest.main()
