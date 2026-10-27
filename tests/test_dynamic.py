from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fpspl.coverage import FeatureLocationMapper
from fpspl.dynamic import read_dynamic_graph
from fpspl.pu_model import graph_tokens


class DynamicGraphTests(unittest.TestCase):
    def test_runtime_edges_are_typed_and_assertion_cone_excludes_unrelated_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            variant = root / "v1"
            (variant / "src").mkdir(parents=True)
            (variant / "test" / "main").mkdir(parents=True)
            (variant / "coverage" / "failed").mkdir(parents=True)
            (variant / "src" / "Account.java").write_text(
                "class Account {\nint value;\nint read(){return value;}\nint unused(){return 0;}\n}", encoding="utf-8")
            (variant / "test" / "main" / "Account_ESTest.java").write_text(
                "class T {\nvoid test0(){\nassertEquals(1, actual);\n}\n}", encoding="utf-8")
            spectrum = ("<coverage><package name='main'><file path='Account.java'>"
                        "<line num='3' type='stmt' featureClass='BankAccount.Account' featureLineNum='3'/>"
                        "<line num='4' type='stmt' featureClass='BankAccount.Account' featureLineNum='4'/>"
                        "</file></package></coverage>")
            (variant / "coverage" / "spectrum_failed_coverage.xml").write_text(spectrum, encoding="utf-8")
            coverage = variant / "coverage" / "failed" / "main.Account_ESTest.test0.coverage.xml"
            coverage.write_text("<coverage/>", encoding="utf-8")
            tsv = root / "graph.tsv"
            tsv.write_text(
                "META\tBRFL_RUNTIME_FACTOR_V1\tmain::Account_test0\tFAIL\n"
                "NODE\t0\tmain.Account:read:()I#3#1\n"
                "NODE\t1\tmain.Account:unused:()I#4#1\n"
                "NODE\t2\tmain.Account_ESTest:test0:()V#3#4\n"
                "EDGE\t2\tmain.Account:read:()I#3#1\tmain.Account_ESTest:test0:()V#3#4\tDATA\n"
                "SUMMARY\t3\t1\t0\n", encoding="utf-8")
            graph = read_dynamic_graph("bug", variant, coverage, "failed", tsv,
                                       {"failure_actual": "0"}, FeatureLocationMapper(variant))
            self.assertEqual(graph.event_count, 3)
            self.assertEqual(graph.features, {"BankAccount", "TEST"})
            self.assertIn(("BankAccount.Account.3", "TEST:main.Account_ESTest:3", "DATA"), graph.typed_links)
            self.assertNotIn("BankAccount.Account.4", graph.nodes)
            self.assertFalse(any("failure_actual" in token for token in graph_tokens(graph)))


if __name__ == "__main__":
    unittest.main()
