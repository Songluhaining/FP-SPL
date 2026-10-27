from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from fpspl.pipeline import process_version


def coverage_xml(counts: tuple[int, int, int], true_count: int = 0) -> str:
    return f'''<?xml version="1.0"?>
<coverage><project><package name="main"><file path="Account.java" name="Account.java">
<line num="1" count="{counts[0]}" type="method" />
<line num="2" count="{counts[1]}" type="stmt" />
<line num="3" count="{counts[2]}" type="stmt" />
<line num="2" truecount="{true_count}" falsecount="0" type="cond" />
</file></package></project></coverage>'''


def spectrum_xml() -> str:
    return '''<?xml version="1.0"?>
<coverage><project><package name="main"><file path="Account.java" name="Account.java">
<line num="1" count="1" type="method" featureClass="BankAccount.Account" featureLineNum="1" />
<line num="2" count="1" type="stmt" featureClass="BankAccount.Account" featureLineNum="2" />
<line num="2" truecount="1" type="cond" featureClass="BankAccount.Account" featureLineNum="2" />
<line num="3" count="1" type="stmt" featureClass="BankAccount.Account" featureLineNum="3" />
</file></package></project></coverage>'''


class PipelineTests(unittest.TestCase):
    def test_failed_template_and_passing_graph_use_only_observed_test_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            version = root / "bug_1"
            version.mkdir()
            with (version / "config.report.csv.done").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerows([
                    ["Product\\Feature", "BankAccount", "__TEST_OUTPUT__"],
                    ["failing", "T", "__FAILED__"],
                    ["passing", "T", "__PASSED__"],
                ])
            (version / "labels.csv").write_text("variant,ground_truth\npassing,FP\n", encoding="utf-8")
            for variant, outcome in (("failing", "failed"), ("passing", "passed")):
                variant_dir = version / "variants" / variant
                (variant_dir / "coverage" / outcome).mkdir(parents=True)
                (variant_dir / "src").mkdir()
                (variant_dir / "test" / "main").mkdir(parents=True)
                (variant_dir / "src" / "Account.java").write_text(
                    "int run() {\nif (balance > 0) {\nreturn balance;\n", encoding="utf-8"
                )
                (variant_dir / "test" / "main" / "Account_ESTest.java").write_text(
                    "public void test0() {\nassertEquals(1, actual);\n}\n", encoding="utf-8"
                )
                (variant_dir / "coverage" / f"spectrum_{outcome}_coverage.xml").write_text(
                    spectrum_xml(), encoding="utf-8"
                )
                (variant_dir / "coverage" / outcome / "main.Account_ESTest.test0.coverage.xml").write_text(
                    coverage_xml((1, 1, 1), 1), encoding="utf-8"
                )
                (variant_dir / "slice.all_statements.backward_slicing.log").write_text(
                    json.dumps({
                        "BankAccount.Account.1": ["BankAccount.Account.1"],
                        "BankAccount.Account.2": ["BankAccount.Account.1", "BankAccount.Account.2"],
                        "BankAccount.Account.3": ["BankAccount.Account.1", "BankAccount.Account.2", "BankAccount.Account.3"],
                    }), encoding="utf-8"
                )
            summary, rows = process_version(version, root / "out")
            self.assertEqual(summary["templates"], 1)
            self.assertEqual(summary["ranked_variants"], 1)
            self.assertEqual(rows[0]["variant"], "passing")
            self.assertEqual(rows[0]["matched_links"], 3)
            self.assertEqual(rows[0]["evidence"], "coverage+slice-reachability")
            templates = json.loads((root / "out" / "failure_templates.json").read_text(encoding="utf-8"))
            self.assertIn("BankAccount", templates[0]["features"])
            self.assertIn("BankAccount.Account.2:T", templates[0]["branches"])
            self.assertIn("assertEquals(1, actual);", templates[0]["observations"])
            # Removing a separate ground-truth file cannot change training or ranking.
            (version / "labels.csv").unlink()
            _, second_rows = process_version(version, root / "out2")
            self.assertEqual(rows, second_rows)
            for variant in ("failing", "passing"):
                (version / "variants" / variant / "slice.all_statements.backward_slicing.log").unlink()
            weaker_summary, weaker_rows = process_version(version, root / "out3", allow_coverage_only=True)
            self.assertEqual(weaker_summary["ranked_variants"], 1)
            self.assertEqual(weaker_rows[0]["matched_links"], 0)
            self.assertEqual(weaker_rows[0]["evidence"], "coverage-only")

    def test_missing_slice_requires_explicit_coverage_only_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            version = Path(temp) / "bug_1"
            version.mkdir()
            (version / "config.report.csv.done").write_text(
                "Product\\Feature,__TEST_OUTPUT__\npassing,__PASSED__\n", encoding="utf-8"
            )
            (version / "variants" / "passing").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                process_version(version, Path(temp) / "out")
            summary, rows = process_version(version, Path(temp) / "out", allow_coverage_only=True)
            self.assertEqual(summary["insufficient_evidence_variants"], 1)
            self.assertEqual(rows[0]["status"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
