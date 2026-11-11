import tempfile
from pathlib import Path
import unittest
from fp_research.slice_reuse import signatures


class SliceInput(unittest.TestCase):
    def test_only_original_tool_inputs_define_equivalence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/"result") as tmp:
            root=Path(tmp);(root/"src").mkdir();(root/"coverage").mkdir()
            source=root/"src/A.java";source.write_text("class A {}")
            report=root/"coverage/spectrum_passed_coverage.xml"
            report.write_text('<r><line count="1" featureClass="F.A" featureLineNum="4" /></r>')
            first=signatures(root)
            report.write_text('<r><line count="7" buggy="true" featureClass="F.A" featureLineNum="4" /></r>')
            second=signatures(root)
            self.assertNotEqual(first[0],second[0]);self.assertEqual(first[1],second[1])
            report.rename(root/"coverage/spectrum_failed_coverage.xml")
            self.assertEqual(second[1],signatures(root)[1])
            (root/"coverage/spectrum_failed_coverage.xml").write_text('<r><line count="0" truecount="7" featureClass="F.A" featureLineNum="4" /></r>')
            self.assertNotEqual(second[1],signatures(root)[1])
            before=signatures(root)[1];source.write_text("class A {int x;}")
            self.assertNotEqual(before,signatures(root)[1])


if __name__=="__main__":unittest.main()
