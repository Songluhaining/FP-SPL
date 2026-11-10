import tempfile
import unittest
from pathlib import Path

from fp_research.latent_fault import Observation, infer, spectrum


class TestLatentDiagnosis(unittest.TestCase):
    def test_posterior_includes_unexecuted_but_built_code(self):
        a, b = ("A.Class", "1"), ("B.Class", "2")
        records = [Observation("f", 1, 0, {a,b}, {a:1}, {}),
                   Observation("p", 0, 1, {a}, {}, {})]
        # A explains the failure with likelihood 1/2; B misses it with likelihood 1/3.
        # The passing test reaches neither statement and contributes 2/3 to both.
        scores = infer(records, "fixture")
        score = next(r["score"] for r in scores if r["method"]=="statement")
        self.assertAlmostEqual(score, 0.6)
        self.assertEqual(next(r["prediction"] for r in scores if r["method"]=="statement"), 1)

    def test_benchmark_markers_and_test_names_are_inert(self):
        template = ('<coverage><tests count="1"><test source="{}"/></tests>'
                    '<project><package><file><line num="5" count="1" featureClass="A.C" '
                    'featureLineNum="7" {}/></file></package></project></coverage>')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"coverage.xml"
            path.write_text(template.format("t", 'buggy="true"'), encoding="utf8")
            result = spectrum(path)
            path.write_text(template.format("unknown", 'buggy="false" fault="other"'), encoding="utf8")
            self.assertEqual(result, spectrum(path))
            path.write_text(template.format("t", ""), encoding="utf8")
            self.assertEqual(result, spectrum(path))
