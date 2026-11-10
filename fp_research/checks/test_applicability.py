import unittest
from fp_research.applicability import applicability


class TestBuildMapping(unittest.TestCase):
    def test_present_without_execution(self):
        scores={"Shared.A:1":0.6,"Option.B:5":0.4}
        r=applicability(scores,{"Shared"},{"Shared.A:1"},set())
        self.assertEqual((r["score"],r["present_unexecuted"]),(0.6,1))
        self.assertEqual(applicability(scores,{"Shared","Option"},set(scores),set(scores))["score"],1)
        self.assertEqual(applicability(scores,set(),set(scores),set())["score"],0)
        self.assertIsNone(applicability(scores,{"Shared"},set(),set()))
