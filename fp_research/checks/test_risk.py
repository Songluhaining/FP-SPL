import unittest
from fp_research.risk import combine

class TestRisk(unittest.TestCase):
    def test_ranking(self):
        self.assertAlmostEqual(combine(0.8,0.25),0.6)
        self.assertGreater(combine(0.8,0.25),combine(1,0.7))
        self.assertEqual(combine(0,0.1),0)
