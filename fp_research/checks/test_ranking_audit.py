import math
import random
import unittest
from sklearn.metrics import average_precision_score
from fp_research.checks.verify_extension import average_precision


class RankingAudit(unittest.TestCase):
    def test_manual_ties_and_missing(self):
        self.assertAlmostEqual(average_precision([1,0,1],[.9,.8,.8]),5/6)
        self.assertAlmostEqual(average_precision([1,0,1],[1,None,None]),5/6)
        self.assertAlmostEqual(average_precision([1,0],[None,None]),.5)
        self.assertIsNone(average_precision([0,0],[1,0]))

    def test_independent_sum_matches_published_metric(self):
        rng=random.Random(191)
        for _ in range(200):
            labels=[1]+[rng.randrange(2) for i in range(rng.randrange(1,80))]
            scores=[rng.choice([None,-2,0,.5,1]) for i in labels]
            expected=average_precision_score(labels,[-1e30 if s is None else s for s in scores])
            self.assertTrue(math.isclose(average_precision(labels,scores),expected,abs_tol=1e-12))


if __name__=='__main__':unittest.main()
