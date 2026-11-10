import unittest
import numpy as np
from fp_research.detection import DetectionNet, descriptor, encode, predict


class TestDetection(unittest.TestCase):
    def test_no_outcome_or_order_feature(self):
        raw=dict(operations=["pay"],assertions=["balance"],inputs=["1"])
        target=descriptor(raw)
        self.assertEqual(target,descriptor(dict(raw,outcome="failed",identity="must_not_use")))
        bag=encode(target,[target,descriptor(dict(raw,assertions=["status"]))])
        model=DetectionNet().eval()
        scores=predict(model,[bag,bag[::-1]])
        self.assertAlmostEqual(float(scores[0]),float(scores[1]),places=6)
        self.assertFalse(np.array_equal(bag[0],bag[1]))
