import unittest

from fp_research.pairing import pair_cases


class TestControl(unittest.TestCase):
    def test_order_and_observation(self):
        cases = {"v/C." + name: dict(operations=["pay"], inputs=["1"], assertions=[oracle])
                 for name, oracle in [("f","balance"),("a","balance"),("b","status")]}
        def run(order):
            row = dict(configuration="v", tests=dict(failed=["C.f"], passed=order))
            return pair_cases(cases, [row])["v/C.f"]["control"]
        self.assertEqual(run(["C.b","C.a"]), "v/C.a")
        self.assertEqual(run(["C.a","C.b"]), "v/C.a")
        self.assertIsNone(run(["C.b"]))
