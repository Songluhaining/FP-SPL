import unittest
from fp_research.wrapper_recovery import translated,MOCK,STANDARD


class TestExceptionEquivalence(unittest.TestCase):
    def test_exact_location_and_declared_exception(self):
        source=[MOCK,"product","Base.Client:54"]
        target=[STANDARD,"product","Base.Client:54"]
        def compare(other):
            return translated("cfg/test 失效位置在插桩后改变 "+repr(source)+" -> "+repr(other))
        self.assertEqual(compare(target),("cfg/test",source,target))
        self.assertIsNone(compare([STANDARD,"product","Base.Client:55"]))
        self.assertIsNone(compare(["java.lang.NullPointerException","product","Base.Client:54"]))
