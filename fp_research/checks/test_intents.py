import unittest

from fp_research.intents import group_failures, parse_source


class TestIntent(unittest.TestCase):
    def test_same_operation_different_observation(self):
        source = """class T {
        @Test public void test1(){ int value = a.withdraw(3); assertEquals(0,value); }
        @Test public void test2(){ int other = a.withdraw(3); assertEquals(0,other); }
        @Test public void test3(){ a.withdraw(3); assertEquals(0,a.balance); }}"""
        descriptions = parse_source(source)
        result = group_failures([(name, d, {("Account", 3)}) for name, d in descriptions.items()])
        self.assertEqual(sorted(result.values()), [["test1", "test2"], ["test3"]])
        self.assertEqual(descriptions["test1"]["assertions"], ["assertEquals/call:withdraw"])
