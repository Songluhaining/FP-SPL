import unittest
from fp_research.isolated_trace import narrow_test
from fp_research.test_sources import methods


class TestIsolatedMethod(unittest.TestCase):
    def test_positions_and_helper_are_preserved(self):
        source='''class Checks {
        int helper(){return 1;}
        @Test public void unused(){String s="}"; /* { */ }
        @Test public void selected(){assertEquals(1,helper());}
        }'''
        slim=narrow_test(source,"selected")
        self.assertEqual(len(source),len(slim))
        self.assertEqual(source.count("\n"),slim.count("\n"))
        self.assertEqual([e[0] for e in methods(slim)],["selected"])
        self.assertEqual(source.index("assertEquals"),slim.index("assertEquals"))
        self.assertIn("int helper()",slim)

    def test_test_calling_another_test_retains_it(self):
        source="class T {@Test public void one(){two();} @Test public void two(){}}"
        self.assertEqual(narrow_test(source,"one"),source)
