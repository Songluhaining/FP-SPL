"""逐点比较原基线公式与缓存计算，覆盖阈值边界和重复轨迹。"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"CLAP"))
from consistent_testing_manager import FPMatricsCaculation as baseline
from fp_research.baseline_cache import install


class Equivalence(unittest.TestCase):
    def test_original_formulas(self):
        original,optimized=install(baseline)
        rng=random.Random(42)
        def cases(n):
            return {str(i):[{"id":str(j)} for j in range(10) if rng.random()<0.4] for i in range(n)}
        for trial in range(60):
            fail={"a":cases(rng.randrange(5))}
            passed={"b":cases(rng.randrange(6))}
            target=cases(rng.randrange(1,8))
            if trial%2==0:target["duplicate"]=target["0"]
            for threshold in [i/10 for i in range(11)]:
                self.assertEqual(original(fail,passed,target,{},threshold),
                                 optimized(fail,passed,target,{},threshold))


if __name__=="__main__": unittest.main()
