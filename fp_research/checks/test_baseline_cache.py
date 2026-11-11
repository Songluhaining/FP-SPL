"""逐点比较原基线公式与缓存计算，覆盖阈值边界和重复轨迹。"""
import random
from copy import deepcopy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"CLAP"))
from consistent_testing_manager import FPMatricsCaculation as baseline
from fp_research.baseline_cache import install

ORIGINAL_PATTERN=baseline.ddu.__globals__["same_pattern"]
ORIGINAL_EXIST=baseline.exist_path
ORIGINAL_REFLECT=baseline.check_correctness_reflectability
ORIGINAL_CONCAT=baseline.concat_slicies
install(baseline)


class Equivalence(unittest.TestCase):
    def test_combined_dependencies(self):
        rng=random.Random(15)
        for _ in range(60):
            left={str(k):[str(i) for i in range(9) if rng.random()<.4] for k in range(7)}
            right={str(k):[str(i) for i in range(9) if rng.random()<.4] for k in range(3,10)}
            expected=ORIGINAL_CONCAT(left,right)
            self.assertEqual(expected,baseline.concat_slicies(left,right))
            self.assertEqual(expected,baseline.concat_slicies(left,right))

    def test_directional_path_coverage(self):
        original=ORIGINAL_EXIST
        optimized=baseline.exist_path
        rng=random.Random(91)
        for trial in range(80):
            def path():return [{"id":str(i),"tested":rng.randrange(2)} for i in range(12) if rng.random()<.6]
            source=path(); targets={str(i):path() for i in range(rng.randrange(5))}
            if trial%3==0 and source:source.append(dict(source[0]))
            for threshold in [-1,0,.3,.8,1,2]:
                self.assertEqual(original(source,targets,threshold),optimized(source,targets,threshold))

    def test_coverage_patterns(self):
        original=ORIGINAL_PATTERN
        optimized=baseline.ddu.__globals__["same_pattern"]
        rng=random.Random(72)
        for trial in range(100):
            matrix={str(i):{str(j):rng.randrange(2) for j in range(9)} for i in range(rng.randrange(1,18))}
            if trial%2==0:matrix["duplicate"]=dict(matrix["0"])
            if trial%3==0:matrix["0"].pop("3")
            left,right=deepcopy(matrix),deepcopy(matrix)
            self.assertEqual(original(left),optimized(right))
            self.assertEqual(left,right)

    def test_original_formulas(self):
        original,optimized=ORIGINAL_REFLECT,baseline.check_correctness_reflectability
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
