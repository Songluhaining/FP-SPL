"""在独立进程比较原50维依赖权重公式，保留字典序分母及其零值行为。"""
from copy import deepcopy
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"PULP"))
from consistent_testing_manager import FPMatricsCaculation as baseline
from fp_research.baseline_cache import install

original=baseline.check_dependencies_by_slicing_type_with_susp_score
install(baseline)
optimized=baseline.check_dependencies_by_slicing_type_with_susp_score
rng=random.Random(42)
checked=0
for trial in range(80):
    left={k:[str(i) for i in range(10) if rng.random()<.4] for k in "abcxyz" if rng.random()<.8}
    right={k:[str(i) for i in range(10) if rng.random()<.4] for k in "abcxyz" if rng.random()<.8}
    for metric in range(5):
        # 同一对依赖集合换用不同权重，实际命中跨公式缓存。
        scores={k:rng.choice([-1.,0.,.3,1.,2.]) for k in "abcxyz"}
        expected={};actual={}
        for direction in ("Backward","Forward","Both"):
            for statement in scores:
                original(expected,statement,left,right,direction,scores)
                optimized(actual,statement,left,right,direction,scores)
                assert actual==expected,(trial,metric,direction,statement)
                checked+=1
print(checked,"original weighted computations matched exactly")
