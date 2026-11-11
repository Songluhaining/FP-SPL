"""等价计算基线的路径相似性特征，重复覆盖用整数位集合表示。"""
from functools import lru_cache


def install(module):
    original=module.check_correctness_reflectability
    positions={}

    def mask(execution):
        result=0
        for item in execution:
            key=item["id"]
            if key not in positions:
                positions[key]=len(positions)
            result |= 1 << positions[key]
        return result

    def similarity(left,right):
        overlap=(left & right).bit_count()
        union=left.bit_count()+right.bit_count()-overlap
        return overlap/union if union else 0

    @lru_cache(maxsize=256)
    def failure_similarity(passed,failed):
        return tuple(max((similarity(p,f) for f in failed),default=0) for p in passed)

    @lru_cache(maxsize=256)
    def scores(failed,passed,targets):
        supported=failure_similarity(passed,failed)
        return tuple(max((min(weight,similarity(target,reference))
                          for reference,weight in zip(passed,supported)),default=0)
                     for target in targets)

    def reflectability(failed,passed,target,suspicious,threshold):
        if not 0 <= threshold <= 1:
            return original(failed,passed,target,suspicious,threshold)
        f=tuple(sorted({mask(case) for variants in failed.values() for case in variants.values()}))
        p=tuple(sorted({mask(case) for variants in passed.values() for case in variants.values()}))
        t=tuple(mask(case) for case in target.values())
        values=scores(f,p,t)
        return sum(value>threshold for value in values)/len(values)

    module.check_correctness_reflectability=reflectability
    def wrap_dependencies(function):
        def deduplicated(folder,variant,failures,*args):
            unique={}
            for donor,tests in failures.items():
                statements={}
                for execution in tests.values():
                    for item in execution:
                        statements.setdefault(item["id"],item)
                unique[donor]={"union":list(statements.values())}
            return function(folder,variant,unique,*args)
        return deduplicated
    for name in ("check_dependencies","check_dependencies_with_susp_score"):
        if hasattr(module,name):
            setattr(module,name,wrap_dependencies(getattr(module,name)))
    return original,reflectability
