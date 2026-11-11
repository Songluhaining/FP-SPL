"""等价计算基线的路径相似性特征，重复覆盖用整数位集合表示。"""
from functools import lru_cache


def install(module):
    original=module.check_correctness_reflectability
    positions={}
    last_vectors=None

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
        nonlocal last_vectors
        if not 0 <= threshold <= 1:
            return original(failed,passed,target,suspicious,threshold)
        # 原入口在同一配置上连续遍历10个阈值，此间三个观察对象均只读。
        if last_vectors is not None and all(a is b for a,b in zip(last_vectors[:3],(failed,passed,target))):
            f,p,t=last_vectors[3:]
        else:
            f=tuple(sorted({mask(case) for variants in failed.values() for case in variants.values()}))
            p=tuple(sorted({mask(case) for variants in passed.values() for case in variants.values()}))
            t=tuple(mask(case) for case in target.values())
            last_vectors=(failed,passed,target,f,p,t)
        values=scores(f,p,t)
        return sum(value>threshold for value in values)/len(values)

    module.check_correctness_reflectability=reflectability
    def wrap_dependencies(function):
        previous=None
        def deduplicated(folder,variant,failures,*args):
            nonlocal previous
            if previous is not None and previous[0] is failures:
                unique=previous[1]
            else:
                unique={}
                for donor,tests in failures.items():
                    statements={}
                    for execution in tests.values():
                        for item in execution:
                            statements.setdefault(item["id"],item)
                    unique[donor]={"union":list(statements.values())}
                previous=(failures,unique)
            return function(folder,variant,unique,*args)
        return deduplicated
    for name in ("check_dependencies","check_dependencies_with_susp_score"):
        if hasattr(module,name):
            setattr(module,name,wrap_dependencies(getattr(module,name)))
    if hasattr(module,"check_dependencies_by_slicing_type_with_susp_score"):
        previous_scores=None
        def weighted(similarities,statement,failed,passed,direction,scores):
            nonlocal previous_scores
            if previous_scores is None or previous_scores[0] is not scores:
                # 严格保留发布实现取“字典序最大语句键”的行为，不改成最大分数。
                previous_scores=(scores,scores[max(scores)])
            denominator=previous_scores[1]
            if denominator==0:
                similarities[statement]={"Backward":0,"Forward":0,"Both":0}
            elif statement in failed and statement in passed:
                value=module.jaccard_similarity(set(passed[statement]),set(failed[statement]))*(scores[statement]/denominator)
                if statement not in similarities:
                    similarities[statement]={"Backward":0,"Forward":0,"Both":0}
                if value>similarities[statement][direction]:similarities[statement][direction]=value
            return similarities
        module.check_dependencies_by_slicing_type_with_susp_score=weighted
    return original,reflectability
