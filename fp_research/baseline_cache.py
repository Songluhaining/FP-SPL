"""等价计算基线的路径相似性特征，重复覆盖用整数位集合表示。"""
from functools import lru_cache
from collections import OrderedDict
import os
import xml.etree.ElementTree as ET


def install_file_cache(version):
    """独立基线进程内只读复用覆盖树；输入摘要由外层在运行前后核对。"""
    original=ET.parse
    prefix=os.path.normcase(os.path.abspath(version))+os.sep
    @lru_cache(maxsize=384)
    def cached(path):return original(path)
    def parse(source,parser=None):
        if parser is None and isinstance(source,(str,os.PathLike)):
            path=os.path.normcase(os.path.abspath(source))
            if path.startswith(prefix) and path.endswith(".xml"):
                return cached(path)
        return original(source,parser=parser)
    ET.parse=parse
    return original,cached


def install(module):
    original=module.check_correctness_reflectability
    positions={}
    last_vectors=None

    # 公开实现对矩形覆盖矩阵逐行两两比较；相同列集合时，按整行分组完全等价。
    # 稀疏或不规则矩阵仍调用原函数，保留其填零和方向性行为。
    if hasattr(module,"ddu"):
        scope=module.ddu.__globals__
        old_pattern=scope["same_pattern"]
        def same_pattern(matrix):
            if not matrix:return old_pattern(matrix)
            columns=tuple(next(iter(matrix.values())))
            keys=set(columns)
            if any(set(row)!=keys for row in matrix.values()):return old_pattern(matrix)
            groups={}; result={k:0 for k in matrix}
            for key,row in matrix.items():
                pattern=tuple(row[c] for c in columns)
                group=groups.setdefault(pattern,[])
                group.append(key)
            for group in groups.values():
                result[group[0]]=len(group)
                for key in group[1:]:result[key]=-1
            return result
        scope["same_pattern"]=same_pattern

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
    if hasattr(module,"exist_path"):
        old_exist=module.exist_path
        path_cache={}; target_cache=None; item_positions={}
        def exact_path(path):
            saved=path_cache.get(id(path))
            if saved is not None and saved[0] is path:return saved[1]
            value=0
            for item in path:
                # 原函数比较整个记录字典，不能只比较语句id而丢失tested字段。
                key=frozenset(item.items())
                if key not in item_positions:item_positions[key]=len(item_positions)
                value|=1<<item_positions[key]
            path_cache[id(path)]=(path,value)
            return value
        @lru_cache(maxsize=2048)
        def covered(path,targets):
            return max(((path&t).bit_count() for t in targets),default=0)
        def exist_path(path,targets,threshold):
            nonlocal target_cache
            if not path or not targets:return False
            try:
                p=exact_path(path)
                # 原函数逐项计数；非重复路径可精确转为整数集合。
                if p.bit_count()!=len(path):return old_exist(path,targets,threshold)
                if target_cache is None or target_cache[0] is not targets:
                    target_cache=(targets,tuple(sorted({exact_path(t) for t in targets.values()})))
                return covered(p,target_cache[1])/len(path)>threshold
            except TypeError:
                return old_exist(path,targets,threshold)
        module.exist_path=exist_path
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
    if hasattr(module,"concat_slicies"):
        original_concat=module.concat_slicies
        combined=OrderedDict()
        def concat(left,right):
            key=(id(left),id(right))
            saved=combined.get(key)
            if saved is not None and saved[0] is left and saved[1] is right:
                combined.move_to_end(key)
                return saved[2]
            value=original_concat(left,right)
            combined[key]=(left,right,value)
            if len(combined)>128:combined.popitem(last=False)
            return value
        module.concat_slicies=concat
    if hasattr(module,"check_dependencies_by_slicing_type_with_susp_score"):
        previous_scores=None
        similarities_by_direction={}
        def dependency_similarity(failed,passed,statement,direction):
            saved=similarities_by_direction.get(direction)
            if saved is None or saved[0] is not passed:
                saved=(passed,{})
                similarities_by_direction[direction]=saved
            donor=saved[1].get(id(failed))
            if donor is None or donor[0] is not failed:
                donor=(failed,{})
                saved[1][id(failed)]=donor
            if statement not in donor[1]:
                donor[1][statement]=module.jaccard_similarity(set(passed[statement]),set(failed[statement]))
            return donor[1][statement]
        def weighted(similarities,statement,failed,passed,direction,scores):
            nonlocal previous_scores
            if previous_scores is None or previous_scores[0] is not scores:
                # 严格保留发布实现取“字典序最大语句键”的行为，不改成最大分数。
                previous_scores=(scores,scores[max(scores)])
            denominator=previous_scores[1]
            if denominator==0:
                similarities[statement]={"Backward":0,"Forward":0,"Both":0}
            elif statement in failed and statement in passed:
                value=dependency_similarity(failed,passed,statement,direction)*(scores[statement]/denominator)
                if statement not in similarities:
                    similarities[statement]={"Backward":0,"Forward":0,"Both":0}
                if value>similarities[statement][direction]:similarities[statement][direction]=value
            return similarities
        module.check_dependencies_by_slicing_type_with_susp_score=weighted
    return original,reflectability
