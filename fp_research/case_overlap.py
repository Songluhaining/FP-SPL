"""仅对主实验已完成的案例提前运行补充实验，并排除跨场景缓存写入冲突。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from . import case_study as study
from .slice_reuse import signatures


class CacheConflict(RuntimeError):
    pass


def conflicts(keys, protected, cache):
    return sorted(k for k in set(keys)&set(protected) if not (cache/k/"complete.json").exists())


def completed_aliases(index,cache):
    return {e['canonical']:e['raw'] for e in index['entries'] if (cache/e['raw']/"complete.json").exists()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system",choices=study.SYSTEMS,required=True)
    parser.add_argument("--data",type=Path,default=Path("D:/dataset/whn"))
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--version",help="先在一个已完成主实验的版本上核验提前运行流程")
    a=parser.parse_args()
    versions=study.case_versions(a.data,a.system)
    candidates=[v for v in versions if not a.version or v.name==a.version]
    if not candidates:raise ValueError("没有选定案例")
    jar=hashlib.sha256((study.ROOT.parent/"PULP/plugins/all-statements-slicing.jar").read_bytes()).hexdigest()
    index=json.loads((versions[0].parent/'.fp_research/slice-signatures.json').read_text(encoding='utf8'))
    if index['jar']!=jar:raise ValueError('输入索引对应的切片工具版本不符')
    # 从当前原始输入重建待写键；不依赖旧索引中的完成状态。
    keys={v.name:set() for v in versions}
    for v in versions:
        for variant in sorted((v/"variants").iterdir()):
            raw,_=signatures(variant)
            keys[v.name].add(hashlib.sha256((jar+raw).encode()).hexdigest())
    registered={e['raw'] for e in index['entries']}
    if set().union(*keys.values())!=registered:raise ValueError('当前输入与既有切片索引不一致')
    print(a.system,'全部原始切片输入摘要已核对',flush=True)
    original=study.slices

    def guarded(source,output,cache):
        owner=next(v for v in versions if v.name==source.name)
        if not (study.results(owner,"main")/"complete.json").exists():
            raise ValueError("本案例主实验未完成，禁止提前生成补充证据")
        pending=[v.name for v in versions if not (study.results(v,"main")/"complete.json").exists()]
        protected=set().union(*(keys[name] for name in pending))
        checked=[]
        for variant in sorted((source/"variants").iterdir()):
            raw,_=signatures(variant)
            key=hashlib.sha256((jar+raw).encode()).hexdigest()
            checked.append(dict(configuration=variant.name,key=key))
        overlap=conflicts([r["key"] for r in checked],protected,cache)
        if overlap:raise CacheConflict("待写缓存与未完成主实验相交："+str(overlap))
        # 主实验仍在推进，每个补充案例开始时补入新完成的等价缓存；不接收未完成文件。
        available=completed_aliases(index,cache)
        with study._GUARD:
            aliases=study._ALIASES.setdefault(cache,{})
            for canonical,key in available.items():aliases.setdefault(canonical,key)
        study.atomic_json(output/"concurrency-check.json",dict(unfinished_main_versions=pending,checked=checked))
        return original(source,output,cache)

    study.slices=guarded
    queued=set(); deferred=set(); running={}
    with ProcessPoolExecutor(max_workers=min(4,a.workers)) as auditing, ThreadPoolExecutor(max_workers=a.workers) as pool:
        study._AUDIT_POOL=auditing
        while True:
            for future,version in list(running.items()):
                if not future.done():continue
                del running[future]
                try:print(*future.result(),flush=True)
                except CacheConflict as error:
                    deferred.add(version.name)
                    print("DEFER",version.name,str(error),flush=True)
                except Exception as error:
                    study.atomic_json(study.results(version,"masked")/"failure.json",
                                      dict(type=type(error).__name__,reason=str(error)))
                    raise
            all_main=all((study.results(v,"main")/"complete.json").exists() for v in versions)
            if all_main and deferred:
                queued-=deferred
                deferred.clear()
            for v in candidates:
                if len(running)>=a.workers:break
                if v.name in queued or (study.results(v,"masked")/"complete.json").exists():continue
                if not (study.results(v,"main")/"complete.json").exists():continue
                queued.add(v.name)
                running[pool.submit(study.process_case,(a.system,v,"masked",False))]=v
            if not running and all((study.results(v,"masked")/"complete.json").exists() for v in candidates):break
            time.sleep(10)
    print(a.system,len(candidates),"个补充案例已核验",flush=True)


if __name__=="__main__":main()
