"""在新增系统的原案例目录内补测；仅共用同系统的参考构建缓存。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from .case_study import ROOT, JAVA, SYSTEMS, case_versions, results, raw_digest
from .extension_replay import build_plan
from .inventory import atomic_json, records
from .replay import replay_version


def run(job):
    system,version=job
    root=results(version,"main")
    before=raw_digest(version)
    inventory=root/"inventory.json"
    if not inventory.exists():
        # 补测可独立建观察清单，不把尚未结束的覆盖核验误报为数据缺失。
        inventory=root/"replay-inventory.json"
        rows=list(records(version))
        if any(not r["source"] or not r["test_source"] or not r["coverage_agrees"] for r in rows):
            raise ValueError("原始源码、测试或观察状态不完整")
        atomic_json(inventory,rows)
    rows=json.loads(inventory.read_text(encoding="utf8"))
    plan=build_plan(system,version,rows)
    target=root/"replay-plan.json"
    atomic_json(target,plan)
    shared=version.parent/".fp_research"
    projects=shared/"reference" if system=="gpl" else ROOT.parent/"CLAP/projects"
    tool=ROOT.parent/"CLAP/plugins/featurehouse_20210219.jar"
    replay_version((version,target,inventory,projects,tool,shared/"replay-cache",JAVA,system))
    if before!=raw_digest(version):
        raise ValueError("补测后原始输入摘要发生改变")
    atomic_json(root/"replay-complete.json",dict(raw_digest=before,configurations=len(plan)))
    (root/"replay-failure.json").unlink(missing_ok=True)
    return system,version.name


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("D:/dataset/whn"))
    p.add_argument("--system",choices=SYSTEMS)
    p.add_argument("--version")
    p.add_argument("--exclude-version",help="已由另一个运行中的首例核验任务处理的版本")
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    jobs=[(s,v) for s in ([a.system] if a.system else SYSTEMS)
          for v in case_versions(a.data,s) if (not a.version or v.name==a.version)
          and v.name!=a.exclude_version]
    if not jobs: raise ValueError("未找到选定案例")
    failures=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        tasks={pool.submit(run,j):j for j in jobs}
        for task in as_completed(tasks):
            job=tasks[task]
            try:
                print(*task.result(),"verified",flush=True)
            except Exception as ex:
                info=dict(system=job[0],version=job[1].name,type=type(ex).__name__,reason=str(ex))
                atomic_json(results(job[1],"main")/"replay-failure.json",info)
                failures.append(info)
                print("FAILED",info,flush=True)
    if failures: raise RuntimeError(f"{len(failures)} 个案例补测未完成，见各案例 replay-failure.json")
