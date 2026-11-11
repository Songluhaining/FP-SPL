"""等待所有原始案例准备完成后评价并刷新统一结果表；任何核验失败均停止后续统计。"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

from .case_study import ROOT,SYSTEMS,case_versions,results
from .checks.verify_extension import verify
from .inventory import atomic_json
from .result_registry import make


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("D:/dataset/whn"))
    p.add_argument("--include-masked",action="store_true")
    p.add_argument("--wait-main-pid",action="append",default=[],help="等待已有主实验进程退出，格式为系统:进程号")
    a=p.parse_args()
    output=ROOT/"experimental_results/gpl_zipme"
    output.mkdir(parents=True,exist_ok=True)
    state=output/"pipeline-status.json"
    registry=ROOT/"experimental_results/六系统单缺陷实验比较.csv"
    env=os.environ.copy()
    env.update(PYTHONUTF8="1",OPENBLAS_NUM_THREADS="1")
    main_processes=[]
    for item in a.wait_main_pid:
        system,pid=item.split(":",1)
        if system not in SYSTEMS:raise ValueError("等待进程的系统名称错误")
        try:
            process=psutil.Process(int(pid))
            main_processes.append((system,process,process.create_time()))
        except psutil.NoSuchProcess:pass
    def active_main():
        active=set()
        for system,process,created in main_processes:
            try:
                if process.is_running() and process.create_time()==created:active.add(system)
            except psutil.NoSuchProcess:pass
        return active
    def status(stage,**more):
        atomic_json(state,dict(updated=datetime.now().astimezone().isoformat(),stage=stage,**more))
    def command(module,*args):
        command=[sys.executable,"-m","fp_research."+module,*args]
        print("RUN",command,flush=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True)
    masked_processes={}
    masked_logs=[]
    def start_masked(system):
        if system in masked_processes:return
        pending=[v for v in case_versions(a.data,system) if not (results(v,"masked")/"complete.json").exists()]
        if not pending:
            masked_processes[system]=None
            return
        log=(ROOT/"result"/("case-masked-"+system+".log")).open("a",encoding="utf8")
        masked_logs.append(log)
        args=[sys.executable,"-m","fp_research.case_study","run","--system",system,
              "--scenario","masked","--workers","4","--pending-only","--data",str(a.data)]
        print("RUN",args,flush=True)
        masked_processes[system]=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    def check_masked_processes():
        for system,process in masked_processes.items():
            if process is not None and process.poll() not in (None,0):
                raise RuntimeError(f"{system}补充实验准备失败，详见case-masked-{system}.log")
    try:
        last_refresh=0
        while True:
            counts={s:sum((results(v,"main")/"complete.json").exists() for v in case_versions(a.data,s)) for s in SYSTEMS}
            totals={s:len(case_versions(a.data,s)) for s in SYSTEMS}
            active=active_main()
            if a.include_masked:
                for system in SYSTEMS:
                    # 同系统主实验不再写切片缓存后，才启动该系统补充实验。
                    if counts[system]==totals[system] and system not in active:start_masked(system)
                check_masked_processes()
            if counts==totals and not active:break
            status("等待所有主实验案例准备完成",completed=counts,total=totals,active_main=sorted(active))
            if time.monotonic()-last_refresh>600:
                make(a.data,registry)
                last_refresh=time.monotonic()
            time.sleep(30)
        for scenario in (["main","masked"] if a.include_masked else ["main"]):
            status("正在评价",scenario=scenario)
            if scenario=="masked":
                while any(p is not None and p.poll() is None for p in masked_processes.values()):
                    check_masked_processes()
                    status("主实验已核验；等待补充实验准备完成",completed={s:sum((results(v,"masked")/"complete.json").exists()
                           for v in case_versions(a.data,s)) for s in SYSTEMS})
                    time.sleep(30)
                check_masked_processes()
            command("case_evaluate","truth","--scenario",scenario,"--data",str(a.data))
            for method in ("CLAP","PULP"):
                command("case_evaluate","fit","--method",method,"--scenario",scenario,"--data",str(a.data))
                command("case_evaluate","repeat","--method",method,"--scenario",scenario,"--data",str(a.data))
            command("case_evaluate","audit","--scenario",scenario,"--data",str(a.data))
            verify(output/scenario)
            if scenario=="main":command("case_evaluate","replay","--data",str(a.data))
            atomic_json(output/scenario/"registry-verified.json",dict(retraining="逐字节一致",independent_statistics="通过",
                        metrics_sha256=hashlib.sha256((output/scenario/"metrics.csv").read_bytes()).hexdigest(),
                        predictions_sha256=hashlib.sha256((output/scenario/"predictions.csv").read_bytes()).hexdigest()))
            make(a.data,registry)
        if a.include_masked:command("case_report")
        status("主实验、补充实验与补测记录均已核验" if a.include_masked else "主实验与补测记录已核验")
    except Exception as ex:
        status("运行或核验失败，需要处理",error=type(ex).__name__,reason=str(ex))
        raise
    finally:
        for log in masked_logs:log.close()


if __name__=="__main__":main()
