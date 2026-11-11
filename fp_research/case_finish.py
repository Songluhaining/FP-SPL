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

from .case_study import ROOT,SYSTEMS,case_versions,results
from .checks.verify_extension import verify
from .inventory import atomic_json
from .result_registry import make


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("D:/dataset/whn"))
    p.add_argument("--include-masked",action="store_true")
    a=p.parse_args()
    output=ROOT/"experimental_results/gpl_zipme"
    output.mkdir(parents=True,exist_ok=True)
    state=output/"pipeline-status.json"
    registry=ROOT/"experimental_results/六系统单缺陷实验比较.csv"
    env=os.environ.copy()
    env.update(PYTHONUTF8="1",OPENBLAS_NUM_THREADS="1")
    def status(stage,**more):
        atomic_json(state,dict(updated=datetime.now().astimezone().isoformat(),stage=stage,**more))
    def command(module,*args):
        command=[sys.executable,"-m","fp_research."+module,*args]
        print("RUN",command,flush=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True)
    try:
        last_refresh=0
        while True:
            counts={s:sum((results(v,"main")/"complete.json").exists() for v in case_versions(a.data,s)) for s in SYSTEMS}
            totals={s:len(case_versions(a.data,s)) for s in SYSTEMS}
            if counts==totals:break
            status("等待所有主实验案例准备完成",completed=counts,total=totals)
            if time.monotonic()-last_refresh>600:
                make(a.data,registry)
                last_refresh=time.monotonic()
            time.sleep(30)
        for scenario in (["main","masked"] if a.include_masked else ["main"]):
            status("正在评价",scenario=scenario)
            if scenario=="masked":
                command("case_study","run","--scenario",scenario,"--workers","4","--data",str(a.data))
            command("case_evaluate","truth","--scenario",scenario,"--data",str(a.data))
            for method in ("CLAP","PULP"):
                command("case_evaluate","fit","--method",method,"--scenario",scenario,"--data",str(a.data))
                command("case_evaluate","repeat","--method",method,"--scenario",scenario,"--data",str(a.data))
            command("case_evaluate","audit","--scenario",scenario,"--data",str(a.data))
            verify(output/scenario)
            atomic_json(output/scenario/"registry-verified.json",dict(retraining="逐字节一致",independent_statistics="通过",
                        metrics_sha256=hashlib.sha256((output/scenario/"metrics.csv").read_bytes()).hexdigest(),
                        predictions_sha256=hashlib.sha256((output/scenario/"predictions.csv").read_bytes()).hexdigest()))
            make(a.data,registry)
        status("分类比较与独立核验已完成，实际补测另行核对")
    except Exception as ex:
        status("运行或核验失败，需要处理",error=type(ex).__name__,reason=str(ex))
        raise


if __name__=="__main__":main()
