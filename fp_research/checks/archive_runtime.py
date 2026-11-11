"""保存本轮计算所用的解释器、依赖库和Java运行环境版本，不收集环境变量。"""
from datetime import datetime
import importlib.metadata
import platform
import subprocess
import sys

from fp_research.case_study import ROOT,JAVA
from fp_research.inventory import atomic_json


def main():
    commit=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    java=subprocess.run([JAVA/'bin/java.exe','-version'],capture_output=True,text=True,check=True)
    data=dict(recorded=datetime.now().astimezone().isoformat(),snapshot_commit_before_archive=commit,
              python=dict(executable=sys.executable,version=sys.version),system=platform.platform(),
              libraries={name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','scikit-learn','joblib','psutil')},
              java=dict(home=str(JAVA),version=(java.stdout+java.stderr).strip()),
              reproduction=dict(case_workers=4,slice_workers=4,slice_heap_mb=1536,slice_backend='indexed',seed=42),
              note='归档时的运行环境；实际训练样本、参数、输入和源码摘要见各场景清单。并行数仅影响调度。')
    atomic_json(ROOT/'experimental_results/gpl_zipme/runtime.json',data)
    print('运行环境版本已保存',data['libraries'],flush=True)


if __name__=='__main__':main()
