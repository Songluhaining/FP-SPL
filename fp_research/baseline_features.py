"""在独立进程中调用两种公开实现；所有输出均写入实验目录。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import sys

from .inventory import SYSTEMS, records

CLAP_FIELDS = ["VARIANT","LABEL","DDU","code_coverage","incorrectness_verifiability",
               "correctness_reflectability","bscp","bug_involving_statements"]


def worker(job,optimize=False):
    code, method, version, slices, out = job
    out.mkdir(parents=True,exist_ok=True)
    os.chdir(out)
    sys.path.insert(0,str(code))
    if method == "CLAP":
        import FileManager
        FileManager.join_path = lambda *parts: Path(os.path.join(*parts)).as_posix()
    from consistent_testing_manager import FPMatricsCaculation as f
    if optimize:
        from fp_research.baseline_cache import install,install_file_cache
        install(f)
        install_file_cache(version)
    rows=list(records(version))
    failing=[r["configuration"] for r in rows if r["outcome"]=="failed"]
    labels=out/"observations.csv"
    with labels.open("w",newline="",encoding="utf8") as s:
        writer=csv.writer(s)
        writer.writerow(["VARIANT","LABEL","FP transformed from F"])
        writer.writerows((r["configuration"],"F" if r["outcome"]=="failed" else "TP","0") for r in rows)
    f.get_labeled_failing_variants=lambda *_: failing
    if not hasattr(f,"_original_dependencies"):
        f._original_dependencies=f.get_dependencies
    @lru_cache(maxsize=512)
    def dependencies(path):
        relative=Path(path).relative_to(version)
        return f._original_dependencies(str(slices/relative))
    f.get_dependencies=dependencies
    target=out/"features.csv"
    args=(str(version),str(labels),str(out/"temporary.csv"),str(target))
    if method=="CLAP":
        f.calculate_attributes(*args,FIELDS=CLAP_FIELDS)
    else:
        from spectrum_manager import Spectrum_Expression as expr
        def ranked(project, positive):
            spaces=f.get_suspicious_space_consistent_version(project,positive,0.0,"")
            all_variants=f.list_dir(f.get_variants_dir(project))
            counts,passed,failed=f.get_infor_for_sbfl(project,positive,[v for v in all_variants if v not in positive],"",0.0)
            metrics=[getattr(expr,n) for n in "JACCARD OP2 TARANTULA BARINEL OCHIAI RUSSELL_RAO DSTAR GP02 GP03 GP19".split()]
            rankings=f.sbfl_ranking(counts,failed,passed,f.get_set_of_stms(spaces),metrics)
            return [{statement:score for statement,score,_ in rankings[m]} for m in metrics]
        f.ranking_suspicious_stmts=ranked
        f.calculate_attributes3(*args)
    with target.open(encoding="utf-8-sig",newline="") as s:
        features=list(csv.DictReader(s))
    expected={r["configuration"] for r in rows if method=="PULP" or r["outcome"]=="passed"}
    key="model" if method=="PULP" else "VARIANT"
    if len(features)!=len(expected) or {r[key] for r in features}!=expected:
        raise ValueError(f"特征行与原始清单不一致 {version}")
    print(method,version.name,len(features),hashlib.sha256(target.read_bytes()).hexdigest()[:12],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--code",type=Path,required=True)
    parser.add_argument("--slices",type=Path,required=True,help="各版本的静态切片目录")
    parser.add_argument("--method",choices=["CLAP","PULP"],required=True)
    parser.add_argument("--system",choices=SYSTEMS,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--version")
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.data.resolve()):
        parser.error("输出目录必须与原始数据隔离")
    jobs=[(args.code.resolve(),args.method,v.resolve(),args.slices.resolve()/v.name,
           args.output.resolve()/args.system/v.name/args.method)
          for v in sorted(args.data.joinpath(*SYSTEMS[args.system]).glob("_MultipleBugs_*"))
          if not args.version or args.version==v.name]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker,jobs))
