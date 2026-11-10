"""从原始文件复现实验。默认从清单到独立验收，支持指定连续阶段。"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STAGES=("prepare","graph","model","features","classify","rank","replay","audit")


def run(a):
    out=a.work.resolve()
    raw=a.data.resolve()
    model_python=str(a.model_python.resolve()) if a.model_python else sys.executable
    work=a.workspace.resolve()
    java=a.java_home.resolve()
    input_data=raw if a.scenario=="main" else a.visible_data.resolve()
    cache=out/("fp" if a.scenario=="main" else "supplement")
    models=out/("models" if a.scenario=="main" else "supp-models")
    truth=out/"evaluation"/("truth.csv" if a.scenario=="main" else "supp-truth.csv")
    def result(method):
        return out/"evaluation"/(method+".csv" if a.scenario=="main" else "supp-"+method+".csv")
    def step(module,*args,deep=False):
        cmd=[model_python if deep else sys.executable,"-m","fp_research."+module,*map(str,args)]
        print(subprocess.list2cmdline(cmd),flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)

    active=STAGES[STAGES.index(a.start):STAGES.index(a.end)+1]
    if not active:
        raise ValueError("起始阶段必须早于结束阶段")
    if "prepare" in active:
        if a.scenario=="main":
            step("inventory","--data",raw,"--output",cache)
            step("pairing","--data",raw,"--output",cache)
            step("truth","--data",raw,"--output",truth)
        else:
            for s,folder in [("bank","pulp_trial"),("email","email_trial")]:
                step("supplement","--data",raw,"--staging",input_data,"--labels",work/folder/"dataset",
                     "--output",cache,"--java-home",java,"--system",s,"--workers",a.workers)
            step("supplement_analysis","--stage","semantics","--data",input_data,"--original-results",out/"fp",
                 "--output",cache,"--java-home",java,"--workers",a.workers)
            step("supplement_truth","--input",cache,"--main-truth",out/"evaluation/truth.csv","--output",truth)
        if a.scenario=="main":
            step("sites","--data",raw,"--input",cache,"--output",out/"sites","--java-home",java,"--workers",a.workers)
        step("refine","--stage","metadata","--sites",out/"sites","--data",input_data,"--runtime",out/"fp/runtime",
             "--input",cache,"--output",cache,"--java-home",java,"--workers",a.workers)
    if "graph" in active:
        step("traced_graphs","--data",input_data,"--input",cache,"--cache",out/"isolated-cache",
             "--java-home",java,"--workers",a.workers)
        step("wrapper_recovery","--data",input_data,"--input",cache,"--cache",out/"isolated-cache",
             "--java-home",java,"--workers",a.workers)
    if "model" in active:
        step("fit","--input",cache,"--output",models,deep=True)
        if a.scenario=="main":
            step("paired_check","--input",cache,"--data",raw,"--models",models,"--output",out/"evaluation/paired-check.csv",deep=True)
    if "features" in active:
        for method in ("CLAP","PULP"):
            for s,folder in [("bank","pulp_trial"),("email","email_trial")]:
                step("baseline_features","--data",input_data,"--code",work/method,"--method",method,
                     "--slices",work/folder/"dataset","--output",cache,"--system",s,"--workers",a.workers)
    if "classify" in active:
        for method in ("CLAP","PULP"):
            step("compare","--code",work/method,"--method",method,"--input",cache,"--truth",truth,
                 "--output",result(method.lower()))
    if "rank" in active:
        step("risk","--input",cache,"--data",input_data,"--models",models,"--output",result("risk"))
    if "replay" in active and a.scenario=="main":
        step("probes","--risk",result("risk"),"--input",cache,"--data",raw,"--models",models,"--output",out/"replay",deep=True)
        step("replay","--data",raw,"--plans",out/"replay","--input",cache,"--projects",work/"CLAP/projects",
             "--tool",work/"CLAP/plugins/featurehouse_20210219.jar","--java-home",java,"--workers",a.workers)
    if "audit" in active:
        args=["--input",cache,"--truth",truth,"--risk",result("risk"),"--clap",result("clap"),"--pulp",result("pulp"),
              "--output",a.results.resolve()/a.scenario]
        if a.scenario=="main":
            args+=["--replay",out/"replay"]
        step("report",*args)
        evidence_args=["--input",cache,"--models",models,"--output",a.results.resolve()/a.scenario]
        if a.scenario=="masked":
            evidence_args+=["--original",raw,"--masked",input_data]
        step("audit_evidence",*evidence_args)
        if a.scenario=="main":
            shutil.copyfile(out/"evaluation/paired-check.csv",a.results.resolve()/"main/paired-check.csv")
            step("probe_evidence","--input",cache,"--plans",out/"replay","--models",models,"--risk",result("risk"),
                 "--output",a.results.resolve()/"main/probe-evidence.csv",deep=True)
            step("provenance","--data",raw,"--input",cache,"--workspace",work,"--output",a.results.resolve())
        report_root=a.results.resolve()
        if (report_root/"main/replay-summary.csv").exists() and (report_root/"masked/metrics.csv").exists():
            step("publish_report","--results",report_root,"--paired",out/"evaluation/paired-check.csv")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k,default in [("data","D:/dataset/whn"),("work",ROOT/"result"),("workspace",ROOT.parent),
                      ("visible-data","D:/dataset/whn/fp_spl_visible_20261110"),("java-home","E:/Program Files/Java/jdk-1.8"),
                      ("results",ROOT/"experimental_results/fp_detection")]:
        p.add_argument("--"+k,default=Path(default),type=Path)
    p.add_argument("--model-python",type=Path)
    p.add_argument("--scenario",choices=["main","masked"],default="main")
    p.add_argument("--start",choices=STAGES,default="prepare")
    p.add_argument("--end",choices=STAGES,default="audit")
    p.add_argument("--workers",type=int,default=3)
    run(p.parse_args())
