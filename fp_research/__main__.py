"""研究流程统一入口。每个子命令保留自己的参数和帮助。"""
import argparse
import runpy
import sys


STAGES={
    "inventory":"inventory",
    "group-and-pair":"pairing",
    "infer":"graphs",
    "fit":"fit",
    "rank":"risk",
    "select-tests":"probes",
    "replay":"replay",
    "baseline-features":"baseline_features",
    "baseline-predict":"compare",
    "ground-truth":"truth",
    "mask-tests":"supplement",
    "masked-inputs":"supplement_analysis",
    "masked-truth":"supplement_truth",
    "paired-check":"paired_check",
    "provenance":"provenance",
    "audit":"report",
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=STAGES)
    stage=parser.parse_args(sys.argv[1:2]).stage
    sys.argv=[sys.argv[0]+" "+stage,*sys.argv[2:]]
    runpy.run_module("fp_research."+STAGES[stage],run_name="__main__")


if __name__=="__main__":
    main()
