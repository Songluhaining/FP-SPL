"""仅供评价：独立重建基准标签，并同时报告正执行次数下的替代标签。"""
import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

from .inventory import records, SYSTEMS


def ground_truth(data, systems=SYSTEMS):
    for system, relative in systems.items():
        for version in sorted(data.joinpath(*relative).glob("_MultipleBugs_*")):
            lines=(version/(version.name+".mutant.log")).read_text(encoding="utf8").splitlines()
            faults=set()
            for line in lines:
                location, number, _=line.split(":",2)
                faults.add(location.rsplit(".",1)[0]+":"+number)
            for record in records(version):
                if record["outcome"]!="passed":
                    continue
                path=version/"variants"/record["configuration"]/"coverage/spectrum_passed_coverage.xml"
                present,hit=False,False
                for line in ET.parse(path).iterfind(".//package/file/line"):
                    key=str(line.get("featureClass"))+":"+str(line.get("featureLineNum"))
                    if key in faults and key.split(".",1)[0] in record["features"]:
                        present=True
                        hit |= any(int(line.get(attr,0))>0 for attr in ("count","truecount","falsecount"))
                yield dict(system=system,version=version.name,configuration=record["configuration"],
                           label=int(present),executed_label=int(hit),
                           base=int(all(k.split(".",1)[0] in {"BankAccount","Base","ExamDB"} for k in faults)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.data.resolve()):
        parser.error("请从原始数据目录外写出标签")
    rows=list(ground_truth(args.data))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8",newline="") as s:
        writer=csv.DictWriter(s,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for system in SYSTEMS:
        group=[r for r in rows if r["system"]==system]
        print(system,len(group),sum(r["label"] for r in group),sum(r["executed_label"] for r in group))
