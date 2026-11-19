"""仅供评价：独立重建基准标签，并同时报告正执行次数下的替代标签。"""
import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

from ..data.inventory import records


def ground_truth(data, systems):
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
