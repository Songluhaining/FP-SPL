"""用实验构造清单扩展原本通过的评价真值，隐藏用例与缺陷位置不进入预测数据。"""
import argparse
import csv
import json
from pathlib import Path
from .risk import save_rows

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-truth",required=True,type=Path)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    with a.main_truth.open(encoding="utf8",newline="") as s:
        rows=[dict(r,converted=0) for r in csv.DictReader(s)]
    base={(r["system"],r["version"]):r["base"] for r in rows}
    for f in a.input.glob("*/*/construction.json"):
        v=f.parent.name
        system=f.parent.parent.name
        for c,receipt in json.loads(f.read_text(encoding="utf8")).items():
            if receipt["status"]=="passed":
                rows.append(dict(system=system,version=v,configuration=c,label=1,executed_label="",
                                 base=base[(system,v)],converted=1))
    keys={(r["system"],r["version"],r["configuration"]) for r in rows}
    if len(keys)!=len(rows):
        raise ValueError("补充样本与原主表重叠")
    save_rows(a.output,rows)
