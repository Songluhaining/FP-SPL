"""核对失效分组、推理成员、折划分及补充测试隔离，归档逐组证据。"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .inventory import SYSTEMS
from .risk import save_rows
from .test_sources import PACKAGE,methods


def check_masked(raw,masked,rows,receipts):
    for name,item in receipts.items():
        if item["status"]!="passed":
            continue
        folder=masked/"variants"/name
        original=raw/"variants"/name/"coverage"/"failed"
        hidden={p.name.removesuffix(".coverage.xml") for p in original.glob("*.coverage.xml")}
        coverage=folder/"coverage"
        if (coverage/"failed").exists() or (coverage/"spectrum_failed_coverage.xml").exists():
            raise ValueError(f"残留失败覆盖 {folder}")
        expected=set(rows[name]["tests"]["passed"])
        tree=ET.parse(coverage/"spectrum_passed_coverage.xml")
        listed={test.get("source").removesuffix(".coverage.xml") for test in tree.iterfind(".//tests/test")}
        if listed!=expected or listed&hidden or len(listed)!=item["remaining"] or len(hidden)!=item["removed"]:
            raise ValueError(f"留存用例与汇总覆盖不一致 {folder}")
        local=set()
        for p in (folder/"test").rglob("*Test.java"):
            source=p.read_text(encoding="utf-8-sig")
            pkg=PACKAGE.search(source)
            cls=(pkg[1]+"." if pkg else "")+p.stem
            local.update(cls+"."+m for m,_,_ in methods(source))
        if local&hidden or not expected<=local:
            raise ValueError(f"源码内存在被隐藏的方法或可见方法缺失 {folder}")


def verify(root,models,out,original=None,masked=None):
    out.mkdir(parents=True,exist_ok=True)
    stages=[]
    folds=[]
    with gzip.open(out/"groups.jsonl.gz","wt",encoding="utf8") as archive:
        for path in sorted(root.glob("*/*/inventory.json")):
            v=path.parent
            sys=v.parent.name
            raw=json.loads(path.read_text(encoding="utf8"))
            rows={r["configuration"]:r for r in raw}
            visible={c+"/"+t for c,r in rows.items() for status in ("passed","failed") for t in r["tests"][status]}
            failed={c+"/"+t for c,r in rows.items() for t in r["tests"]["failed"]}
            sem=json.loads((v/"semantics.json").read_text(encoding="utf8"))
            members=[identity for ids in sem["groups"].values() for identity in ids]
            if len(members)!=len(failed) or set(members)!=failed or set(sem["pairs"])!=failed or set(sem["cases"])!=visible:
                raise ValueError(f"覆盖、分组或配对的主键不一致 {v}")
            for key,item in sem["pairs"].items():
                c=item["control"]
                if c and (c not in visible-failed or c.split("/")[0]!=key.split("/")[0]):
                    raise ValueError(f"对照状态或配置不匹配 {v} {key}")
            ok=0
            for group,ids in sem["groups"].items():
                expected={i:"failed" for i in ids}
                expected.update({sem["pairs"][i]["control"]:"passed" for i in ids if sem["pairs"][i]["control"]})
                record=json.loads((v/"graphs"/(group+".json")).read_text(encoding="utf8"))
                if record["executions"]!=expected:
                    raise ValueError(f"执行成员与可见对照不一致 {v} {group}")
                if record["status"]=="ok":
                    if not record["scores"] or not all(math.isfinite(s) and 0<=s<=1 for s in record["scores"].values()):
                        raise ValueError("图后验异常")
                    ok+=1
                archive.write(json.dumps(dict(system=sys,version=v.name,group=group,**record),ensure_ascii=False)+"\n")
            stages.append(dict(system=sys,version=v.name,visible_tests=len(visible),failures=len(failed),
                paired=sum(i["control"] is not None for i in sem["pairs"].values()),groups=len(sem["groups"]),
                scored_groups=ok,unscored_groups=len(sem["groups"])-ok))
            if masked is not None:
                check_masked(original.joinpath(*SYSTEMS[sys],v.name),masked.joinpath(*SYSTEMS[sys],v.name),rows,
                             json.loads((v/"construction.json").read_text(encoding="utf8")))
    save_rows(out/"stage-checks.csv",stages)
    for f in sorted(models.glob("*/fold.json")):
        data=json.loads(f.read_text(encoding="utf8"))
        train={tuple(Path(p).parts[-2:]) for p in data["train"]}
        test={tuple(Path(p).parts[-2:]) for p in data["test"]}
        if train&test or not train or not test:
            raise ValueError("模型训练和测试版本重叠")
        model_hash=hashlib.sha256((f.parent/"model.pt").read_bytes()).hexdigest()
        for role,vs in (("train",train),("test",test)):
            for sys,v in sorted(vs):
                folds.append(dict(split=f.parent.name,role=role,system=sys,version=v,
                                  model_sha256=model_hash,training_subsets=data["train_bags"],
                                  validation_subsets=data["held_out_bags"],validation_brier=data["brier"] if data["held_out_bags"] else None))
    if len(stages)!=109 or len({r["split"] for r in folds})!=12:
        raise ValueError("双系统版本或训练折不全")
    save_rows(out/"folds.csv",folds)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("input","models","output"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--original",type=Path)
    p.add_argument("--masked",type=Path)
    a=p.parse_args()
    verify(a.input,a.models,a.output,a.original,a.masked)
