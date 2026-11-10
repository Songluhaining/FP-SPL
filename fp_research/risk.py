"""合并配置适用证据和条件检出能力，报告最高风险组和未评分原因。"""
import argparse
import csv
import json
from pathlib import Path

from .applicability import applicability,build_evidence
from .folds import splits
from .inventory import SYSTEMS


def combine(applicable,detection):
    if not 0<=applicable<=1+1e-9 or not 0<=detection<=1:
        raise ValueError("分数范围异常")
    return min(1.0,applicable)*(1-detection)


def score_version(version,source,model_dir,split):
    rows=json.loads((version/"inventory.json").read_text(encoding="utf8"))
    semantics=json.loads((version/"semantics.json").read_text(encoding="utf8"))
    estimates=json.loads((model_dir/(version.name+".json")).read_text(encoding="utf8"))
    graphs={}
    errors={}
    for group in semantics["groups"]:
        path=version/"graphs"/(group+".json")
        if not path.is_file():
            raise FileNotFoundError(f"本版本尚未完成组推理：{path}")
        graph=json.loads(path.read_text(encoding="utf8"))
        if graph["status"]=="ok":
            graphs[group]=graph["scores"]
        else:
            errors[group]=graph["status"]
    output=[]
    for r in rows:
        if r["outcome"]!="passed":
            continue
        cid=r["configuration"]
        present,executed=build_evidence(source/"variants"/cid)
        best={}
        for group,scores in graphs.items():
            evidence=applicability(scores,r["features"],present,executed)
            if evidence is None:
                continue
            a=evidence["score"]
            neural,ablated,freq=estimates[cid][group]
            options={"full":combine(a,neural),"no_config":combine(1,neural),
                     "no_assertion":combine(a,ablated),"frequency":combine(a,freq)}
            for method,score in options.items():
                if method not in best or (score,group)>(best[method][0],best[method][1]):
                    best[method]=(score,group,evidence)
        for method in ("full","no_config","no_assertion","frequency"):
            score,group,evidence=best.get(method,(None,None,{}))
            status="insufficient" if score is None else "partial" if errors else "scored"
            output.append(dict(system=version.parent.name,version=version.name,configuration=cid,split=split,
                          method=method,score=score,prediction=int(score>=0.5) if score is not None else 0,
                          status=status,group=group,usable_groups=len(graphs),failed_groups=len(errors),
                          evidence=json.dumps(evidence,ensure_ascii=False)))
    return output


def save_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf8") as s:
        writer=csv.DictWriter(s,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("input","data","models","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--only",choices=SYSTEMS)
    a=p.parse_args()
    rows=[]
    for name,training,testing in splits(a.input):
        if a.only and not name.startswith(a.only+"-fold"):
            continue
        for v in testing:
            rows.extend(score_version(v,a.data.joinpath(*SYSTEMS[v.parent.name],v.name),a.models/name,name))
    save_rows(a.output,rows)
