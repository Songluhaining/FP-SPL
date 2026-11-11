"""用同一版本划分重训比较方法，仅向有监督模型提供训练折标签。"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from .folds import splits


def load_features(version,method):
    table=pd.read_csv(version/method/"features.csv")
    names=table.pop("VARIANT" if method=="CLAP" else "model").tolist()
    observations=table.pop("LABEL").tolist()
    table=table.loc[:,~table.columns.str.startswith("Unnamed:")]
    if len(table.columns)!=(6 if method=="CLAP" else 50) or not np.isfinite(table.to_numpy()).all():
        raise ValueError(f"特征列错误或非有限数值 {version}/{method}")
    return names,observations,table


def run(root,repo,method,truth,target,only=None,systems=("bank", "email"),extra_splits=(),
        partitions=None,identities=None):
    sys.path.insert(0,str(repo.resolve()))
    if method=="CLAP":
        import PassingVariants_Classification as implementation
    else:
        from fp_detection.PUBaggingLearning import PUBaggingLearning
    predictions=[]
    selected=[*splits(root,systems), *extra_splits] if partitions is None else partitions
    def identity(version):
        return identities[version] if identities is not None else (version.parent.name,version.name)
    for name,training,testing in selected:
        if only and not name.startswith(only+"-fold"):
            continue
        x,y=[],[]
        for v in training:
            ids,obs,table=load_features(v,method)
            x.append(table)
            y.extend([truth[(*identity(v),i)] for i in ids] if method=="CLAP" else obs)
        X=pd.concat(x)
        if method=="PULP":
            positive=y.count("F")
            ratio=(positive+1.e-6)/(len(y)-positive)
            model=PUBaggingLearning(sampling_ratio=ratio,random_state=42)
            model.prepare_data(X.to_numpy(),y,None)
            model.predict()
        ids_all=[]
        test_tables=[]
        for v in testing:
            ids,obs,table=load_features(v,method)
            keep=np.array([o!="F" for o in obs])
            test_tables.append(table.loc[keep])
            ids_all.extend((*identity(v),i) for i,k in zip(ids,keep) if k)
        test=pd.concat(test_tables)[X.columns].to_numpy()
        if method=="CLAP" and len(set(y))==1:
            # 原版二分类器无法在单类训练集上拟合。保留留出配置与缺失分数。
            for s,v,c in ids_all:
                predictions.append(dict(system=s,version=v,configuration=c,split=name,method=method,
                                        score=None,prediction=0,status="insufficient"))
            print(name,method,"single_training_class",len(test),flush=True)
            continue
        if method=="CLAP":
            # 发布实现只输出准确率；截取它实际训练的模型进行折外打分，替换依赖真值的打印。
            captured=[]
            def factory(**parameters):
                captured.append(SVC(**parameters))
                return captured[-1]
            implementation.svm=SimpleNamespace(SVC=factory)
            implementation.overall_performance_measurement=lambda *_: None
            implementation.classify_by_svm(io.StringIO(),None,X.to_numpy(),test,y,np.zeros(len(test)))
            model=captured[0]
            scores=model.decision_function(test)
            labels=model.predict(test)
        else:
            scores=sum(w*e.predict_proba(test)[:,1] for w,e in zip(model.estimator_weights,model.estimators))
            labels=scores>0.5
        for (s,v,c),score,label in zip(ids_all,scores,labels):
            predictions.append(dict(system=s,version=v,configuration=c,split=name,method=method,
                                    score=float(score),prediction=int(label),status="scored"))
        print(name,method,len(test),flush=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--code",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--truth",type=Path)
    p.add_argument("--method",choices=["PULP","CLAP"],required=True)
    p.add_argument("--only",choices=["bank","email"])
    a=p.parse_args()
    truth={}
    if a.method=="CLAP":
        with a.truth.open(encoding="utf8") as s:
            truth={(r["system"],r["version"],r["configuration"]):int(r["label"]) for r in csv.DictReader(s)}
    run(a.input,a.code,a.method,truth,a.output,a.only)
