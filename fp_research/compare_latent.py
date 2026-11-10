"""在同一批配置上对齐固定推断与既有折外预测，并按缺陷版本配对重采样。"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .report import metrics


KEYS = ["system","version","configuration"]


def paired_difference(frame, baseline):
    cells = []
    for _, version in frame.groupby("version"):
        counts = []
        for method in ("statement", baseline):
            g = version[version.method == method]
            y, pred = g.label.to_numpy(), g.prediction.to_numpy()
            counts.append([sum((y == 1) & (pred == 1)), sum(y == 1), sum(pred == 1)])
        cells.append(counts)
    cells = np.asarray(cells)
    rng = np.random.default_rng(2026)
    totals = cells[rng.integers(len(cells), size=(10000,len(cells)))].sum(axis=1)
    f1 = 2 * totals[:,:,0] / np.maximum(1,totals[:,:,1]+totals[:,:,2])
    return np.quantile(f1[:,0]-f1[:,1],[0.025,0.975])


def evaluate(prior, working, output):
    output.mkdir(parents=True,exist_ok=True)
    reports, intervals, predictions, evidence = [], [], [], []
    transfer, replay = [], []
    for scenario in ("main", "masked"):
        available = pd.read_csv(prior/scenario/"predictions.csv", low_memory=False)
        ref = available[available.split.str.contains("-fold") & available.method.isin(["CLAP","PULP","full"])].copy()
        columns = KEYS + ["split","label","base","executed_label"]
        if scenario == "masked":
            columns.append("converted")
        truth = ref[columns].drop_duplicates()
        if truth.duplicated(KEYS).any():
            raise ValueError("同一配置的评价标签或划分不一致")
        if set(ref.method) != {"CLAP","PULP","full"}:
            raise ValueError("缺少基准或旧方案")
        for method, part in ref.groupby("method"):
            check = truth.merge(part[KEYS], on=KEYS, how="outer", indicator=True, validate="one_to_one")
            if not (check._merge=="both").all():
                raise ValueError("折外基准候选清单不一致："+method)
        model = pd.read_csv(working/(scenario+".csv"))
        raw = pd.read_csv(working/("raw-"+scenario)/"predictions.csv")
        pd.testing.assert_frame_equal(model,raw,check_exact=True)
        for method, part in model.groupby("method"):
            check = truth.merge(part[KEYS],on=KEYS,how="outer",indicator=True,validate="one_to_one")
            if not (check._merge=="both").all():
                raise ValueError("候选集不一致："+method)
        model = model.merge(truth,on=KEYS,validate="many_to_one")
        model["status"] = "scored"
        frame = pd.concat([ref,model],ignore_index=True)
        predictions.append(model.assign(scenario=scenario))
        evidence.append(pd.read_csv(working/("raw-"+scenario)/"audit.csv").assign(scenario=scenario))
        trans = available[available.split.str.contains("-to-") & available.method.isin(["CLAP","PULP","full"])]
        for method, part in trans.groupby("method"):
            test = truth.merge(part[KEYS], on=KEYS, how="outer", indicator=True, validate="one_to_one")
            if not (test._merge=="both").all():
                raise ValueError(f"跨系统比较对象不齐 {method}")
        for (system, method), part in pd.concat([trans,model]).groupby(["system","method"]):
            transfer.append(dict(scenario=scenario,system=system,method=method,**metrics(part)))
        if scenario=="main":
            probes=pd.read_csv(prior/"main"/"replay-outcomes.csv", usecols=KEYS+["status"])
            confirmed=probes.loc[probes.status=="confirmed",KEYS].drop_duplicates()
            for (system, method), part in frame.merge(confirmed,on=KEYS,validate="many_to_one").groupby(["system","method"]):
                replay.append(dict(system=system,method=method,confirmed=len(part),detected=int(part.prediction.sum())))
        for system, group in frame.groupby("system"):
            for subset,subset_rows in [("all", group),("base",group[group.base==1]),("other",group[group.base==0])]:
                for method, part in subset_rows.groupby("method"):
                    reports.append(dict(scenario=scenario,system=system,method=method,subset=subset,**metrics(part)))
            if scenario=="main":
                for method,part in group.groupby("method"):
                    reports.append(dict(scenario=scenario,system=system,method=method,subset="positive_execution",
                                        **metrics(part,"executed_label")))
            for baseline in ("CLAP","PULP"):
                low,high=paired_difference(group,baseline)
                diff=metrics(group[group.method=="statement"])["f1"]-metrics(group[group.method==baseline])["f1"]
                intervals.append(dict(scenario=scenario,system=system,baseline=baseline,
                                      difference=diff,lower=low,upper=high,versions=group.version.nunique()))
    pd.concat(predictions).to_csv(output/"predictions.csv",index=False)
    pd.concat(evidence).to_csv(output/"coverage-audit.csv",index=False)
    pd.DataFrame(reports).to_csv(output/"metrics.csv",index=False)
    pd.DataFrame(transfer).to_csv(output/"transfer-comparison.csv",index=False)
    pd.DataFrame(replay).to_csv(output/"replay-positive-check.csv",index=False)
    pd.DataFrame(intervals).to_csv(output/"paired-differences.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("baseline","working","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    evaluate(a.baseline,a.working,a.output)
