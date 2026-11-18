"""从原始报告、后验样本和完整语句排名独立重算定位统计。"""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fp_research.inventory import atomic_json
from fp_research.localization_study import (
    ROOT, OUTPUT, manifest, work_folder, inference_folder, feature_folder, read, digest,
)
from fp_research.varcop_localization import source_hashes, input_fingerprint


def positions(sequence):
    result = {}
    start = 0
    while start < len(sequence):
        end = start+1
        while end < len(sequence) and sequence[end][1:] == sequence[start][1:]:
            end += 1
        for item in sequence[start:end]:
            if item[0] in result:
                raise ValueError("排名中有重复语句")
            result[item[0]] = end
        start = end
    return result


def inference(job):
    folder = inference_folder(job)
    metadata = read(folder / "evidence.json")
    proof = read(folder / "joint.json")
    for name, sha in proof["artifacts"].items():
        if digest(folder / name) != sha:
            raise ValueError("联合预测输出摘要改变")
    with np.load(folder / "evidence.npz") as evidence, np.load(folder / "posterior-states.npz") as archive:
        states = [archive[key] for key in archive.files]
        saved = pd.read_csv(folder / "joint.csv").set_index("configuration")
        for key, field in (("score", "membership"), ("lower", "lower_membership"), ("upper", "upper_membership")):
            for cid, row in saved.iterrows():
                index = metadata["configurations"].index(cid)
                member = evidence[field][index]
                # 对每一粒子的实际故障组相乘，然后平均；不调用模型风险函数。
                values = [np.mean([1-math.prod(1-float(member[g]) for g in np.flatnonzero(state))
                                   for state in population]) for population in states]
                if not math.isclose(float(row[key]), float(np.mean(values)), abs_tol=1e-12):
                    raise ValueError("后验风险独立计算不同")
    return saved


def folds(jobs, tables):
    groups = {}
    for j in jobs:
        groups.setdefault((j["system"], j["stratum"]), []).append(j)
    expected = {}
    for (system, stratum), selected in groups.items():
        selected.sort(key=lambda j: hashlib.sha256(("42/"+system+"/"+j["version"]).encode()).digest())
        for fold in range(5):
            testing = {(j["system"], j["version"]) for j in selected[fold::5]}
            training = {(j["system"], j["version"]) for j in selected} - testing
            expected[f"{system}-{stratum}fault-fold{fold}"] = (training, testing)
    for method in ("CLAP", "PULP", "PULP-default"):
        proofs = read(OUTPUT / f"{method}-training.json")
        if len(proofs) != len(expected):
            raise ValueError("基线折数错误")
        for proof in proofs:
            spec = proof["protocol"]
            train, test = expected[spec["split"]]
            if set(map(tuple, spec["training"])) != train or set(map(tuple, spec["testing"])) != test or train & test:
                raise ValueError("基线版本分折不一致")
            path = OUTPUT / "fold_predictions" / f"{method}-{spec['split']}.csv"
            if digest(path) != proof["sha256"]:
                raise ValueError("折预测改变")
            for name, sha in spec["input_files"].items():
                if digest(Path(name)) != sha:
                    raise ValueError("训练或评估输入改变")
            if method != "CLAP":
                counted = list(spec["tables"].items())
                train_folders = {str(feature_folder(j)) for j in jobs if (j["system"], j["version"]) in train}
                total = sum(d["rows"] for p, d in counted if p in train_folders)
                positive = sum(d["visible_failures"] for p, d in counted if p in train_folders)
                unlabeled = total-positive
                ratio = .05 if method == "PULP-default" else (positive+1e-6)/unlabeled
                assert math.isclose(spec["sampling_ratio"], ratio)
                assert spec["proxy_negatives_per_estimator"] == int(unlabeled*ratio)
        frame = tables[method]
        for split, sub in frame.groupby("split"):
            if set(zip(sub.system, sub.version)) != expected[split][1]:
                raise ValueError("折预测归属不一致")


def main():
    jobs = manifest()
    detailed = pd.read_csv(OUTPUT / "fault_positions.csv")
    case_table = pd.read_csv(OUTPUT / "cases.csv")
    metrics = pd.read_csv(OUTPUT / "metrics.csv")
    detection = {m: pd.read_csv(OUTPUT / f"{m}.csv") for m in ("CLAP", "PULP", "PULP-default")}
    folds(jobs, detection)
    indexed = detailed.set_index(["system", "stratum", "version", "method", "metric", "statement"])
    if not indexed.index.is_unique:
        raise ValueError("故障位置评价行重复")
    visited = set()
    predictions = []
    source = source_hashes()
    for job in jobs:
        v = Path(job["path"])
        receipt = read(work_folder(job) / "varcop/complete.json")
        if receipt["protocol"]["source"] != source:
            raise ValueError("定位计算源码版本不一致")
        if input_fingerprint(v)[0] != receipt["protocol"]["input"]:
            raise ValueError("原始定位输入改变")
        joint = inference(job)
        with (v / "config.report.csv.done").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header, original = next(reader), list(reader)
        passed = {r[0].strip() for r in original if r[-1].strip() == "__PASSED__"}
        raw_faults = []
        for text in (v / (v.name+".mutant.log")).read_text(encoding="utf-8").splitlines():
            left, number, _ = text.split(":", 2)
            raw_faults.append(left.rsplit(".", 1)[0]+"."+number)
        faults = set(raw_faults)
        expected_excluded = {"original": set()}
        for method in ["joint", "CLAP", "PULP", "PULP-default"]:
            frame = joint.reset_index() if method == "joint" else detection[method]
            if method != "joint":
                frame = frame[(frame.system == job["system"]) & (frame.version == job["version"])]
            if set(frame.configuration) != passed or frame.configuration.duplicated().any():
                raise ValueError("识别方法候选分母不一致")
            expected_excluded[method] = set(frame.loc[frame.prediction == 1, "configuration"])
            for row in original:
                cid = row[0].strip()
                if cid not in passed:
                    continue
                enabled = {f for f, flag in zip(header[1:-1], row[1:-1]) if flag.strip() == "T"}
                present, executed = set(), set()
                path = v / "variants" / cid / "coverage/spectrum_passed_coverage.xml"
                import xml.etree.ElementTree as ET
                for line in ET.parse(path).iterfind(".//package/file/line"):
                    feature = line.get("featureClass", "")
                    if feature.split(".")[0] not in enabled:
                        continue
                    key = feature+"."+str(line.get("featureLineNum"))
                    present.add(key)
                    if any(int(line.get(a, 0)) > 0 for a in ("count", "truecount", "falsecount")):
                        executed.add(key)
                prediction = frame.loc[frame.configuration == cid].iloc[0]
                predictions.append(dict(system=job["system"], stratum=job["stratum"], version=job["version"],
                    configuration=cid, method=method, score=prediction["score"],
                    prediction=int(prediction["prediction"]), label=int(bool(faults & present)),
                    executed_label=int(bool(faults & executed))))
        if receipt["protocol"]["methods"] != {m: sorted(x) for m, x in expected_excluded.items()}:
            raise ValueError("定位使用的排除集合与预测不同")
        universe = len(receipt["universe"])
        for method, entry in receipt["methods"].items():
            alias = entry.get("reused", method)
            folder = work_folder(job) / "varcop" / alias
            with (folder / "observations.csv").open(encoding="utf-8-sig", newline="") as stream:
                retained = list(csv.reader(stream))[1:]
            assert {r[0].strip() for r in retained} == {r[0].strip() for r in original} - expected_excluded[method]
            with gzip.open(folder / "rankings.json.gz", "rt", encoding="utf-8") as stream:
                ranks = json.load(stream)
            for metric, lists in ranks.items():
                focus = positions(lists["isolated"])
                full = positions(lists["full"])
                removed_prefix = [0]
                for row in lists["full"]:
                    removed_prefix.append(removed_prefix[-1] + int(row[0] in focus))
                rest = {s: len(focus)+rank-removed_prefix[rank]
                        for s, rank in full.items() if s not in focus}
                for statement in faults:
                    expected = focus.get(statement, rest.get(statement)) if receipt["variability_case"] else full.get(statement)
                    key = (job["system"], job["stratum"], job["version"], method, metric, statement)
                    r = indexed.loc[key]
                    visited.add(key)
                    assert int(r.found) == (expected is not None)
                    if expected is not None:
                        assert expected == r["rank"], (key, expected, r["rank"])
                    penalty = expected if expected is not None else universe+1
                    assert penalty == r.penalized_rank
                    assert math.isclose(r.exam, min(penalty, universe)/universe)
        if len(visited) % 5000 < 100:
            print("已独立核验", job["system"], job["version"], len(visited), flush=True)
    if visited != set(indexed.index):
        raise ValueError("定位明细有遗漏或多余")
    for r in metrics.to_dict("records"):
        sub = detailed[(detailed.system == r["system"]) & (detailed.stratum == r["stratum"]) &
                       (detailed.method == r["method"]) & (detailed.metric == r["metric"])]
        assert len(sub) == r["fault_positions"]
        assert sub.version.nunique() == r["cases"]
        assert (sub.found == 0).sum() == r["unranked"]
        assert math.isclose(sub.penalized_rank.sum()/len(sub), r["mean_rank"])
        assert math.isclose(sub.exam.sum()/len(sub), r["mean_exam"])
        original_sub = detailed[(detailed.system == r["system"]) & (detailed.stratum == r["stratum"]) &
                                  (detailed.method == "original") & (detailed.metric == r["metric"])]
        left = sub.set_index(["version", "statement"])
        right = original_sub.set_index(["version", "statement"])
        improvement = right.penalized_rank - left.penalized_rank
        assert (improvement > 0).sum() == r["improved"]
        assert (improvement == 0).sum() == r["tied"]
        assert (improvement < 0).sum() == r["worsened"]
        first, all_ranks = [], []
        for version, group in sub.groupby("version"):
            first.append(group.penalized_rank.min())
            all_ranks.append(group.penalized_rank.max())
        assert math.isclose(np.mean(first), r["first_rank"])
        assert math.isclose(np.mean(all_ranks), r["all_rank"])
        for budget in [1, 3, 5, 10]:
            assert ((sub.found == 1) & (sub["rank"] <= budget)).sum() == r["hit"+str(budget)]
            any_hits = sum(((g.found == 1) & (g["rank"] <= budget)).any() for _, g in sub.groupby("version"))
            all_hits = sum(((g.found == 1) & (g["rank"] <= budget)).all() for _, g in sub.groupby("version"))
            assert any_hits == r["cases_hit"+str(budget)]
            assert all_hits == r["cases_all"+str(budget)]
    detections = pd.DataFrame(predictions)
    if len(detections) != 4*sum(j["candidates"] for j in jobs):
        raise ValueError("配置明细分母不完整")
    detections.to_csv(OUTPUT / "fp_predictions.csv", index=False)
    tables = []
    for (system, stratum, method), sub in detections.groupby(["system", "stratum", "method"]):
        tp = int(((sub.label == 1) & (sub.prediction == 1)).sum())
        fp = int(((sub.label == 0) & (sub.prediction == 1)).sum())
        tn = int(((sub.label == 0) & (sub.prediction == 0)).sum())
        fn = int(((sub.label == 1) & (sub.prediction == 0)).sum())
        scored = int(sub.score.notna().sum())
        tables.append(dict(system=system, stratum=int(stratum), method=method, n=len(sub), scored=scored,
            insufficient=len(sub)-scored, tp=tp, fp=fp, tn=tn, fn=fn,
            precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn and scored else None,
            f1=2*tp/(2*tp+fp+fn) if tp+fn and scored else None,
            accuracy=(tp+tn)/len(sub) if scored else None))
    pd.DataFrame(tables).to_csv(OUTPUT / "fp_metrics.csv", index=False)
    atomic_json(OUTPUT / "verified.json", dict(cases=len(jobs), fault_rows=len(detailed),
        prediction_rows=len(detections), configurations=sum(j["configurations"] for j in jobs),
        independent_particle_risks=True, independent_rankings=True, independent_statistics=True,
        raw_inputs_unchanged=True, folds_disjoint=True,
        files={p.name: digest(p) for p in (OUTPUT / "fault_positions.csv", OUTPUT / "metrics.csv",
                                         OUTPUT / "fp_predictions.csv", OUTPUT / "fp_metrics.csv")}))
    print("完整独立核验通过", len(jobs), len(detailed), len(detections), flush=True)


if __name__ == "__main__":
    main()
