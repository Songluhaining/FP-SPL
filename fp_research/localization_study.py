"""六系统语句级定位实验；原案例只读，预测修正只排除通过配置。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import pandas as pd

from .inventory import atomic_json
from . import multifault_study as study

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experimental_results/statement_localization"
SCENARIO = "statement_localization"
UPSTREAM = "62b413b52bbfba1f86a244ddb9f1daed232e329e"
METRICS = ["Tarantula", "Ochiai", "Op2", "Barinel", "Dstar", "Jaccard"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    locations = read(ROOT / "fp_research/dataset_locations.json")["systems"]
    jobs = []
    for system, location in locations.items():
        base = Path(location)
        for stratum in ([1] if system in {"gpl", "zipme"} else [1, 2, 3]):
            layer = base / f"4wise-{base.name}-{stratum}BUG-Full"
            versions = sorted(layer.glob("_MultipleBugs_*"))
            if not versions:
                raise ValueError(f"找不到案例 {layer}")
            for version in versions:
                with (version / "config.report.csv.done").open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.reader(stream))[1:]
                states = [row[-1].strip() for row in rows]
                if set(states) - {"__PASSED__", "__FAILED__"}:
                    raise ValueError(f"配置状态无效 {version}")
                jobs.append(dict(system=system, stratum=stratum, version=version.name,
                                 path=str(version), configurations=len(rows),
                                 candidates=states.count("__PASSED__"),
                                 report_sha256=digest(version / "config.report.csv.done")))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(jobs=jobs, varcop_commit=UPSTREAM, primary_metric="Ochiai",
                    additional_metrics=[m for m in METRICS if m != "Ochiai"],
                    correction="排除检测为假通过的原本通过配置；保留真实可见失败配置及其测试",
                    aggregation="AGGREGATION_ARITHMETIC_MEAN", normalization="ENABLE_NORMALIZATION",
                    alpha=.5, coverage_filter=0.0,
                    methods=["original", "joint", "CLAP", "PULP", "PULP-default"],
                    joint_parameters=study.PARAMETERS, seeds=list(study.SEEDS),
                    partitions="各系统各缺陷数按版本固定五折；同一版本的配置整体留出",
                    metrics="最坏并列名次；每故障位置明细；Hit@1/3/5/10、EXAM、首故障及全部故障检查量",
                    note="不进行源码修复或将预测当作真实失败测试。固定全系统语句分母；漏出排名的位置保留未定位状态。",
                    deferred=["GPL-2fault", "GPL-3fault", "ZipMe-2fault", "ZipMe-3fault"])
    path = OUTPUT / "manifest.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("实验协议已经登记，须检查差异后才能改变")
    atomic_json(path, protocol)
    return jobs


def work_folder(job):
    return Path(job["path"]) / ".fp_research" / SCENARIO


def inference_folder(job):
    old = Path(job["path"]) / ".fp_research/multifault_joint"
    if job["stratum"] > 1 and job["system"] in {"elevator", "email", "examdb"}:
        return old
    return work_folder(job)


def feature_folder(job):
    legacy = (ROOT / "result/fp" / job["system"] / job["version"] if job["system"] in {"bank", "email"}
              else Path("D:/dataset/whn/fp_extension_work/main") / job["system"] / job["version"])
    for folder in (Path(job["path"]) / ".fp_research/multifault_joint",
                   Path(job["path"]) / ".fp_research/main",
                   legacy if job["stratum"] == 1 else work_folder(job), work_folder(job)):
        if all((folder / m / "features.csv").exists() for m in ("CLAP", "PULP")):
            return folder
    return work_folder(job)


def fit(job):
    from .joint_inputs import verify_unchanged
    folder = inference_folder(job)
    if folder.name == "multifault_joint":
        receipt = read(folder / "joint.json")
        if receipt["code"] != study.code_hash("joint"):
            raise ValueError("此前联合推断的实现与当前不一致")
        info = read(folder / "evidence.json")
        verify_unchanged(Path(job["path"]), info)
        for name, value in receipt["artifacts"].items():
            if digest(folder / name) != value:
                raise ValueError("此前推断输出发生变化")
        return "verified previous joint inference"
    study.SCENARIO = SCENARIO
    return study.fit(job)


def prepare(job):
    from .compare import load_features
    folder = feature_folder(job)
    if folder != work_folder(job):
        for method in ("CLAP", "PULP"):
            names, labels, table = load_features(folder, method)
            if not names or len(names) != len(set(names)):
                raise ValueError("已有基线特征缺失或重复")
        return "reuse features; original input digest checked at localization"
    study.SCENARIO = SCENARIO
    from . import multifault_cache
    from .localization_cache import seed_with_process_lock
    multifault_cache.seed = seed_with_process_lock
    return study.prepare(job)


def perform(job, stage):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    out = work_folder(job)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        if stage == "fit":
            value = fit(job)
        elif stage == "prepare":
            value = prepare(job)
        elif stage == "rank":
            from .varcop_localization import rank_case
            value = rank_case(job)
        else:
            raise ValueError(stage)
        atomic_json(out / f"{stage}-status.json",
                    dict(ok=True, result=value, seconds=time.monotonic()-start))
        print(job["system"], job["stratum"], job["version"], stage, value, flush=True)
        return True
    except Exception as error:
        atomic_json(out / f"{stage}-status.json",
                    dict(ok=False, type=type(error).__name__, reason=str(error),
                         traceback=traceback.format_exc(), seconds=time.monotonic()-start))
        print("FAILED", job["system"], job["stratum"], job["version"], stage, repr(error), flush=True)
        return False


def classify(jobs, method, original_default=False):
    from .truth import ground_truth
    from .compare import run, load_features
    mapping = {}
    for system, stratum in sorted({(j["system"], j["stratum"]) for j in jobs}):
        first = next(j for j in jobs if (j["system"], j["stratum"]) == (system, stratum))
        layer = Path(first["path"]).parent
        for row in ground_truth(layer.parent, {system: (layer.name,)}):
            mapping[(system, row["version"], row["configuration"])] = row["label"]
    identities = {feature_folder(j): (j["system"], j["version"]) for j in jobs}
    parts = []
    for system, stratum in sorted({(j["system"], j["stratum"]) for j in jobs}):
        selected = [j for j in jobs if (j["system"], j["stratum"]) == (system, stratum)]
        selected.sort(key=lambda j: hashlib.sha256(("42/"+system+"/"+j["version"]).encode()).digest())
        ordered = [feature_folder(j) for j in selected]
        for fold in range(5):
            testing = ordered[fold::5]
            parts.append((f"{system}-{stratum}fault-fold{fold}",
                          [p for p in ordered if p not in testing], testing))
    name = "PULP-default" if original_default else method
    receipts = []
    directory = OUTPUT / "fold_predictions"
    directory.mkdir(exist_ok=True)
    for split, training, testing in parts:
        inputs = {}
        tables = {}
        for folder in training + testing:
            inputs[str(folder / method / "features.csv")] = digest(folder / method / "features.csv")
            names, observations, table = load_features(folder, method)
            tables[str(folder)] = dict(rows=len(names), columns=list(table.columns),
                order_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest(),
                visible_failures=observations.count("F"))
        positives = sum(tables[str(p)]["visible_failures"] for p in training)
        total = sum(tables[str(p)]["rows"] for p in training)
        ratio = (.05 if original_default else (positives+1e-6)/(total-positives)) if method == "PULP" else None
        classifier_sources = (["PassingVariants_Classification.py"] if method == "CLAP" else
                              ["fp_detection/PUBaggingLearning.py", "fp_detection/BaseClassifier.py"])
        spec = dict(split=split, training=[identities[p] for p in training],
                    testing=[identities[p] for p in testing], input_files=inputs,
                    method=name, sampling_ratio=ratio,
                    proxy_negatives_per_estimator=int((total-positives)*ratio) if ratio is not None else None,
                    estimators=30 if method == "PULP" else 1, tables=tables,
                    classifier_sources={n: digest(ROOT.parent / method / n) for n in classifier_sources},
                    wrapper_sha256=digest(ROOT / "fp_research/compare.py"))
        path = directory / f"{name}-{split}.csv"
        proof = path.with_suffix(".json")
        if not (proof.exists() and path.exists() and
                read(proof).get("protocol") == spec and read(proof).get("sha256") == digest(path)):
            run(ROOT, ROOT.parent / method, method, mapping, path, partitions=[(split, training, testing)],
                identities=identities, pu_sampling_ratio=.05 if original_default else None)
            atomic_json(proof, dict(protocol=spec, sha256=digest(path)))
        receipts.append(read(proof))
    all_rows = pd.concat([pd.read_csv(directory / f"{name}-{p[0]}.csv") for p in parts], ignore_index=True)
    if all_rows.duplicated(["system", "version", "configuration"]).any():
        raise ValueError("基线折外预测重复")
    all_rows.to_csv(OUTPUT / f"{name}.csv", index=False)
    atomic_json(OUTPUT / f"{name}-training.json", receipts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["manifest", "fit", "prepare", "classify", "rank", "evaluate"])
    p.add_argument("--system")
    p.add_argument("--stratum", type=int, choices=[1, 2, 3])
    p.add_argument("--version")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--method", choices=["CLAP", "PULP"])
    p.add_argument("--original-default", action="store_true")
    args = p.parse_args()
    jobs = manifest()
    selected = [j for j in jobs if (not args.system or j["system"] == args.system)
                and (not args.stratum or j["stratum"] == args.stratum)
                and (not args.version or j["version"] == args.version)]
    if not selected:
        p.error("所选系统、缺陷层或版本没有匹配案例")
    if args.stage == "manifest":
        print(len(jobs), sum(j["configurations"] for j in jobs), sum(j["candidates"] for j in jobs))
    elif args.stage == "classify":
        if not args.method:
            p.error("分类必须指定方法")
        classify(selected, args.method, args.original_default)
    elif args.stage == "evaluate":
        from .localization_evaluate import evaluate
        evaluate(jobs)
    else:
        failures = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(perform, j, args.stage) for j in selected]
            for future in as_completed(futures):
                failures += not future.result()
        if failures:
            raise SystemExit(f"{failures} cases failed; inspect per-case status")


if __name__ == "__main__":
    main()
