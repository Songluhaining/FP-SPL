"""核对全部基线可见标签，并在六系统各一案例重新生成特征比较。"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from fp_research.localization_study import ROOT, OUTPUT, manifest, feature_folder, work_folder, digest
from fp_research.compare import load_features
from fp_research.inventory import atomic_json


def representative(system):
    return next(j for j in manifest() if j["system"] == system and j["stratum"] == 1)


def worker(system, method):
    from fp_research.baseline_features import worker as calculate
    job = representative(system)
    version, saved = Path(job["path"]), feature_folder(job)
    slices = (ROOT.parent / ("pulp_trial" if system == "bank" else "email_trial") / "dataset" / job["version"]
              if system in {"bank", "email"} else saved / "slices")
    target = work_folder(job) / "feature-reference" / method
    calculate((ROOT.parent / method, method, version, slices, target), optimize=True)
    a_ids, a_labels, a = load_features(saved, method)
    b_ids, b_labels, b = load_features(target.parent, method)
    left = pd.DataFrame(a.to_numpy(), index=a_ids, columns=a.columns).sort_index()
    right = pd.DataFrame(b.to_numpy(), index=b_ids, columns=b.columns).sort_index()
    assert left.index.equals(right.index) and left.columns.equals(right.columns)
    assert dict(zip(a_ids, a_labels)) == dict(zip(b_ids, b_labels))
    assert np.allclose(left.to_numpy(), right.to_numpy(), atol=1e-12, rtol=1e-10)
    atomic_json(target / "verified.json", dict(system=system, method=method, version=job["version"],
        source_features=str(saved / method / "features.csv"),
        source_sha256=digest(saved / method / "features.csv"), rebuilt_sha256=digest(target / "features.csv"),
        rows=len(left), columns=len(left.columns), maximum_difference=float(np.max(np.abs(left-right).to_numpy()))))


def launch(item):
    system, method = item
    target = work_folder(representative(system)) / "feature-reference"
    target.mkdir(parents=True, exist_ok=True)
    with (target / (method+"-run.log")).open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", __spec__.name, "--worker", system, "--method", method],
                       cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, check=True)
    return system, method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--method")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.method)
    checked = []
    for job in manifest():
        with (Path(job["path"]) / "config.report.csv.done").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))[1:]
        visible = {r[0].strip(): "F" if r[-1].strip() == "__FAILED__" else "TP" for r in rows}
        folder = feature_folder(job)
        for method in ("CLAP", "PULP"):
            ids, labels, values = load_features(folder, method)
            expected = {c: s for c, s in visible.items() if method == "PULP" or s != "F"}
            assert len(ids) == len(expected) and dict(zip(ids, labels)) == expected
            checked.append(dict(system=job["system"], version=job["version"], method=method,
                                path=str(folder / method / "features.csv"), rows=len(ids),
                                sha256=digest(folder / method / "features.csv")))
    atomic_json(OUTPUT / "feature-cohorts.json", checked)
    tasks = [(s, m) for s in ["bank", "elevator", "email", "examdb", "gpl", "zipme"] for m in ("CLAP", "PULP")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(launch, tasks):
            print("特征重算一致", *result, flush=True)
    atomic_json(OUTPUT / "feature-reference.json", [
        __import__("json").loads((work_folder(representative(s)) / "feature-reference" / m / "verified.json").read_text(encoding="utf-8"))
        for s, m in tasks])


if __name__ == "__main__":
    main()
