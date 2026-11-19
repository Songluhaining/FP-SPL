"""独立读取修正报告及保存排名，重算各主表并核查配置无剔除。"""
import csv
import gzip
import json
import math
from pathlib import Path
import shutil

import pandas as pd

from fp_research import varcop_report_relabel as implementation
from fp_research.localization_study import read, digest
from fp_research.inventory import atomic_json


def rank_map(sequence):
    result = {}
    start = 0
    while start < len(sequence):
        end = start+1
        while end < len(sequence) and sequence[end][1:] == sequence[start][1:]:
            end += 1
        for row in sequence[start:end]:
            assert row[0] not in result
            result[row[0]] = end
        start = end
    return result


def main():
    output = implementation.OUTPUT
    jobs = read(output / "manifest.json")["jobs"]
    frame = pd.read_csv(output / "fault_positions.csv")
    indexed = frame.set_index(["system", "stratum", "version", "method", "metric", "statement"])
    assert indexed.index.is_unique
    examined = set()
    report_rows, changed = 0, {}
    archive = output / "case_receipts"
    original_frame = pd.read_csv(implementation.PREVIOUS_OUTPUT / "fault_positions.csv")
    for job in jobs:
        version, folder = Path(job["path"]), implementation.folder(job)
        proof = read(folder / "complete.json")
        assert proof["protocol"]["source"] == implementation.sources()
        assert proof["original_rankings_reproduced"] and proof["original_inputs_unchanged"]
        if "runtime_optimization" in proof:
            optimization = proof["runtime_optimization"]
            assert optimization["every_index_checked_against_original"]
            assert optimization["source_sha256"] == digest(implementation.ROOT / "fp_research/varcop_score_lookup.py")
        assert implementation.legacy.input_fingerprint(version)[0] == proof["protocol"]["input"]
        for path, sha in proof["protocol"]["original_reports"].items():
            assert digest(Path(path)) == sha
        for path, sha in proof["files"].items():
            assert digest(folder / path) == sha
        for reference in proof["protocol"]["references"].values():
            assert digest(Path(reference["path"])) == reference["sha256"]
        with implementation.report_source(version).open(encoding="utf-8-sig", newline="") as stream:
            raw = list(csv.reader(stream))
        assert len(raw)-1 == job["configurations"]
        positions = set()
        for line in (version / (job["version"]+".mutant.log")).read_text(encoding="utf-8").splitlines():
            location, number, _ = line.split(":", 2)
            positions.add(location.rsplit(".", 1)[0]+"."+number)
        for method, info in proof["methods"].items():
            ids = set(proof["protocol"]["methods"][method])
            if method == "original":
                assert not ids
            else:
                predictions = pd.read_csv(proof["protocol"]["references"][method]["path"])
                if method != "joint":
                    predictions = predictions[(predictions.system == job["system"]) & (predictions.version == job["version"])]
                passed_ids = {row[0] for row in raw[1:] if row[-1] == "__PASSED__"}
                assert set(predictions.configuration) == passed_ids and not predictions.configuration.duplicated().any()
                assert ids == set(predictions.loc[predictions.prediction == 1, "configuration"])
            with (folder / method / "config.report.csv").open(encoding="utf-8-sig", newline="") as stream:
                corrected = list(csv.reader(stream))
            assert raw[0] == corrected[0] and len(raw) == len(corrected)
            actually_changed = set()
            for before, after in zip(raw[1:], corrected[1:]):
                assert before[:-1] == after[:-1]
                if before[-1] != after[-1]:
                    assert before[-1] == "__PASSED__" and after[-1] == "__FAILED__"
                    actually_changed.add(before[0])
            assert actually_changed == ids
            assert info["removed"] == 0 and info["retained"] == job["configurations"]
            report_rows += len(corrected)-1
            changed[method] = changed.get(method, 0)+len(ids)
            with gzip.open(folder / method / "rankings.json.gz", "rt", encoding="utf-8") as stream:
                rankings = json.load(stream)
            for metric, ranks in rankings.items():
                full, focus = rank_map(ranks["full"]), rank_map(ranks["isolated"])
                for statement in positions:
                    key = (job["system"], job["stratum"], job["version"], method, metric, statement)
                    row = indexed.loc[key]
                    isolated = focus.get(statement)
                    if isolated is None and statement in full:
                        pos = full[statement]
                        isolated = len(focus)+pos-sum(r[0] in focus for r in ranks["full"][:pos])
                    value = isolated if info["variability_case"] else full.get(statement)
                    penalty = value if value is not None else len(proof["universe"])+1
                    assert row.penalized_rank == penalty
                    assert row.denominator == len(proof["universe"])
                    assert math.isclose(row.exam, min(penalty, row.denominator)/row.denominator, abs_tol=1e-14)
                    assert row.removed == 0
                    examined.add(key)
        target = archive / job["system"] / job["version"]
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "complete.json", target / "complete.json")
    assert len(examined) == len(frame) == 61410
    assert report_rows == 162470
    assert frame.groupby('method').size().nunique() == 1
    assert len(jobs) == 1054
    keys = ["system", "stratum", "version", "method", "metric", "statement"]
    original = frame[frame.method == "original"].merge(original_frame[original_frame.method == "original"], on=keys, validate="one_to_one", suffixes=("_new", "_old"))
    assert len(original) == 12282 and (original.penalized_rank_new == original.penalized_rank_old).all()
    for name, group_fields in (("metrics.csv", ["system", "stratum", "method", "metric"]),
                               ("overall_metrics.csv", ["method", "metric"])):
        saved = pd.read_csv(output / name).set_index(group_fields)
        reconstructed = []
        for identity, group in frame.groupby(group_fields):
            row = dict(zip(group_fields, identity))
            row.update(cases=len(group[["system", "version"]].drop_duplicates()), positions=len(group),
                mean_rank=float(group.penalized_rank.sum()/len(group)), mean_exam=float(group.exam.sum()/len(group)),
                improved=int((group.original_rank > group.penalized_rank).sum()),
                tied=int((group.original_rank == group.penalized_rank).sum()),
                worsened=int((group.original_rank < group.penalized_rank).sum()), unranked=int((group.found == 0).sum()))
            for budget in (1, 3, 5, 10):
                row['hit'+str(budget)] = int(((group.found == 1) & (group['rank'] <= budget)).sum())
            expected = saved.loc[identity]
            for key, value in row.items():
                if key not in group_fields:
                    assert math.isclose(float(expected[key]), value, rel_tol=1e-12, abs_tol=1e-14), (name, identity, key)
            reconstructed.append(row)
        pd.DataFrame(reconstructed).to_csv(output / ("independent_"+name), index=False)
    status = dict(cases=len(jobs), positions=2047, rows=len(frame), report_rows=report_rows,
        relabeled=changed, report_only_correction=True, removed_products=0,
        original_rankings_reproduced=True, all_metrics_independently_reconstructed=True,
        files={p.name:digest(p) for p in output.glob('*.csv')}, verification_source_sha256=digest(Path(__file__)))
    atomic_json(output / "independently_verified.json", status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
