"""一个真实案例核对只读适配及原版 VARCOP 全排名一致。"""
import csv
import gzip
import json
from pathlib import Path

import pandas as pd

from fp_research import varcop_localization as v
from fp_research.localization_study import manifest, inference_folder, work_folder, OUTPUT


def check(system, stratum):
    job = next(j for j in manifest() if j["system"] == system and j["stratum"] == stratum)
    version = Path(job["path"])
    with (version / "config.report.csv.done").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header, configs = next(reader), list(reader)
    joint_path = inference_folder(job) / "joint.csv"
    predictions = pd.read_csv(joint_path)
    choices = {"original": set(), "joint": set(predictions.loc[predictions.prediction == 1, "configuration"])}
    # 专门检查排除分支，使用一个原本通过配置；这一集合仅用于入口核验。
    probe = next(row[0] for row in configs if row[-1].strip() == "__PASSED__")
    choices["exclusion-check"] = {probe}
    v.correction_sets = lambda _: (header, configs, choices,
                                  {"joint": dict(path=str(joint_path), sha256=v.digest(joint_path))}, {})
    original_folder = work_folder(job)
    v.work_folder = lambda _: original_folder / "smoke"
    before = v.input_fingerprint(version)[0]
    print(v.rank_case(job), flush=True)
    folder = v.work_folder(job) / "varcop"
    receipt = v.read(folder / "complete.json")
    with gzip.open(folder / "original/rankings.json.gz", "rt", encoding="utf-8") as stream:
        actual = json.load(stream)
    ranking, products, spc = v.upstream()
    all_statements, failing = ranking.get_executed_stms_of_the_system(version.as_posix(), "", 0.)
    isolated = v.isolate(version, folder / "original", set(), configs, header, spc)
    spaces = {ranking.SS_ALL_STMS: all_statements, ranking.SS_STMS_IN_F_PRODUCTS: failing,
              ranking.SS_VARCOP: isolated, ranking.SS_SLICING: {}}
    original_global = ranking.global_ranking_a_suspicious_list
    captured = []

    def capture(*args, **kwargs):
        result = original_global(*args, **kwargs)
        captured.append(result)
        return result

    ranking.global_ranking_a_suspicious_list = capture
    try:
        ranking.ranking_multiple_bugs([], version.as_posix(), spaces, list(actual),
                                     ranking.AGGREGATION_ARITHMETIC_MEAN,
                                     ranking.NORMALIZATION_ENABLE, "", 0., .5)
    finally:
        ranking.global_ranking_a_suspicious_list = original_global
    assert len(captured) == 3 * len(actual)
    for i, (metric, lists) in enumerate(actual.items()):
        assert [list(x) for x in captured[3*i]] == lists["full"], metric
        assert [list(x) for x in captured[3*i+1]] == lists["isolated"], metric
    assert v.input_fingerprint(version)[0] == before
    with v.excluded_products(ranking, products, choices["exclusion-check"]):
        retained = ranking.list_dir((version / "variants").as_posix())
        assert set(retained) == {r[0] for r in configs} - choices["exclusion-check"]
    assert len(ranking.list_dir((version / "variants").as_posix())) == len(configs)
    proof = dict(system=system, stratum=stratum, version=job["version"],
                 all_six_full_rankings_identical=True, all_six_isolated_rankings_identical=True,
                 correction_exclusion_checked=True, raw_inputs_unchanged=True)
    v.atomic_json(OUTPUT / f"smoke-{system}-{stratum}.json", proof)
    print(proof, flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--stratum", type=int, default=1)
    a = p.parse_args()
    check(a.system, a.stratum)
