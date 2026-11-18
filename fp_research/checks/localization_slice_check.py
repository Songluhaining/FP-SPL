"""用当前缓存重算六个代表案例，与已经运行的官方整案例输出比较。"""
from pathlib import Path

from fp_research.localization_study import manifest, work_folder, OUTPUT, read, digest
from fp_research import varcop_localization as v
from fp_research.varcop_localization import input_fingerprint, source_hashes
from fp_research.checks.localization_smoke import assert_same_ranking
from fp_research.varcop_slice import batch
from fp_research.inventory import atomic_json


def main():
    jobs = manifest()
    for system in ["bank", "elevator", "email", "examdb", "gpl", "zipme"]:
        job = next(j for j in jobs if j["system"] == system and j["stratum"] == 1)
        folder = work_folder(job) / "smoke/varcop"
        assert input_fingerprint(Path(job["path"]))[0] == read(folder / "complete.json")["protocol"]["input"]
        log, reference = folder / "original/spc.log", folder / "original/slicing.json"
        target = folder / "equivalence"
        target.mkdir(exist_ok=True)
        actual = target / "original.json"
        batch(Path(job["path"]), {"original": (log, actual)}, target)
        a, b = read(actual), read(reference)
        assert a.keys() == b.keys()
        assert all(a[c].keys() == b[c].keys() for c in a), system
        counter_differences = [dict(configuration=c, statement=s, reconstructed=a[c][s], reference=b[c][s])
                              for c in a for s in a[c] if a[c][s] != b[c][s]]
        # 官方定位仅使用切片的语句集合，不读取交互次数；直接核对当前输出的定位分数与最坏并列名次。
        ranking, _, _ = v.upstream()
        version = Path(job["path"])
        all_stms, failing = ranking.get_executed_stms_of_the_system(version.as_posix(), "", 0.)
        names = set(all_stms)
        current = v.read_isolation(version, actual, set(), names)
        official = v.read_isolation(version, reference, set(), names)
        from fp_research.localization_study import METRICS
        local = ranking.local_ranking_a_suspicious_list(version.as_posix(), failing, METRICS, "")
        product_scores = ranking.product_based_assessment(version.as_posix(), all_stms, METRICS, "")
        for metric in METRICS:
            args = (local[metric], product_scores[metric], metric,
                    ranking.AGGREGATION_ARITHMETIC_MEAN, ranking.NORMALIZATION_ENABLE, .5)
            assert_same_ranking(ranking.global_ranking_a_suspicious_list(all_stms, failing, official, *args),
                                ranking.global_ranking_a_suspicious_list(all_stms, failing, current, *args))
        atomic_json(OUTPUT / f"slice-equivalence-{system}.json", dict(system=system, version=job["version"],
            products=len(b), statement_sets_identical=True, rankings_identical=True,
            full_counts_identical=not counter_differences, unused_counter_differences=counter_differences,
            basis_reconstruction_checked=True, whole_case_official_jar_checked=True,
            official_slicing_sha256=digest(reference), spc_sha256=digest(log),
            reconstructed_sha256=digest(actual)))
        print("整案例切片与官方结果一致", system, len(read(reference)), flush=True)
    features = read(OUTPUT / "feature-reference.json")
    assert len(features) == 12
    names = [f"slice-equivalence-{s}.json" for s in
             ["bank", "elevator", "email", "examdb", "gpl", "zipme"]]
    atomic_json(OUTPUT / "adapters-verified.json", dict(source=source_hashes(),
        whole_case_equivalence={name: digest(OUTPUT / name) for name in names},
        representative_features=digest(OUTPUT / "feature-reference.json"), feature_rebuilds=12))


if __name__ == "__main__":
    main()
