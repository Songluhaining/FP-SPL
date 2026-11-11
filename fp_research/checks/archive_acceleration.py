"""汇集已完成的等价性核验，重新核对保存的切片和特征文件。"""
import hashlib
import json
from pathlib import Path

from fp_research.case_study import ROOT, SYSTEMS
from fp_research.inventory import atomic_json
from fp_research.slice_runner import compiled_runner


def main():
    data = Path("D:/dataset/whn")
    runner, key = compiled_runner()
    source_hash = hashlib.sha256((ROOT / "fp_research/checks/SliceProfile.java").read_bytes()).hexdigest()
    binary_hash = hashlib.sha256((runner / "SliceProfile.class").read_bytes()).hexdigest()
    proofs = []
    for system, parts in SYSTEMS.items():
        base = data.joinpath(*parts)
        for path in sorted(base.glob("*/.fp_research/profile-indexed/*/verified.json")):
            record = json.loads(path.read_text(encoding="utf8"))
            assert record["runner_sha256"] == source_hash
            assert record["binary_sha256"] == binary_hash
            for direction in ("forward", "backward"):
                name = f"slice.all_statements.{direction}_slicing.log"
                reference = base / ".fp_research/slice-cache" / record["cache"] / name
                output = path.parent / "variants" / record["configuration"] / name
                read = lambda p: {k: set(v) for k, v in json.loads(p.read_text()).items()}
                assert read(reference) == read(output), str(path)
            proofs.append(dict(record, receipt=str(path)))
    assert len(proofs) == 6
    base = data.joinpath(*SYSTEMS["gpl"], "_MultipleBugs_.NOB_1.ID_1/.fp_research")
    features = []
    for method, folder in (("CLAP", "fast-baseline-check"), ("PULP", "fast-baseline-final-check")):
        reference = base / "main" / method / "features.csv"
        output = base / folder / method / "features.csv"
        assert reference.read_bytes() == output.read_bytes()
        features.append(dict(method=method, reference=str(reference), output=str(output),
                             sha256=hashlib.sha256(output.read_bytes()).hexdigest(), byte_equal=True))
    atomic_json(ROOT / "experimental_results/gpl_zipme/acceleration-verification.json",
                dict(runner_key=key, slice_comparisons=proofs, feature_comparisons=features,
                     scope="等价计算核验；不代表两个系统分类实验已完成"))
    print("6 real slice comparisons and 2 complete feature tables verified")


if __name__ == "__main__":
    main()
