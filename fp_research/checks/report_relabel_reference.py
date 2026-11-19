"""用官方组合枚举和未加速Java切片核验银行账户修正报告。"""
import json
from pathlib import Path
import subprocess

from fp_research import varcop_report_relabel as current
from fp_research.localization_study import read, digest


def check():
    jobs = read(current.PREVIOUS_OUTPUT / "manifest.json")["jobs"]
    selected = [j for j in jobs if j["system"] == "bank" and j["stratum"] == 1]
    _, _, spc = current.legacy.upstream()
    checked = 0
    # 比较集合是因为官方与等价计算可能按不同顺序输出同一行。
    for job in selected:
        version, output = Path(job["path"]), current.folder(job)
        for method in ["original", "joint", "CLAP", "PULP", "PULP-default"]:
            report = output / method / "config.report.csv"
            target = output / method / "reference-spc.log"
            with current.use_report(version, report):
                _, rows = current.load_report(report)
                features, names, passed, failed = spc.load_configs(str(report), {r[0]: 1. for r in rows}, 0.)
                spc.detect_SPCs(features, passed, failed, names, str(version / "variants"), str(target))
            normalize = lambda p: {line.replace('\\', '/') for line in p.read_text(encoding="utf-8").splitlines()}
            assert normalize(target) == normalize(output / method / "spc.log"), (job["version"], method)
            checked += 1
    job = next(j for j in selected if j["version"] == "_MultipleBugs_.NOB_1.ID_14")
    output, version = current.folder(job), Path(job["path"])
    reference = output / "joint/reference-slicing.json"
    command = [str(current.legacy.JAVA), "-Xmx1536m", "-XX:+UseSerialGC", "-XX:ActiveProcessorCount=2",
               "-XX:CICompilerCount=2", "-Djava.io.tmpdir="+str(output),
               "-Dspc_path="+str(output / "joint/spc-observed-failures.log"),
               "-Dslicing_output_path="+str(reference), "-Dcoverage_file_name=spectrum_failed_coverage.xml",
               "-jar", str(current.legacy.JAR)]
    with (output / "joint/reference-slice-run.log").open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)
    names = {r[0] for r in current.load_report(current.report_source(version))[1]}
    expected = current.legacy.read_isolation(version, reference, set(), names)
    actual = current.legacy.read_isolation(version, output / "joint/slicing.json", set(), names)
    assert {cid:set(stms) for cid, stms in expected.items()} == {cid:set(stms) for cid, stms in actual.items()}
    proof = dict(cases=len(selected), corrected_reports_checked=checked,
        official_combinations_identical=True, official_uncached_slice_identical=True,
        reference_version=job["version"], reference_method="joint",
        java_slicer_sha256=digest(current.legacy.JAR), verification_source_sha256=digest(Path(__file__)))
    current.atomic_json(current.OUTPUT / "pilot-reference.json", proof)
    print(json.dumps(proof, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    check()
