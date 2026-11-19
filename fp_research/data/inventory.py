"""清点原始配置、源码及逐测试成败，不读取评价真值。"""
import csv
from pathlib import Path
def records(version):
    report = version / "config.report.csv.done"
    if not report.is_file():
        report = version / "config.report.csv"
    with report.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header[-1].strip() != "__TEST_OUTPUT__":
            raise ValueError(f"未知配置表头 {version}")
        seen = set()
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"配置列数不一致 {version}: {row}")
            name, *flags, outcome = [cell.strip() for cell in row]
            if name in seen or outcome not in {"__PASSED__", "__FAILED__"} or set(flags) - {"T", "F"}:
                raise ValueError(f"配置记录异常 {version}/{name}")
            seen.add(name)
            folder = version / "variants" / name
            tests = {}
            for status in ("passed", "failed"):
                tests[status] = sorted(p.name.removesuffix(".coverage.xml") for p in
                                       (folder / "coverage" / status).glob("*.coverage.xml"))
            if set(tests["passed"]) & set(tests["failed"]):
                raise ValueError(f"用例状态冲突 {version}/{name}")
            observed = "failed" if tests["failed"] else "passed" if tests["passed"] else "unknown"
            yield dict(configuration=name, outcome=outcome.strip("_").lower(),
                       features=[key for key, flag in zip(header[1:-1], flags) if flag == "T"],
                       tests=tests, coverage_agrees=observed == outcome.strip("_").lower(),
                       source=any((folder / "src").rglob("*.java")),
                       test_source=any((folder / "test").rglob("*.java")))
        actual = {p.name for p in (version / "variants").iterdir() if p.is_dir()}
        if seen != actual:
            raise ValueError(f"配置目录与报告不一致 {version}: {seen ^ actual}")
