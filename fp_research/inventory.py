"""从原始配置报告和逐测试覆盖目录建立观察清单，不读取研究真值。"""
import argparse
import csv
import json
import os
from pathlib import Path

SYSTEMS = {
    "bank": ("BankAccountTP", "4wise-BankAccountTP-1BUG-Full"),
    "email": ("Email-FH-JML", "4wise-Email-FH-JML-1BUG-Full"),
}


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


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temp, path)


def inventory(data, output, systems, selected=None):
    counts = {}
    for system in systems:
        root = data.joinpath(*SYSTEMS[system])
        versions = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("_MultipleBugs_"))
        if selected:
            versions = [p for p in versions if p.name == selected]
        if not versions:
            raise ValueError(f"找不到缺陷版本 {root}/{selected or '*'}")
        counts[system] = dict(versions=len(versions), configurations=0, passed=0, failed=0,
                              conflicting_coverage=0, missing_source=0, passed_tests=0, failed_tests=0)
        c = counts[system]
        for version in versions:
            rows = list(records(version))
            for row in rows:
                c["configurations"] += 1
                c[row["outcome"]] += 1
                c["conflicting_coverage"] += int(not row["coverage_agrees"])
                c["missing_source"] += int(not (row["source"] and row["test_source"]))
                for status, ids in row["tests"].items():
                    c[status + "_tests"] += len(ids)
            atomic_json(output / system / version.name / "inventory.json", rows)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--version")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.data.resolve()):
        parser.error("输出目录须与原始数据隔离")
    counts = inventory(args.data, args.output, [args.system] if args.system else SYSTEMS, args.version)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
