"""等待准备与独立联合推断结束，继续分类、定位、汇总和独立核验。"""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

from .inventory import atomic_json
from .localization_study import ROOT, OUTPUT, manifest, work_folder, read, digest


def run(arguments, filename):
    path = ROOT / "result" / filename
    with path.open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", *arguments], cwd=ROOT, env=os.environ.copy(),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    jobs = manifest()
    previous = None
    while True:
        states = {}
        for stage in ("fit", "prepare"):
            states[stage] = [read(p) for j in jobs
                             if (p := work_folder(j) / f"{stage}-status.json").exists()]
        current = {s: dict(complete=sum(r["ok"] for r in rows), failed=sum(not r["ok"] for r in rows))
                   for s, rows in states.items()}
        if current != previous:
            print(datetime.now().astimezone().isoformat(), current, flush=True)
            atomic_json(OUTPUT / "progress.json", current)
            previous = current
        if all(v["complete"] == len(jobs) for v in current.values()):
            break
        time.sleep(20)
    classifiers = []
    for method, default in (("CLAP", False), ("PULP", False), ("PULP", True)):
        name = "PULP-default" if default else method
        stream = (ROOT / "result" / f"localization-classify-{name}.log").open("w", encoding="utf-8")
        command = [sys.executable, "-m", "fp_research.localization_study", "classify", "--method", method]
        if default:
            command.append("--original-default")
        classifiers.append((name, subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                                    stdout=stream, stderr=subprocess.STDOUT), stream))
    failed = []
    for name, process, stream in classifiers:
        code = process.wait()
        stream.close()
        if code:
            failed.append(name)
    if failed:
        raise RuntimeError(f"比较方法失败 {failed}；查看日志并修复后恢复")
    run(["fp_research.localization_study", "rank", "--workers", "4"], "localization-full-rank.log")
    run(["fp_research.localization_study", "evaluate"], "localization-evaluate.log")
    run(["fp_research.checks.verify_localization"], "localization-verify.log")
    atomic_json(OUTPUT / "completion.json", dict(cases=len(jobs), verified=digest(OUTPUT / "verified.json"),
        files={p.name: digest(p) for p in (OUTPUT / "fault_positions.csv", OUTPUT / "metrics.csv",
                                         OUTPUT / "语句级定位比较.csv", OUTPUT / "实验报告.md")},
        completed=datetime.now().astimezone().isoformat()))
    print("完成全部定位与核验", len(jobs), flush=True)


if __name__ == "__main__":
    main()
