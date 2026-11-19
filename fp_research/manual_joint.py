"""人工运行当前联合贝叶斯识别：逐案例独立推断，默认新建结果目录。"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parents[1]
LOCATIONS = ROOT / "fp_research/dataset_locations.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_report(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or rows[0][-1].strip() != "__TEST_OUTPUT__":
        raise ValueError("缺少有效的原始配置报告")
    if any(len(r) != len(rows[0]) or r[-1].strip() not in {"__PASSED__", "__FAILED__"} for r in rows[1:]):
        raise ValueError("原始配置报告的列数或状态无效")
    if len({r[0].strip() for r in rows[1:]}) != len(rows)-1:
        raise ValueError("原始配置报告包含重复配置")
    return rows


def make_job(case, system, stratum):
    case = Path(case).resolve()
    if not case.is_dir():
        raise ValueError(f"案例目录不存在：{case}")
    # 当前核心以这份原始报告核对逐测试覆盖，不读取后来生成的修正报告。
    rows = read_report(case / "config.report.csv.done")
    return dict(system=system, stratum=stratum, version=case.name, path=str(case),
                configurations=len(rows)-1,
                candidates=sum(r[-1].strip() == "__PASSED__" for r in rows[1:]))


def select_jobs(locations, system=None, stratum="1", version=None, case=None):
    systems = locations["systems"]
    deferred = set(locations.get("deferred_multifault", []))
    if case is not None:
        if version is not None or system is not None:
            raise ValueError("直接指定案例路径时，不再同时指定系统或版本名")
        case = Path(case).resolve()
        match = re.fullmatch(r"_MultipleBugs_\.NOB_([123])\.ID_.+", case.name)
        if not match:
            raise ValueError("案例名应为_MultipleBugs_.NOB_缺陷数.ID_编号")
        count = int(match[1])
        for name, raw in systems.items():
            base = Path(raw).resolve()
            if case.parent == base / f"4wise-{base.name}-{count}BUG-Full":
                if name in deferred and count > 1:
                    raise ValueError("当前协议暂缓该系统双、三缺陷案例")
                return [make_job(case, name, count)]
        raise ValueError("案例路径不属于路径配置中六个系统的标准缺陷层")
    if system is None:
        raise ValueError("请指定--system或--case")
    chosen = list(systems) if system == "all" else [system]
    if any(name not in systems for name in chosen):
        raise ValueError("系统代码不在数据路径配置中")
    counts = [1, 2, 3] if str(stratum) == "all" else [int(stratum)]
    if not set(counts) <= {1, 2, 3}:
        raise ValueError("缺陷层只能为1、2、3或all")
    jobs = []
    for name in chosen:
        base = Path(systems[name]).resolve()
        for count in counts:
            if name in deferred and count > 1:
                if system != "all" and str(stratum) != "all":
                    raise ValueError("当前协议暂缓该系统双、三缺陷案例")
                continue
            layer = base / f"4wise-{base.name}-{count}BUG-Full"
            if not layer.is_dir():
                raise ValueError(f"系统缺陷层不存在：{layer}")
            for candidate in sorted(layer.glob(f"_MultipleBugs_.NOB_{count}.ID_*")):
                if candidate.is_dir() and (version is None or candidate.name == version):
                    jobs.append(make_job(candidate, name, count))
    if not jobs:
        raise ValueError("没有找到匹配案例，请先用--list检查名称")
    return jobs


def scenario(run_name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("运行名只允许字母、数字、下划线和连字符，长度1至64")
    return "manual_joint_detection/"+run_name


def corrected_report(original, predictions, target):
    """生成本次人工运行的修正报告，保留其余字节，不覆盖案例根目录文件。"""
    original, predictions, target = Path(original), Path(predictions), Path(target)
    if target.resolve() == original.resolve():
        raise ValueError("不能覆盖原始报告")
    raw = original.read_bytes()
    rows = read_report(original)
    with predictions.open(encoding="utf-8-sig", newline="") as stream:
        predicted = list(csv.DictReader(stream))
    passing = {r[0].strip() for r in rows[1:] if r[-1].strip() == "__PASSED__"}
    if len(predicted) != len(passing) or {r["configuration"] for r in predicted} != passing:
        raise ValueError("识别结果缺少原本通过配置，或出现重复、额外配置")
    if any(r["prediction"] not in {"0", "1"} for r in predicted):
        raise ValueError("预测值应为0或1")
    selected = {r["configuration"] for r in predicted if r["prediction"] == "1"}
    lines = raw.splitlines(keepends=True)
    if len(lines) != len(rows):
        raise ValueError("配置报告包含多行字段，不能逐行修正")
    for i, row in enumerate(rows[1:], 1):
        if row[0].strip() in selected:
            prefix, final = lines[i].rsplit(b",", 1)
            if not final.startswith(b"__PASSED__"):
                raise ValueError("配置报告最后一列格式不符")
            lines[i] = prefix+b","+final.replace(b"__PASSED__", b"__FAILED__", 1)
    result = b"".join(lines)
    if target.exists():
        if target.read_bytes() != result:
            raise ValueError("同名人工修正报告已被修改，请选择新的运行名")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(result)
    assert original.read_bytes() == raw and target.read_bytes() == result
    return len(selected)


def run_job(job, run_name):
    from . import multifault_study as core
    from .inventory import atomic_json
    if job["candidates"] == 0:
        raise ValueError("此案例没有原本通过配置，无需进行假通过识别")
    previous = core.SCENARIO
    core.SCENARIO = scenario(run_name)
    try:
        result = core.fit(job)
        output = core.folder(job)
    finally:
        core.SCENARIO = previous
    count = corrected_report(Path(job["path"]) / "config.report.csv.done", output / "joint.csv",
                             output / "config.report.joint.csv")
    proof = read_json(output / "joint.json")
    atomic_json(output / "manual-run.json", dict(job=job, run_name=run_name, result=result,
        fake_passing_predictions=count, stable=proof["stable"], parameters=core.PARAMETERS, seeds=core.SEEDS,
        input_report="config.report.csv.done", threshold=.5, output=str(output),
        entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        corrected_report_sha256=hashlib.sha256((output / "config.report.joint.csv").read_bytes()).hexdigest(),
        created=datetime.now().astimezone().isoformat()))
    print(f"完成：{job['system']} / {job['version']}；候选{job['candidates']}，识别假通过{count}，稳定性{proof['stable']}；{result}", flush=True)
    print(f"本次结果：{output}", flush=True)
    return proof["stable"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--system", choices=["bank", "email", "elevator", "examdb", "gpl", "zipme", "all"])
    group.add_argument("--case", type=Path, help="直接提供一个案例的完整路径")
    parser.add_argument("--stratum", choices=["1", "2", "3", "all"], default="1", help="缺陷层，仅用于选择目录，不告知模型")
    parser.add_argument("--version", help="完整案例目录名")
    parser.add_argument("--list", action="store_true", help="只列出案例，不推断、不生成案例文件")
    parser.add_argument("--run-name", help="指定人工运行目录名；相同名称可复用本次入口自己的有效缓存")
    args = parser.parse_args(argv)
    try:
        name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6]
        scenario(name)
        jobs = select_jobs(read_json(LOCATIONS), args.system, args.stratum, args.version, args.case)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.list:
        for job in jobs:
            print(f"{job['system']}\t{job['stratum']}\t{job['version']}\t配置{job['configurations']}\t候选{job['candidates']}\t{job['path']}")
        print(f"共{len(jobs)}个案例；只列清单，未运行推断。")
        return 0
    failed, unstable = 0, 0
    for job in jobs:
        try:
            unstable += not run_job(job, name)
        except Exception as error:
            failed += 1
            print(f"失败：{job['path']}；{type(error).__name__}：{error}", flush=True)
    print(f"案例{len(jobs)}，失败{failed}，未达到稳定性条件{unstable}；运行名{name}", flush=True)
    return 1 if failed else 2 if unstable else 0


if __name__ == "__main__":
    raise SystemExit(main())
