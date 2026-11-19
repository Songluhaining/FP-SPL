"""六系统的只读案例选择与报告核验。"""
import csv
from pathlib import Path
import re
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
