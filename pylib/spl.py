import csv
import os
import shlex
import shutil
import subprocess
import textwrap
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from SUMethods.mutual_information import su_calculation
from deleteLogfiles import delete_files_with_name
from pylib.countmap import CountMap
import time
from func_timeout import func_set_timeout
import func_timeout
from typing import Any, Dict, List, Set
import json
from multiprocessing import Pool, TimeoutError
import re
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter

from os.path import join, getsize

from pyutils.FileManager import get_spc_log_file_path, is_path_exist, logger, get_model_configs_report_path, \
    get_variants_dir, get_file_name_with_parent, get_src_dir, join_path
from pyutils.SuspiciousStatementManager import get_multiple_buggy_statements
from pyutils.TestingCoverageManager import statement_coverage_of_variants
from pyutils.clustering import cluster_failed_traces
from pyutils.get_trace_vector import get_spectrum_stm, gererateObDataFromSB
from pyutils.mappingUtils import rewrite_logs_for_variant
from pyutils.ranking.RankingManager import get_set_of_stms, get_executed_stms_of_the_system
from pyutils.ranking.Spectrum_Expression import *
from pyutils.sbflPrio import get_sbfl_prio
from pyutils.utils import group_items_by_class, load_test_file_source, llm_match_passed_for_failed_in_class, call_llm
from pyutils.llm_config import get_llm_config
from pylib.llm_selection_cache import (
    cache_enabled,
    derive_cache_identity,
    load_variant_cache,
    save_variant_cache,
)
from recompile_failed_products import get_failed_product_names


use_simple_filter = False

def utf8open(filename):
    return open(filename, encoding='utf-8', errors='ignore')
def utf8open_w(filename):
    return open(filename, 'w+', encoding='utf-8', errors='ignore')

def utf8open_a(filename):
    return open(filename, 'a+', encoding='utf-8', errors='ignore')


checkoutbase = utf8open('checkout.config').readline().strip()
projectbase = os.path.abspath(".")

def getdirsize(dir:str):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([getsize(join(root, name)) for name in files])
    return size

def getinstclassinfo(proj: str):
    cmdline = f"defects4j query -p {proj} -q \"bug.id,classes.relevant.src,classes.relevant.test,tests.trigger,tests.relevant\"  -o ./d4j_resources/{proj}.csv"
    cmdline += f' > trace/runtimelog/{proj}query.log'
    os.system(cmdline)

def parseprofile(line: str, trigger_tests: Set[str], testmethods: Set[str]):
    line = line[3:]
    sp = line.split('::')
    class_name = sp[0].strip()
    method_name = sp[1].strip()
    is_trigger = (class_name, method_name) in trigger_tests
    is_test = (class_name, method_name) in testmethods
    return class_name, method_name, is_trigger, is_test


def resolve_profile(profile: List[str], classes_relevant: List[str], trigger_tests: List[str], testmethods: List[str], debug=True) -> List[str]:
    if debug:
        print(f'parsing profile, length:{len(profile)}')
        print('trigger tests are: ', trigger_tests)
    relevant = []
    relevant_cnt = CountMap()

    # currelevant = False
    trigger_tests_set, trigger_tests_map = parse_trigger_tests(trigger_tests)
    testmethods_set = parse_test_methods(testmethods)
    fail_coverage = get_fail_coverage(
        profile, trigger_tests_set, testmethods_set)
    curclass = ''
    curmethod = ''
    for line in profile:
        if line.strip() == '':
            continue
        class_name, method_name, is_trigger, is_test = parseprofile(
            line, trigger_tests_set, testmethods_set)
        if is_test:
            curclass, curmethod = class_name, method_name
            # currelevant = False
            continue
        # if currelevant:
        #     continue
        # relevant
        if curclass == '' or curmethod == '':
            continue
        if (class_name, method_name) in fail_coverage:
            # FIXME weird use-before-def bug for curclass,curmethod at Math-2.
            curtest = (curclass, curmethod)
            relevant.append(curtest)
            relevant_cnt.add(curclass, curmethod)
            # currelevant = True
    # TODO use relevant_cnt for filtering
    # relevant = list(set(relevant))
    # return sorted(relevant)

    if use_simple_filter:
        return relevant_cnt.method_filter_simple(trigger_tests_map, testmethods_set)
    else:
        return relevant_cnt.method_filter(trigger_tests_map, testmethods_set)


def get_fail_coverage(profile, trigger_tests_set, testmethods_set):
    fail_coverage = set()
    curtrigger = False
    for line in profile:
        if line.strip() == '':
            continue
        class_name, method_name, is_trigger, is_test = parseprofile(
            line, trigger_tests_set, testmethods_set)
        if is_test:
            # curclass, curmethod = class_name, method_name
            curtrigger = is_trigger
            continue
        if curtrigger:
            fail_coverage.add((class_name, method_name))
            # print(fail_coverage)
    return fail_coverage


def parse_test_methods(testmethods):
    testmethods_set = set()
    for testmethod in testmethods:
        testmethod = testmethod.strip()
        if testmethod == '':
            continue
        sp = testmethod.split('::')
        methods = sp[1].split(',')
        # print(sp[0], sp[1])
        for method in methods:
            method = method.strip()
            if method != '':
                testmethods_set.add((sp[0], method))
    return testmethods_set


def parse_trigger_tests(trigger_tests):
    trigger_tests_set = set()
    trigger_tests_map = {}
    for trigger_test in trigger_tests:
        trigger_test = trigger_test.strip()
        if trigger_test == '':
            continue
        sp = trigger_test.split('::')
        classname = sp[0].strip()
        methodname = sp[1].strip()
        trigger_tests_set.add((classname, methodname))
        if classname in trigger_tests_map:
            if not methodname in trigger_tests_map[classname]:
                trigger_tests_map[classname].append(methodname)
        else:
            trigger_tests_map[classname] = [methodname]
    return trigger_tests_set, trigger_tests_map


def collect_items(_dir: Path) -> List[Dict]:
    """
    解析类似 'ElevatorSystem.Elevator_ESTest.test33.coverage.xml'
    返回 [{'sut_class': 'ElevatorSystem.Elevator', 'test_method': 'test33',
          'estest_class': 'ElevatorSystem.Elevator_ESTest'} ...]
    """
    items = []
    if not _dir.exists():
        return items
    pat = re.compile(
        r'^(?P<fqbase>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)_ESTest\.(?P<mtd>[A-Za-z_]\w*)\.coverage\.xml$'
    )
    for p in _dir.iterdir():
        if not p.is_file() or not p.name.endswith(".coverage.xml"):
            continue
        m = pat.match(p.name)
        if not m:
            continue
        fqbase = m.group("fqbase")               # 'ElevatorSystem.Elevator'
        mtd    = m.group("mtd")                  # 'test33'
        items.append({
            "sut_class": fqbase,
            "test_method": mtd,
            "estest_class": fqbase + "_ESTest",  # 仅当确有该测试类时才使用
            "raw": p.name
        })
    return items

def _collect_sut_from_coverage_names(reltest_dict: Dict[str, List[str]]) -> List[str]:
    """
    由测试类名反推出 SUT 类名（去掉 _ESTest 后缀）。
    """
    sut = set()
    for tcls in reltest_dict.keys():
        if tcls.endswith("_ESTest"):
            sut.add(tcls[:-7])   # 去掉尾部 "_ESTest" 7 个字符
    return sorted(sut)

def _collect_sut_classes_from_src(src_dir: Path) -> List[str]:
    """
    从 src/ 解析 package + 文件名得到 FQCN，仅收集 SUT（过滤掉 *Test）。
    生成的列表用于 instrumentingclass 的“业务类集合”。
    """
    sut = set()
    for f in src_dir.rglob("*.java"):
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue
        m = re.search(r'^\s*package\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*;', txt, re.M)
        if not m:
            continue  # 无 package 的类跳过
        pkg = m.group(1)
        cls = f.stem
        if cls.endswith("Test") or cls.endswith("_ESTest"):
            continue
        sut.add(f"{pkg}.{cls}")
    # 若类非常多，可在这里改为按包前缀聚合（前提：你的 agent 支持前缀/通配）
    return sorted(sut)

# JUNIT_STANDALONE_JAR = "./lib/junit-platform-console-standalone-1.10.2.jar"
JUNIT_STANDALONE_JAR = os.path.join(projectbase, "lib", "junit-platform-console-standalone-1.10.2.jar")
CLOVER_RUNTIME_JAR   = os.path.join(projectbase, "lib", "clover-runtime-4.4.1.jar")
EVOSUITE_RUNTIME_JAR   = os.path.join(projectbase, "lib", "evosuite-standalone-runtime-1.0.6.jar")
SLF4J_SIMPLE_JAR = os.path.join(projectbase, "lib", "slf4j-simple-1.7.30.jar")
# 如有额外依赖的 jar（第三方库），放到此目录下；没有就置为 None
EXTRA_LIB_DIR = None  # e.g., "/path/to/lib"

def _dir_has_class(d: Path) -> bool:
    return d.exists() and any(d.rglob("*.class"))

def _build_classpath_for_your_layout(variant_root: Path, debug_dir: Path) -> str:
    """
    你的目录形态：build/main 下直接是顶层包名目录（ElevatorSystem/, main/, TestSpecifications/）
    因此 classpath 的主根应为 build/main。
    若该目录不存在或没有 .class，则报错并在 debug 日志里提示。
    """
    vr = Path(variant_root).resolve()
    dbg = Path(debug_dir); dbg.mkdir(parents=True, exist_ok=True)
    cp_log = dbg / "cp.check.txt"

    main_root = vr / "build" / "main"          # <== 关键：你的 classpath 根
    test_root = None                            # 测试类也在 main_root 下，无需单独 test 根

    jars = []
    if not Path(JUNIT_STANDALONE_JAR).exists():
        raise RuntimeError(f"[classpath] JUnit console jar not found: {JUNIT_STANDALONE_JAR}")
    jars.append(Path(JUNIT_STANDALONE_JAR).resolve())
    if EXTRA_LIB_DIR and os.path.isdir(EXTRA_LIB_DIR):
        for j in sorted(os.listdir(EXTRA_LIB_DIR)):
            if j.endswith(".jar"):
                jars.append((Path(EXTRA_LIB_DIR) / j).resolve())

    # 强校验：build/main 必须存在且含有 .class
    if not _dir_has_class(main_root):
        with cp_log.open("w", encoding="utf-8") as w:
            w.write(f"[variant_root] {vr}\n")
            w.write(f"[expect main root] {main_root}\n")
            w.write("[hint] We expect .class files under build/main/<TopLevelPackage>/...\n")
            w.write("       e.g., build/main/ElevatorSystem/Elevator.class\n")
        raise RuntimeError(
            f"[classpath] No .class found under {main_root}. "
            f"See debug: {cp_log}"
        )

    parts = [str(main_root.resolve()), *map(str, jars)]
    # 打印与落盘
    print("[classpath parts]")
    for p in parts: print("  ", p)
    with cp_log.open("w", encoding="utf-8") as w:
        w.write(f"[variant_root] {vr}\n")
        w.write(f"[main_root] {main_root.resolve()}\n")
        w.write(f"[jar] {jars[0]}\n")

        # 抽样列几个 .class，方便肉眼核对
        w.write("\n[sample .class under main_root]\n")
        k = 0
        for c in main_root.rglob("*.class"):
            w.write("  " + str(c.relative_to(main_root)) + "\n")
            k += 1
            if k >= 12: break

    return os.pathsep.join(parts)

def _fqcn_to_relpath(fqcn: str) -> str:
    return fqcn.replace('.', '/') + '.class'

def _preflight_verify_instclasses(classesdir: Path, inst_list: list, dbg_dir: Path) -> None:
    dbg_dir.mkdir(parents=True, exist_ok=True)
    out = []
    miss = 0
    ok = 0
    for fqcn in inst_list:
        rel = _fqcn_to_relpath(fqcn)
        p = classesdir / rel
        if p.exists():
            ok += 1
            out.append(f"OK   {fqcn} -> {rel}")
        else:
            miss += 1
            out.append(f"MISS {fqcn} -> {rel}")
    (dbg_dir / "instcheck.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"[preflight] instclasses check: OK={ok}, MISS={miss} -> {dbg_dir/'instcheck.txt'}")

AGENT_JAR = os.path.abspath("./target/ppfl-0.0.1-SNAPSHOT-jar-with-dependencies.jar")
JAVA_BIN  = os.environ.get("JAVA_BIN", shutil.which("java") or "/usr/bin/java")

def _class_exists(classes_roots: List[Path], est: str) -> bool:
    rel = est.replace('.', os.sep) + '.class'
    for root in classes_roots:
        if (root / rel).exists():
            return True
    return False

def build_test_desc(item: Dict[str, Any]) -> str:
    """
    为 LLM 构造一个可读描述。后面你可以在这里加更多信息：
    - 对应的被测类 SUT
    - 若能定位到测试源码，可以加方法注释 / JML 规格 / 名字中的数字模式等
    """
    est = item.get("estest_class", "<UNKNOWN_EST>")
    tm  = item.get("test_method", "<UNKNOWN_TEST>")
    sut = item.get("sut_class", "<UNKNOWN_SUT>")
    status = item.get("status", "unknown")

    return f"[status={status}] estest_class={est}, test_method={tm}, sut_class={sut}"



def call_chat_model(prompt: str, model: str = None) -> str:
    """
    单轮对话调用：把一段 prompt 交给配置好的大模型，返回纯文本。

    具体用哪个厂商 / 哪个模型 / 哪个 Key，全部来自 llm_config.json 或
    BRFL_LLM_* 环境变量（见 pyutils/llm_config.py），代码里不写死任何密钥。
    """
    messages = [
        {"role": "system", "content": "You are an assistant that only outputs valid JSON."},
        {"role": "user", "content": prompt},
    ]
    return call_llm(messages, model=model, temperature=0.0)


# 旧函数名保留为别名，避免外部脚本调用失效。
def call_deepseek_chat(prompt: str, model: str = None) -> str:
    return call_chat_model(prompt, model=model)

def llm_match_passed_for_failed(
        failed_item: Dict[str, Any],
        candidate_passed: List[Dict[str, Any]],
        variant_root: Path,
        top_k: int,
        dbg_dir: Path,
) -> List[str]:
    """
    对于单个 failing test，用 deepseek 从 candidate_passed 中选出 top_k 个“最相似的通过测试”。
    返回值是若干 "estest_class::test_method" 字符串。
    """
    f_desc = build_test_desc({**failed_item, "status": "failed"})

    cand_entries = []
    for idx, it in enumerate(candidate_passed):
        desc = build_test_desc({**it, "status": "passed"})
        est = it["estest_class"]
        tm = it["test_method"]
        cand_entries.append({
            "id": f"{est}::{tm}",
            "description": desc,
        })

    # 为了避免 prompt 过长，必要时可以只取前 N 个候选
    MAX_CAND = 50
    cand_entries = cand_entries[:MAX_CAND]

    # 组装 JSON 格式的候选列表
    cand_json = json.dumps(cand_entries, ensure_ascii=False, indent=2)

    # prompt 约定 LLM 只输出 JSON，便于解析
    prompt = textwrap.dedent(f"""
    You are given one failing test and a list of passing tests for the same software product.
    Your task is to select up to {top_k} passing tests that are most similar to the failing test
    in terms of behavior / scenario / covered functionality.

    The failing test:
    {f_desc}

    The candidate passing tests (JSON array):
    {cand_json}

    Please output ONLY a JSON object in the following format:
    {{
      "failed_test": "<estest_class>::<test_method>",
      "matched_passed": [
        "<estest_class>::<test_method>",
        ...
      ]
    }}

    Rules:
    - "matched_passed" must contain at most {top_k} distinct IDs taken from the candidate list.
    - If you cannot find any reasonable match, return an empty array for "matched_passed".
    - Do not add any extra keys or explanations outside this JSON object.
    """)

    raw_resp = call_chat_model(prompt)

    # 调试写盘：方便你后面看 LLM 行为
    dbg_file = dbg_dir / f"llm_match_{failed_item['estest_class']}__{failed_item['test_method']}.json"
    dbg_file.write_text(raw_resp, encoding="utf-8")

    try:
        resp_obj = json.loads(raw_resp)
    except json.JSONDecodeError as e:
        print(f"[llm_match_passed_for_failed][WARN] JSON parse error: {e}")
        return []

    matched = resp_obj.get("matched_passed", [])
    if not isinstance(matched, list):
        return []

    # 做一个简单的清洗，防止超过 top_k
    matched = [str(x) for x in matched][:top_k]
    return matched

def collect_items_passed(
    passed_dir: Path,
    failed_items: List[Dict[str, Any]],
    variant_root: Path,
    # 模型 / 温度 / top_k / API Key：留空表示用 llm_config.json 或 BRFL_LLM_* 环境变量
    model_name: str = None,
    top_k: int = None,
    temperature: float = None,
    api_key: str = None,
    # 持久化缓存身份；为 None 时从 variant_root 路径推导
    # （system=数据集目录名, mutant=祖父目录, variant=叶子目录）。
    system: str = None,
    mutant: str = None,
    variant: str = None,
) -> List[Dict[str, Any]]:
    """
    使用大模型，从所有 passed 测试中，为每个 failing 测试在“同一个测试类”内
    选出少量反事实性质的通过用例。

    使用哪个厂商 / 模型 / 端点 / Key，由 llm_config.json（或 BRFL_LLM_* 环境变量）
    决定，见 pyutils/llm_config.py；代码中不含任何密钥。

    返回：被选中的 passed item 列表（item 结构与 collect_items 一致）。

    LLM 的选择结果会被持久化到 <project_root>/llm_selection_cache/ 下，
    （路径与变体目录无关，不会因变体被删除而丢失）。再次运行时若命中缓存，
    则直接复用、不再调用大模型，避免重复产生 API 费用。
    """
    passed_items_all = collect_items(passed_dir)

    failed_by_class, passed_by_class = group_items_by_class(
        failed_items, passed_items_all
    )

    # ---- 持久化 LLM 缓存（重跑实验时复用，省去大模型费用） ----
    use_cache = cache_enabled()
    variant_cache: Dict[str, Dict[str, List[str]]] = {}
    cache_dirty = False
    if use_cache:
        c_system, c_mutant, c_variant = derive_cache_identity(variant_root, system=system)
        if mutant:
            c_mutant = mutant
        if variant:
            c_variant = variant
        variant_cache = load_variant_cache(c_system, c_mutant, c_variant)
        if variant_cache:
            print(f"[llm_cache] loaded {len(variant_cache)} cached test class(es) "
                  f"for {c_system}/{c_mutant}/{c_variant}")

    selected_passed: List[Dict[str, Any]] = []
    seen_passed_keys = set()  # 用于去重：(estest_class, test_method)

    def _apply_mapping(estest_class, mapping, f_list, p_list):
        passed_index = {it["test_method"]: it for it in p_list}
        for f_item in f_list:
            fname = f_item["test_method"]
            for mname in mapping.get(fname, []):
                key = (estest_class, mname)
                if key in seen_passed_keys:
                    continue
                seen_passed_keys.add(key)
                passed_item = passed_index.get(mname)
                if passed_item is not None:
                    selected_passed.append(passed_item)

    for estest_class, f_list in failed_by_class.items():
        p_list = passed_by_class.get(estest_class, [])
        if not p_list:
            # 这个测试类里没有任何通过测试，跳过
            continue

        # 命中缓存：复用已存的选择，跳过大模型
        if use_cache and estest_class in variant_cache:
            mapping = variant_cache[estest_class]
            print(f"[LLM][cache-hit] {estest_class}: reusing stored selection "
                  f"({sum(len(v) for v in mapping.values())} matches)")
            _apply_mapping(estest_class, mapping, f_list, p_list)
            continue

        print(f"[LLM] {estest_class}: {len(f_list)} failing, {len(p_list)} passing")

        test_src = load_test_file_source(variant_root, estest_class)

        try:
            mapping = llm_match_passed_for_failed_in_class(
                estest_class=estest_class,
                test_source=test_src,  # load_test_source_for_estest 读到的整份 .java
                failed_items_in_class=f_list,  # 该测试类下所有 failing items
                passed_items_in_class=p_list,  # 该测试类下所有 passing items
                model_name=model_name,
                temperature=temperature,
                top_k=top_k,
                api_key=api_key,  # 留空表示用 llm_config.json / 环境变量里的 Key
            )
        except Exception as e:
            print(f"[LLM][WARN] failed for {estest_class}: {e}")
            continue

        if use_cache:
            variant_cache[estest_class] = mapping
            cache_dirty = True

        _apply_mapping(estest_class, mapping, f_list, p_list)

    if use_cache and cache_dirty:
        save_variant_cache(c_system, c_mutant, c_variant, variant_cache)

    print(f"[collect_items_passed] selected {len(selected_passed)} passed tests via LLM")
    return selected_passed

def make_test_name_for_oracle(sut_class: str, test_method: str) -> str:
    # sut_class 形如 "ElevatorSystem.Elevator"
    # test_method 形如 "test53"
    base = f"{sut_class}_{test_method}"             # ElevatorSystem.Elevator_test53
    if "." in base:
        cls, meth = base.split(".", 1)              # "ElevatorSystem", "Elevator_test53"
        return f"{cls}::{meth}"                     # ElevatorSystem::Elevator_test53
    return f"::{base}"                              # default-package class (TankWar): empty pkg, keep '::'

def getsplcmdline(variant_path: str, variant: str, debug: bool = True) -> List[str]:
    vr = Path(variant_path).resolve()
    dbg = vr / ".smartfl-debug"
    dbg.mkdir(parents=True, exist_ok=True)

    failed_items = collect_items(vr / "coverage" / "failed")
    passed_items = collect_items(vr / "coverage" / "passed")
    # passed_items = collect_items_passed(
    #     vr / "coverage" / "passed",
    #     failed_items=failed_items,
    #     variant_root=vr,
    #     model_name="qwen-plus",
    #     top_k=5,
    # )

    print(f"[getsplcmdline] failed items:  {len(failed_items)}")
    print(f"[getsplcmdline] passed items:  {len(passed_items)}")

    if not failed_items:
        return []

    # ---------- tests-oracle.csv ----------
    oracle_rows: List[str] = []
    oracle_seen: set = set()   # 只控制 oracle 行去重

    def _register_oracle(it: dict, status: str):
        # status: "FAIL" / "PASS"
        sut = it["sut_class"]
        tmeth = it["test_method"]
        test_name = make_test_name_for_oracle(sut, tmeth)
        if test_name in oracle_seen:
            return
        oracle_seen.add(test_name)
        oracle_rows.append(f"{test_name},{status}")

    all_items = failed_items + passed_items

    # 2) 收集 SUT 类
    sut_from_src = _collect_sut_classes_from_src(vr / "src")
    sut_from_cov = sorted({it["sut_class"] for it in all_items})
    sut_classes  = sorted(set(sut_from_src) | set(sut_from_cov))
    print(f"[getsplcmdline] SUT classes collected: {len(sut_classes)}")

    # inst_list   = sut_classes
    test_classes = sorted({it["estest_class"] for it in all_items})
    inst_list = sorted(set(sut_classes) | set(test_classes))
    instclasses = ":".join(inst_list)

    # instclasses = ":".join(inst_list)
    (dbg / "instclasses.txt").write_text("\n".join(inst_list[:1000]), encoding="utf-8")

    # 3) classpath：build/main + build/test + JUnit + Clover
    classesdir_main = (vr / "build" / "main").resolve()
    classesdir_test = (vr / "build" / "test").resolve()

    if not classesdir_main.exists():
        raise RuntimeError(f"[classpath] {classesdir_main} not found")

    cp_parts = [str(classesdir_main)]
    if classesdir_test.exists():
        cp_parts.append(str(classesdir_test))

    if not Path(JUNIT_STANDALONE_JAR).exists():
        raise RuntimeError(f"[classpath] JUnit jar not found: {JUNIT_STANDALONE_JAR}")
    if not Path(CLOVER_RUNTIME_JAR).exists():
        raise RuntimeError(f"[classpath] Clover runtime jar not found: {CLOVER_RUNTIME_JAR}")
    if not Path(EVOSUITE_RUNTIME_JAR).exists():
        raise RuntimeError(f"[classpath] EvoSuite runtime jar not found: {EVOSUITE_RUNTIME_JAR}")

    cp_parts.extend([
        JUNIT_STANDALONE_JAR,
        CLOVER_RUNTIME_JAR,
        EVOSUITE_RUNTIME_JAR,  # ★ 新增
        SLF4J_SIMPLE_JAR,
    ])
    cp = os.pathsep.join(cp_parts)
    quoted_cp = shlex.quote(cp)

    print("[classpath parts]")
    print("   ", classesdir_main)
    if classesdir_test.exists():
        print("   ", classesdir_test)
    print("   ", JUNIT_STANDALONE_JAR)
    print("   ", CLOVER_RUNTIME_JAR)
    print("   ", EVOSUITE_RUNTIME_JAR)
    agent_extra = f",classesdir={classesdir_main}"

    cmds: List[str] = []

    def _build_cmd(it: dict) -> str:
        """根据一个 coverage item 构造一条 ConsoleLauncher 命令"""
        sut   = it["sut_class"]
        tmeth = it["test_method"]
        est   = it["estest_class"]

        logfile_name = f"{sut}_{tmeth}.log"

        agent_arg = (
            f'-noverify -javaagent:{AGENT_JAR}='
            f'instrumentingclass={instclasses},logfile={logfile_name},project={variant}{agent_extra}'
        )

        if _class_exists([classesdir_main, classesdir_test], est):
            cmd = (
                f'{shlex.quote(JAVA_BIN)} {agent_arg} -cp {quoted_cp} '
                f'org.junit.platform.console.ConsoleLauncher '
                f'--details=tree --select-method {est}#{tmeth}'
            )
        else:
            cmd = (
                f'{shlex.quote(JAVA_BIN)} {agent_arg} -cp {quoted_cp} '
                f'org.junit.platform.console.ConsoleLauncher '
                f'--details=tree --scan-class-path --fail-if-no-tests'
            )

        return cmd

    # 4) 构造命令（去重：同一个 est/test_method 只跑一次）
    seen_tests: set = set()  # (estest_class, test_method)

    # failing tests
    for it in failed_items:
        key = (it["estest_class"], it["test_method"])
        if key in seen_tests:
            continue
        seen_tests.add(key)
        _register_oracle(it, "FAIL")
        cmds.append(_build_cmd(it))

    # passing tests
    for it in passed_items:
        key = (it["estest_class"], it["test_method"])
        if key in seen_tests:
            continue
        seen_tests.add(key)
        _register_oracle(it, "PASS")
        cmds.append(_build_cmd(it))

    (dbg / "first_cmd.txt").write_text(cmds[0] if cmds else "<none>", encoding="utf-8")
    print(f"[getsplcmdline] generated {len(cmds)} commands for variant={variant}")

    # 写 oracle（注意这里路径要和 SplGraphBuilder 对齐）
    oracle_path = vr / "tests-oracle.csv"   # 若你已把 SplGraphBuilder 改成 variantRoot + "/tests-oracle.csv"
    oracle_path.write_text("\n".join(oracle_rows) + "\n", encoding="utf-8")

    return cmds



def checkout(proj: str, id: str):
    checkoutpath = f'{checkoutbase}/{proj}/{id}'
    if not(os.path.exists(checkoutpath)):
        os.makedirs(checkoutpath)
    if not(os.path.exists(checkoutpath + '/.defects4j.config')):
        print("in checkout")
        os.system(
            f'defects4j checkout -p {proj} -v {id}b -w {checkoutbase}/{proj}/{id} >/dev/null 2>&1')


def deletecheckout(proj: str, id: str):
    checkoutpath = f'{checkoutbase}/{proj}/{id}'
    if (os.path.exists(checkoutpath)):
        os.system(f'rm -rf {checkoutbase}/{proj}/{id}')


def cleanupcheckout(proj: str, id: str):
    checkoutpath = f'{checkoutbase}/{proj}/{id}'
    if (os.path.exists(checkoutpath)):
        os.system(f'rm -rf {checkoutpath}/trace/logs/mytrace/')
        os.system(f'rm -rf {checkoutpath}/trace/logs/run/')
        os.system(f'rm -rf {checkoutpath}/trace/classcache/')


def clearcache(proj: str, id: str):
    cachepath = f'./d4j_resources/metadata_cached/{proj}/{id}.*'
    os.system('rm '+cachepath)


def list_dir(current_dir, full_path=False, sort=False):
    files = list(filter(lambda d: not d.startswith("."), os.listdir(current_dir)))
    if full_path:
        files = [os.path.join(current_dir, file) for file in files]
    if sort:
        files.sort()
    return files


def getFaildProductNames(csv_path):
    # 读取并做稳健清洗：保留字符串、去除BOM/空白
    df = pd.read_csv(csv_path, dtype=str, engine="python")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]  # 列名清洗
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # 单元格去空白

    product_col = "Product\\Feature"
    test_col = "__TEST_OUTPUT__"

    # 过滤 __FAILED__ 并取产品名
    failed_products = (
        df.loc[df[test_col] == "__FAILED__", product_col]
        .dropna()
        .tolist()
    )
    return failed_products


def _run_one(cmd_cwd):
    cmd, cwd = cmd_cwd
    print("[RUN]", cmd)  # 先别重定向到 /dev/null，方便看错误
    return subprocess.run(cmd, shell=True, cwd=cwd).returncode

def _run_and_capture(cmd: str, cwd: Path, dbg_dir: Path) -> int:
    """
    运行一条 Java 测试命令：
    - 从命令行中解析 logfile=<XXX>.log
    - stdout 分两路：
        * 原始 stdout 全量保存到 dbg_dir 下面
        * 只把以 'opcode' 或 '###' 开头的行写入真正的 trace .log
    - stderr 写入 dbg_dir 下面
    """
    print("[RUN]", cmd)

    # 1) 解析 logfile 名
    m = re.search(r"logfile=([^,\s]+)", cmd)
    logfile_name = m.group(1) if m else "unknown.log"

    # 2) 构造路径
    log_dir = cwd / "trace" / "logs" / "run"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / logfile_name

    dbg_dir.mkdir(parents=True, exist_ok=True)
    raw_stdout_path = dbg_dir / f"{logfile_name}.stdout.raw.txt"
    stderr_path     = dbg_dir / f"{logfile_name}.stderr.txt"

    # 3) 行级过滤输出
    with log_path.open("w", encoding="utf-8") as log_f, \
         raw_stdout_path.open("w", encoding="utf-8") as raw_out_f, \
         stderr_path.open("w", encoding="utf-8") as err_f:

        p = subprocess.Popen(
            cmd,
            shell=True,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # 行缓冲
        )

        # stdout：一行一行读
        for line in p.stdout:
            raw_out_f.write(line)     # 原样保存一份

            # 只保留“插桩行”到真正的 trace log
            if line.startswith("opcode") or line.startswith("###"):
                log_f.write(line)

        # stderr 一次性收集
        err = p.stderr.read()
        if err:
            err_f.write(err)

        p.wait()

    print(f"[RUN] rc={p.returncode}, log={log_path}, raw_stdout={raw_stdout_path}")
    return p.returncode

def runspl(mutant, buggy_systems_folder, debug=True):
    mutant_path = os.path.join(buggy_systems_folder, mutant)
    failed_products = getFaildProductNames(os.path.join(mutant_path, "config.report.csv.done"))
    variants_dir = os.path.join(mutant_path, "variants")
    stmt2id = {}
    for variant in failed_products:
        variant_path = os.path.join(variants_dir, variant)
        failed_spectrum_file_path = join_path(variant_path, "coverage/spectrum_failed_coverage.xml")
        stmt2id = get_spectrum_stm(failed_spectrum_file_path, stmt2id)
    failed_trace = {}
    for variant in failed_products:
        variant_path = os.path.join(variants_dir, variant)
        variant_root = Path(variant_path).resolve()
        dbg_dir = variant_root / ".smartfl-debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        trace_dir = variant_root / "trace"
        if trace_dir.exists():
            # print(f"  [CLEAN] remove {trace_dir}")
            shutil.rmtree(trace_dir)
        # ✅ 确保 Agent 期望的目录存在
        run_dir = variant_root / "trace" / "logs" / "run"
        mytrace_dir = variant_root / "trace" / "logs" / "mytrace"
        run_dir.mkdir(parents=True, exist_ok=True)
        mytrace_dir.mkdir(parents=True, exist_ok=True)

        # 环境自检
        (dbg_dir / "env.txt").write_text(
            f"PATH={os.environ.get('PATH')}\n"
            f"JAVA_BIN={JAVA_BIN}\n"
            f"which_java={shutil.which('java')}\n",
            encoding="utf-8"
        )
        failed_trace[variant] = gererateObDataFromSB(variant_path, stmt2id)
        cmds = getsplcmdline(variant_path, variant, debug=debug)
        if not cmds:
            print(f"[runspl] no commands for {variant}")
            continue

        # ✅ 把所有命令存下来方便检查
        (dbg_dir / "all_cmds.txt").write_text(
            "\n\n".join(cmds), encoding="utf-8"
        )

        # ✅ 依次执行每一条命令（每个 failing test 一条命令）
        for i, cmd in enumerate(cmds, 1):
            print(f"[RUN {i}/{len(cmds)}] {cmd}")
            rc = _run_and_capture(cmd, variant_root, dbg_dir)

            # JUnit ConsoleLauncher 约定：
            #   rc = 0 : 所有测试通过
            #   rc = 1 : 有测试失败（在本场景下是预期行为，因为这些就是 failing tests）
            if rc not in (0, 1):
                print(f"[runspl][ERR] command {i} failed (rc={rc}). Stop to inspect.")
                # 你可以选择 return 或者 break，这里我先 break 只退出当前 variant
                break
            if rc == 1:
                print(f"[runspl][WARN] tests failed in command {i} (rc=1), "
                      f"treat as expected failing tests and continue.")

        # ✅ 最后统计一下该 variant 生成了多少个 .log
        log_dir = variant_root / "trace" / "logs" / "run"
        count = len([f for f in log_dir.glob("*.log")])
        print(f"[runspl] logs in {variant}: {count} files -> {log_dir}")
    return failed_trace
        # rewrite_logs_for_variant(variant_root)

def getdirsize(start_path: str) -> int:
    """递归计算目录大小（字节数）"""
    total_size = 0
    start = Path(start_path)
    if not start.exists():
        return 0
    for p in start.rglob("*"):
        if p.is_file():
            try:
                total_size += p.stat().st_size
            except OSError:
                pass
    return total_size

def is_child_switch(switch, target_switch):
    if len(switch.intersection(target_switch)) == len(switch):
        return True
    return False

def satisfy_spc_minimality(current_SPC, SPC_set):
    for added_SPC in SPC_set:
        if is_child_switch(added_SPC, current_SPC):
            return False
    return True

def eucliDist(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))

def find_switched_feature_selections(failed_config, passed_config):
    switched_feature_selections = set()
    for feature_position, (failed_config_fs, passed_config_fs) in enumerate(zip(failed_config, passed_config)):
        if failed_config_fs != passed_config_fs:
            feature_selection = f"{feature_position}_{failed_config_fs}"
            switched_feature_selections.add(feature_selection)
    return switched_feature_selections

def minimize_switches(switches):
    switches.sort(key=lambda s: len(s), reverse=False)
    for i, current_switch in enumerate(switches):
        for target_switch in switches[i + 1:]:
            if is_child_switch(current_switch, target_switch):
                switches.remove(target_switch)
    return switches

def split_positioned_feature_selection(positioned_fs):
    # positioned_fs 形如 "3_True" 或 "5_False"
    feature_position_str, fs_str = positioned_fs.split("_", 1)
    feature_position = int(feature_position_str)

    fs_str = fs_str.strip()
    if fs_str in ("True", "true", "1"):
        fs = True
    elif fs_str in ("False", "false", "0"):
        fs = False
    else:
        # 如果遇到奇怪格式，可以视情况改成 return False 或直接 raise
        raise ValueError(f"Unexpected feature selection flag: {fs_str}")

    return feature_position, fs

def exist_configs_contain_spc(SPC, configs):
    has_configs_contain_spc = False
    for fc in configs:
        valid_fs = []
        for spc_fs in SPC:
            feature_position, config_fs = split_positioned_feature_selection(spc_fs)   #feature_position为数字，表示第几个特征；config_fs为bool类型，表示该特征的选择
            if fc[feature_position] == config_fs:
                valid_fs.append(True)
        if len(valid_fs) == len(SPC):   #SPC为configs的子集
            has_configs_contain_spc = True
            break

    return has_configs_contain_spc

def union_all_switched_feature_selections(switches):
    switched_feature_selections = switches[0].union(*switches[1:])      #union并集  intersection交集  difference差集
    return switched_feature_selections

def satisfy_spc_necessity(SPC, passed_configs, failed_configs):
    return exist_configs_contain_spc(SPC, failed_configs) and not exist_configs_contain_spc(SPC, passed_configs)

def remove_subsets(input_list):
    # 创建一个空列表来存储结果
    result = []

    # 遍历输入列表中的每个集合
    for current_set in input_list:
        is_subset = False

        # 检查当前集合是否是其他集合的子集
        for other_set in input_list:
            if len(current_set) > len(other_set):
                continue
            if current_set != other_set and current_set.issubset(other_set):
                is_subset = True
                break

        # 如果当前集合不是任何其他集合的子集，则将其添加到结果列表中
        if not is_subset:
            result.append(current_set)

    return result

def combine_spc_with_feature_names(feature_names, current_SPC):
    new_SPC = {}
    for spc_fs in current_SPC:
        feature_position, fs = split_positioned_feature_selection(spc_fs)
        current_feature_name = feature_names[feature_position]
        if fs is True:
            new_SPC[current_feature_name] = fs
        else:
            new_SPC["#" + current_feature_name] = fs
    return ",".join(new_SPC.keys())

def find_failed_configs_contains_spc(current_SPC, failed_configs):
    needing_failed_configs = []
    for fc in failed_configs:
        valid_fs = []
        for spc_fs in current_SPC:
            feature_position, config_fs = split_positioned_feature_selection(spc_fs)
            if fc[feature_position] == config_fs:
                valid_fs.append(True)
        if len(valid_fs) == len(current_SPC):
            needing_failed_configs.append(fc)

    return needing_failed_configs

def detect_SPCs(system, feature_names, passed_configs, failed_configs, variant_names, variants_dir, spc_log_file_path, C_relevance_dict, df):
    SPC_set = []
    switches_list = []
    Cache_set = set()
    saved_counter=0
    total_counter=0
    inclusion_rate = 0
    fl = len(feature_names)
    if (len(passed_configs) == 0 or len(failed_configs) == 0):
        spc_file = open(spc_log_file_path, "w")
        spc_file.close()
        return spc_log_file_path
    else:
        experience_dis = {"Email": 2.0, "Elevator": 2.0, "ExamDB": 2.5, "GPL": 2.5}
        logger.info(f"Finding SPCs and write to [{get_file_name_with_parent(spc_log_file_path)}]")
        with open(spc_log_file_path, "w+") as spc_log_file:
            # Core Algorithm
            for current_failed_config in failed_configs:
                switches = []
                current_failed_config_name = variant_names[tuple(current_failed_config)]
                position_current_failed_config = df.loc[df['Product\Feature']==current_failed_config_name]
                position_current_failed_config = position_current_failed_config.iloc[0, 2:].tolist()
                X = np.array(position_current_failed_config)
                X_with_each_failed_config_dis = {}
                avg_dis = 0
                for current_passed_config in passed_configs:
                    current_passed_config_name = variant_names[tuple(current_passed_config)]
                    position_current_passed_config = df.loc[df['Product\Feature'] == current_passed_config_name]
                    position_current_passed_config = position_current_passed_config.iloc[0, 2:].tolist()
                    Y = np.array(position_current_passed_config)
                    X_with_each_failed_config_dis[tuple(current_passed_config)] = eucliDist(X, Y)
                    avg_dis += X_with_each_failed_config_dis[tuple(current_passed_config)]
                X_with_each_failed_config_dis = sorted(X_with_each_failed_config_dis.items(), key=lambda x: x[1])
                X_with_each_failed_config_dis = dict(X_with_each_failed_config_dis)
                if system in experience_dis:
                    avg_dis = experience_dis[system]
                else:
                    avg_dis= round(avg_dis/len(X_with_each_failed_config_dis), 5)
                for current_passed_config in X_with_each_failed_config_dis:
                    if X_with_each_failed_config_dis[current_passed_config] < avg_dis:
                        current_switch = find_switched_feature_selections(current_failed_config,
                                                                      current_passed_config)

                        switches.append(current_switch)

                if len(switches) > 0:
                    switches = minimize_switches(switches)
                    switched_feature_selections = union_all_switched_feature_selections(switches)  # 各个switch的特征交集
                    switches_list.append(switched_feature_selections)
                else:
                    for current_passed_config in passed_configs:
                        current_switch = find_switched_feature_selections(current_failed_config,
                                                                          current_passed_config)
                        switches.append(current_switch)
                    if len(switches) > 0:
                        switches = minimize_switches(switches)
                        switched_feature_selections = union_all_switched_feature_selections(
                            switches)
                        switches_list.append(switched_feature_selections)
            cached_spc = []
            old_switches = switches_list
            switches_list = remove_subsets(switches_list)
            inclusion_rate = float((len(old_switches) - len(switches_list)) / len(old_switches))
            nway_spc_number = []
            for k in range(1, 8):
                inner_start = time.time()
                spc_number = 0
                for i, switched_feature_selected in enumerate(switches_list):

                    kth_set = set(combinations(switched_feature_selected, k))
                    for item in kth_set:
                        current_SPC = set(item)
                        total_counter += 1
                        if len(C_relevance_dict.intersection(current_SPC)) <= 0:
                            saved_counter += 1
                            continue
                        if tuple(current_SPC) in Cache_set:
                            saved_counter += 1
                            continue
                        spc_number += 1
                        Cache_set.add(tuple(current_SPC))
                        if satisfy_spc_minimality(current_SPC, SPC_set) and satisfy_spc_necessity(current_SPC,
                                                                                                  passed_configs,
                                                                                                  failed_configs):
                            combined_spc = combine_spc_with_feature_names(feature_names, current_SPC)
                            if combined_spc.strip() and combined_spc not in cached_spc:
                                # minimized_failed_config = find_minimized_failed_config_contains_spc(current_SPC,
                                #                                                                     failed_configs)
                                spc_failed_configs = find_failed_configs_contains_spc(current_SPC,
                                                                                      failed_configs)
                                for spc_config in spc_failed_configs:
                                    spc_log_file.write(
                                        f"{combined_spc}; {get_src_dir(join_path(variants_dir, variant_names[tuple(spc_config)]))}\n")
                                cached_spc.append(combined_spc)
                            SPC_set.append(current_SPC)
                nway_spc_number.append(spc_number)
            if total_counter > 0:
                duplication_rate = (float(saved_counter / total_counter))
            else:
                duplication_rate = 0
                Cache_set = set()
            # with open("/home/whn/codes/VARCOP-gh-pages/SPC_set.txt", 'a') as file:
            #     file.write(str(SPC_set)+'\n')

            return spc_log_file_path, (total_counter - saved_counter), nway_spc_number, inclusion_rate, duplication_rate

def su_to_feature_priors(selected_featured_name):
    """
    输入:
        selected_featured_name : dict[str, float]
            形如 {"Base": su_base, "Weight": su_weight, ...}
            su 为 Symmetrical Uncertainty (>=0)

    输出:
        priors_correct : dict[str, float]
            形如 {"Base": p_correct_base, ...}
            表示在因子图里使用的 P(feature 正确) 先验，范围约为 [0.5, 1.0]
    """
    if not selected_featured_name:
        return {}

    # 转 float，避免 numpy 类型干扰
    items = [(k, float(v)) for k, v in selected_featured_name.items()]
    scores = [s for _, s in items]
    s_min = min(scores)
    s_max = max(scores)

    priors_correct = {}

    # 如果所有 SU 一样，没有区分度 => 给统一的 0.5
    if s_max == s_min:
        for k, _ in items:
            priors_correct[k] = 0.5
        return priors_correct

    for k, s in items:
        # 归一化到 [0,1]
        r = (s - s_min) / (s_max - s_min)  # SU 最小 -> 0, SU 最大 -> 1

        # “故障先验”最多 0.5（再高就太极端了）
        p_faulty = 0.5 * r                 # in [0, 0.5]

        # 图里用的是“正确先验”
        p_correct = 1.0 - p_faulty         # in [0.5, 1.0]

        priors_correct[k] = p_correct

    return priors_correct

def find_SPCs(mutated_project_dir):      #SPCs = Suspicious Partial Configurations
    # 原有路径和覆盖信息
    config_report_path = get_model_configs_report_path(mutated_project_dir)
    import pandas as pd

    # 读取完整的 config.report.csv，保留 Product\Feature 这一列
    df_full = pd.read_csv(config_report_path)

    # 复制一份用于数值化预处理：第一列是 Product\Feature，其余列是特征 + __TEST_OUTPUT__
    df_num = df_full.copy()

    # 从第二列开始都是需要数值化的列
    feature_cols = df_num.columns[1:]

    # 将 T/F 和 PASSED/FAILED（以及带空格版本）统一映射成 0/1
    replace_dict = {
        'T': 1,
        'F': 0,
        '  T  ': 1,
        '  F  ': 0,
        '__PASSED__': 1,
        '__FAILED__': 0,
    }
    df_num[feature_cols] = df_num[feature_cols].replace(replace_dict)

    # 强制转成 int，确保后续不会再有 object/字符串
    df_num[feature_cols] = df_num[feature_cols].astype(int)

    # 这里打印一下 df_num，确认已经是数值 + 保留 Product\Feature
    print("df:\n", df_num)

    # ===== SU 计算部分：用已经数值化的 df_num 做 =====
    # 丢掉第一列 Product\Feature，只保留数值列
    df_data = df_num.iloc[:, 1:]
    columns = df_data.columns

    x = df_data[columns[:-1]]          # 特征列（0/1）
    y = df_data[columns[-1]]           # 标签列（0/1）

    dim = x.shape[1]
    columns_name = x.columns
    C_relevance_dict = set()
    selected_featured_name = {}

    for i in range(dim):
        tem = su_calculation(x.values[:, i], y.values)
        selected_featured_name[columns_name[i]] = tem
        print("")
        if tem > 0:
            C_relevance_dict.add(f"{i}_{True}")
            C_relevance_dict.add(f"{i}_{False}")

    # === 新增：SU → “温和”的特征先验，并写入 mutated_project_dir ===
    priors_correct = su_to_feature_priors(selected_featured_name)
    print("priors_correct: ", priors_correct)

    mutated_project_dir = Path(mutated_project_dir)
    mutated_project_dir.mkdir(parents=True, exist_ok=True)
    feat_prior_path = mutated_project_dir / "feature_su_priors.txt"

    with feat_prior_path.open("w", encoding="utf-8") as f:
        for feat, p in priors_correct.items():
            # 一行：<FeatureName> <p_correct>
            f.write(f"{feat} {float(p):.6f}\n")

    print("wrote feature priors to:", feat_prior_path)

    # return spc_log_file_path, spc_runtime, total_counter, nway_spc_number, inclusion_rate, duplication_rate

# # ===== 调用 detect_SPCs：传入的是 df_num（第一列是 Product\Feature，其余为 0/1）=====
#     spc_log_file_path, total_counter, nway_spc_number, inclusion_rate, duplication_rate = detect_SPCs(
#         "Elevator",
#         feature_names,
#         passed_configs,      # 保持原样传递（如果 detect_SPCs 用它只是当名字/索引，这样更安全）
#         failed_configs,
#         variant_names,
#         variants_dir,
#         spc_log_file_path,
#         C_relevance_dict,
#         df_num               # 关键：不再传 df_data，而是完整、数值化的 df_num
#     )
#
#     spc_runtime = time.time() - start_time


def parse_spl(PROJECT_BASE, root, mutant, cluster_members, buggy_systems_root: str, debug: bool = True):
    mutant_path = root / mutant
    config_csv = mutant_path / "config.report.csv.done"
    if not config_csv.exists():
        print(f"[parse_spl][SKIP] {mutant}: no config.report.csv")
        return

    failed_products = set(get_failed_product_names(str(config_csv)))

    if not failed_products:
        print(f"[parse_spl][INFO] {mutant}: no failed products")
        return

    variants_dir = mutant_path / "variants"
    if not variants_dir.exists():
        print(f"[parse_spl][WARN] {mutant}: variants dir not found: {variants_dir}")
        return

    print(f"\n[parse_spl] mutant={mutant}, failed_products={failed_products}")
    mutant_path_tem = join_path(buggy_systems_root, mutant)

    find_SPCs(mutant_path_tem)

    # get_sbfl_prio("BankAccountTP", mutant_path)


    # === 关键：按簇构图，而不是一次性构一个总图 ===
    # cluster_members: Dict[int, List[Tuple[variant, test_name]]]

    for cluster_id, members in cluster_members.items():
        # 过滤掉不在 failed_products 里的
        members = [(v, t) for (v, t) in members if v in failed_products]
        if not members:
            print(f"[parse_spl][INFO] {mutant} cluster={cluster_id}: no members after filtering")
            continue

        # 1) 这个簇涉及到哪些变体？
        variants_in_cluster = sorted({v for (v, _t) in members})

        # 2) 为该簇准备 variant_roots
        variant_roots = []
        for variant in variants_in_cluster:
            vr = variants_dir / variant
            if not vr.exists():
                print(f"[parse_spl][WARN] cluster={cluster_id} variant dir not found: {vr}")
                continue

            log_dir = vr / "trace" / "logs" / "run"
            size_mb = getdirsize(str(log_dir)) / (1024 * 1024)
            print(f"[parse_spl] cluster={cluster_id}, {variant}: trace size={size_mb:.2f}MB")
            if size_mb > 3000:
                print(f"[parse_spl][SKIP] cluster={cluster_id}, {variant}: trace too large (>3GB)")
                continue

            variant_roots.append(str(vr.resolve()))

        if not variant_roots:
            print(f"[parse_spl][INFO] {mutant} cluster={cluster_id}: no usable variants")
            continue

        # 3) 为该簇写一个“测试过滤文件”，供 SplGraphBuilder 使用
        #    里面你可以写成 "variant testName" 或 "traceFileName" 等，
        #    具体格式要和 Java 端约定好
        cluster_filter_file = mutant_path / f"cluster_{cluster_id}_tests.txt"
        with open(cluster_filter_file, "w", encoding="utf-8") as f:
            for v, t in members:
                f.write(f"{v} {t}\n")

        # 4) 组装传给 Java 的参数：
        #    建议：最后两个参数传 cluster_id 和 filter_file，前面是 variant_roots
        args_list = variant_roots + [str(cluster_id), str(cluster_filter_file)]
        args_str = " ".join(args_list)

        cmdline = (
            f'cd {PROJECT_BASE} && '
            'mvn compile -q && '
            'mvn exec:java '
            '-Dexec.mainClass=ppfl.spl.SplGraphBuilder '
            f"-Dexec.args='{args_str}' "
            '-Dexec.cleanupDaemonThreads=false '
            "-Dexec.jvmArgs='-Xms2g -Xmx4g'"
        )
        if not debug:
            cmdline += ' >/dev/null 2>&1'

        print(f"[parse_spl][RUN] mutant={mutant}, cluster={cluster_id}")
        os.system(cmdline)


def parse(proj: str, id: str, debug=True):
    path = f"{checkoutbase}/{proj}/{id}/trace/logs/run" #trace 运行日志目录
    size = getdirsize(path)/(1024*1024) #转成 MB
    if(size>3000):
        return
    cmdline = f'mvn compile -q && mvn exec:java "-Dexec.mainClass=ppfl.defects4j.GraphBuilder" "-Dexec.args={proj} {id}"'
    if(not debug):
        cmdline += '>/dev/null 2>&1'
    os.system(cmdline)

def merge_infresult_logs_and_rank(project_base, mutant, metric: str = None):
    """
    兼容 Java 输出：
      InfResult-<mutant>_Cluster<cid>_<metric>.log
    """
    project_base = Path(project_base).resolve()
    mytrace_dir = project_base / "trace" / "logs" / "mytrace"
    if not mytrace_dir.exists():
        print(f"[merge] mytrace dir not found: {mytrace_dir}")
        return []

    if metric:
        # ✅ 适配：_Cluster\d+_<metric>.log
        pattern = re.compile(
            rf"^InfResult-{re.escape(mutant)}_Cluster\d+_{re.escape(metric)}\.log$"
        )
    else:
        # 兼容旧：InfResult-<mutant>_Cluster\d+.log
        pattern = re.compile(
            rf"^InfResult-{re.escape(mutant)}_Cluster\d+\.log$"
        )

    log_files = [p for p in mytrace_dir.iterdir() if p.is_file() and pattern.match(p.name)]
    if not log_files:
        print(f"[merge] no InfResult logs found for mutant={mutant}, metric={metric}")
        return []

    stmt2prob = {}

    def canonical_stmt_key(raw_stmt: str) -> str:
        try:
            parts = raw_stmt.split('#')
            if len(parts) < 2:
                return raw_stmt
            sig = parts[0]
            line_str = parts[1]
            cls = sig.split(':', 1)[0]
            return f"{cls}.{line_str}"
        except Exception:
            return raw_stmt

    def is_test_stmt_key(stmt_key: str) -> bool:
        try:
            cls_part, _line = stmt_key.rsplit(".", 1)
        except ValueError:
            return False
        simple_name = cls_part.split(".")[-1].lower()
        return ("junit" in simple_name) or simple_name.endswith("test") or simple_name.endswith("tests") or ("_estest" in simple_name)

    for log_path in log_files:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or "prob_bp" not in line:
                    continue
                try:
                    before, after = line.split("prob_bp", 1)
                    raw_stmt = before.strip()
                    eq_idx = after.find("=")
                    if eq_idx == -1:
                        continue
                    prob_token = after[eq_idx + 1:].strip().split(",", 1)[0].strip().split()[0]
                    prob = float(prob_token)
                except Exception:
                    continue

                stmt_key = canonical_stmt_key(raw_stmt)
                if is_test_stmt_key(stmt_key):
                    continue

                old = stmt2prob.get(stmt_key)
                # prob_bp 越小越可疑：取最小
                if old is None or prob < old:
                    stmt2prob[stmt_key] = prob

    if not stmt2prob:
        print(f"[merge] no stmt probabilities parsed for mutant={mutant}, metric={metric}")
        return []

    return sorted(stmt2prob.items(), key=lambda x: x[1])

def clear_folder(folder):
    folder = os.path.abspath(folder)

    if not os.path.isdir(folder):
        raise ValueError(f"{folder} 不是一个有效的文件夹")

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

def _get_or_create_wb(path: str, metric_names):
    p = Path(path)
    if p.exists():
        wb = load_workbook(p)
    else:
        wb = Workbook()
        # 删除默认 Sheet
        if "Sheet" in wb.sheetnames and len(wb.sheetnames) == 1:
            wb.remove(wb["Sheet"])

    # 确保每个指标都有一个 sheet + 表头
    headers = ["mutated_project_name", "RANK", "EXAM"]
    for m in metric_names:
        sheet_name = str(m)[:31]  # Excel sheet 名最长 31
        if sheet_name not in wb.sheetnames:
            ws = wb.create_sheet(sheet_name)
            ws.append(headers)
        else:
            ws = wb[sheet_name]
            if ws.max_row == 0:
                ws.append(headers)
    return wb

def _append_rows(wb, metric, rows):
    ws = wb[str(metric)[:31]]
    for r in rows:
        ws.append(r)

def _autofit_basic(wb, metric_names):
    for m in metric_names:
        ws = wb[str(m)[:31]]
        for col in range(1, 4):  # 3 列
            maxlen = 0
            for row in ws.iter_rows(min_col=col, max_col=col, values_only=True):
                v = row[0]
                if v is None: continue
                maxlen = max(maxlen, len(str(v)))
            ws.column_dimensions[get_column_letter(col)].width = min(maxlen + 2, 60)


#@func_set_timeout(3600)
def fl(debug=True):
    args = parse_arguments()
    buggy_systems_folder = args.buggy_systems_folder
    system_name = args.system
    ProjectBase = Path(__file__).resolve().parent.parent

    sbfl_metrics = [TARANTULA, OCHIAI, OP2, BARINEL, DSTAR, KULCZYNSKI2, M2, HARMONIC_MEAN, ZOLTAR, GEOMETRIC_MEAN]#TARANTULA, OCHIAI, OP2, BARINEL, DSTAR, KULCZYNSKI2, M2, HARMONIC_MEAN, ZOLTAR, GEOMETRIC_MEAN

    if os.path.exists(buggy_systems_folder):
        delete_files_with_name(buggy_systems_folder, "cluster_0_tests.txt")
        delete_files_with_name(buggy_systems_folder, "cluster_1_tests.txt")
        delete_files_with_name(buggy_systems_folder, "cluster_2_tests.txt")
        delete_files_with_name(buggy_systems_folder, "feature_su_priors.txt")
        delete_files_with_name(buggy_systems_folder, "stmt_sbfl_priors.txt")

    root = Path(buggy_systems_folder).resolve()
    if not root.exists():
        print(f"[parse_spl][FATAL] root not found: {root}")
        return

    # ✅ 输出 xlsx（每个 metric 一个 sheet）
    result_dir = "./experimental_results"
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"rank_2BUG_results-{system_name}-withoutLLM2.xlsx")
    wb = _get_or_create_wb(result_path, sbfl_metrics)
    #detect_mutants = [12, 111, 118, 119, 120, 121]  #
    for mutant in sorted(os.listdir(root)):
        buggy_id = int(mutant.split('_')[-1])

        # if buggy_id not in detect_mutants:
        #     continue
        clear_folder("./trace/logs/mytrace")
        mutated_project_dir = join_path(buggy_systems_folder, mutant)

        # 1) 运行插桩/trace
        failed_trace = runspl(mutant, buggy_systems_folder)

        # 2) 聚类
        best_k, labels, cluster_members, best_score = cluster_failed_traces(failed_trace, min_k=2, max_k=3)
        if best_score < 0.75:
            labels = np.zeros_like(labels)
            all_ids = []
            for cid_members in cluster_members.values():
                all_ids.extend(cid_members)
            cluster_members = {0: all_ids}

        get_sbfl_prio(system_name, mutated_project_dir)
        parse_spl(ProjectBase, root, mutant, cluster_members, buggy_systems_folder, True)

        buggy_statements = get_multiple_buggy_statements(mutant, mutated_project_dir)
        all_stms_of_the_system, _ = get_executed_stms_of_the_system(mutated_project_dir, "", 0.1)
        all_stms = len(get_set_of_stms(all_stms_of_the_system))

        for metric in sbfl_metrics:
            sorted_results = merge_infresult_logs_and_rank(ProjectBase, mutant, metric=str(metric))

            rows_for_this_mutant_metric = []
            for buggy_statement in buggy_statements:
                for index, (key, value) in enumerate(sorted_results):
                    if key == buggy_statement:
                        while index < len(sorted_results) - 1 and sorted_results[index][1] == sorted_results[index + 1][1]:
                            index += 1
                        rank = index + 1
                        exam = (rank / all_stms) * 100.0
                        rows_for_this_mutant_metric.append([mutant, rank, exam])
                        break

            if rows_for_this_mutant_metric:
                _append_rows(wb, metric, rows_for_this_mutant_metric)
                print(f"[RESULT] metric={metric} appended {len(rows_for_this_mutant_metric)} rows for {mutant}")
            else:
                print(f"[RESULT] metric={metric} no buggy statements found for {mutant}")

        _autofit_basic(wb, sbfl_metrics)
        wb.save(result_path)
        mutated_project_Path = Path(mutated_project_dir).resolve()
        if mutated_project_Path.exists():
            # print(f"  [CLEAN] remove {trace_dir}")
            shutil.rmtree(mutated_project_Path)
    print("[DONE] wrote:", result_path)

def load_configs(config_report_path, variants_testing_coverage, filtering_coverage_rate):
    logger.info(f"Loading config report file [{get_file_name_with_parent(config_report_path)}]")
    with open(config_report_path) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        feature_names = header[1:]
        variant_names = {}
        passed_configs = []
        failed_configs = []
        for row in reader:
            current_variant_name, current_config, current_test_result = row[0], row[1:-1], row[-1]
            current_config = list(map(lambda fs: fs.strip() == "T", current_config))
            if current_test_result == "__NOASWR__":
                logger.fatal(f"Found untested variant [{current_variant_name}]")
            elif current_test_result == "__PASSED__":
                #if variants_testing_coverage[current_variant_name] >= filtering_coverage_rate:
                passed_configs.append(current_config)
                variant_names[tuple(current_config)] = current_variant_name
            else:
                failed_configs.append(current_config)
                variant_names[tuple(current_config)] = current_variant_name
        return feature_names, variant_names, passed_configs, failed_configs
