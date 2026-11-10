#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import shutil
import subprocess
from pathlib import Path

# ====== 路径配置：按你的环境改成一致的 ======
# Java 8 安装路径（和 s.py 里用的一致）
JAVA_HOME = "/usr/lib/jvm/java-8-openjdk-amd64"
JAVAC_BIN = os.path.join(JAVA_HOME, "bin", "javac")

# SMARTFL 工程根目录
PROJECT_BASE = "/home/whn/codes/SMARTFL"

# JUnit + Clover（和 s.py 里的保持一致）
JUNIT_STANDALONE_JAR = os.path.join(PROJECT_BASE, "lib", "junit-platform-console-standalone-1.10.2.jar")
CLOVER_RUNTIME_JAR   = os.path.join(PROJECT_BASE, "lib", "clover-runtime-4.4.1.jar")
EVOSUITE_RUNTIME_JAR   = os.path.join(PROJECT_BASE, "lib", "evosuite-standalone-runtime-1.0.6.jar")
# 缺陷数据集根目录（就是包含 _MultipleBugs_.NOB_1.ID_107 这一层的目录）
BUGGY_SYSTEMS_ROOT = "/home/whn/codes/datasets/4wise-ExamDB-1BUG-Full"

# ====== 工具函数 ======

def list_dir(path: str):
    """类似你 runspl 里的 list_dir：只要子目录名"""
    return sorted(
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    )

def get_failed_product_names(config_csv_path: str):
    """
    解析 config.report.csv，返回所有 __TEST_OUTPUT__ 为 __FAILED__ 的产品名（第一列）
    """
    failed = []
    if not os.path.exists(config_csv_path):
        print(f"[WARN] config.report.csv not found: {config_csv_path}")
        return failed

    with open(config_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"[WARN] empty csv: {config_csv_path}")
            return failed

        try:
            out_idx = header.index("__TEST_OUTPUT__")
        except ValueError:
            print(f"[WARN] '__TEST_OUTPUT__' column not found in {config_csv_path}")
            return failed

        for row in reader:
            if not row:
                continue
            status = row[out_idx].strip()
            if status == "__FAILED__":
                prod = row[0].strip()
                if prod:
                    failed.append(prod)

    return failed

def recompile_variant(variant_path: Path):
    """
    对单个产品：
      - 删除 build/
      - 删除 trace/
      - 重新编译 src 和 test 下所有 .java 到 build/main
    """
    print(f"[REBUILD] {variant_path}")

    build_dir = variant_path / "build"
    trace_dir = variant_path / "trace"

    # 1) 删除旧的 build 和 trace
    if build_dir.exists():
        print(f"  [CLEAN] remove {build_dir}")
        shutil.rmtree(build_dir)
    if trace_dir.exists():
        print(f"  [CLEAN] remove {trace_dir}")
        shutil.rmtree(trace_dir)

    # 2) 重新创建 build/main
    main_out = build_dir / "main"
    main_out.mkdir(parents=True, exist_ok=True)

    # 3) 自动探测源码根目录（既包含主程序也包含测试）
    src_roots = []
    for rel in ["src", "src/main/java", "src/java", "test", "src/test/java"]: #,
        cand = variant_path / rel
        if cand.exists():
            src_roots.append(cand)

    if not src_roots:
        print(f"[SKIP] no known src/test dirs under {variant_path}")
        return

    # 4) 收集所有 .java 源文件（路径相对 variant 根目录，便于 javac）
    java_files = []
    for root in src_roots:
        for p in root.rglob("*.java"):
            java_files.append(str(p.relative_to(variant_path)))

    if not java_files:
        print(f"[SKIP] no .java files in {src_roots}")
        return

    # 5) 组装编译 classpath（注意补上你需要的所有 jar）
    cp = os.pathsep.join([
        ".",
        JUNIT_STANDALONE_JAR,
        CLOVER_RUNTIME_JAR,
        EVOSUITE_RUNTIME_JAR,
        # 如果有 EvoSuite runtime / Hamcrest 等，也在这里加
        # EVOSUITE_RT_JAR,
        # HAMCREST_JAR,
    ])

    cmd = [
        JAVAC_BIN,
        "-g",
        "-encoding", "UTF-8",
        "-cp", cp,
        "-d", str(main_out),
    ] + java_files

    print("  [CMD]", " ".join(cmd))
    try:
        res = subprocess.run(
            cmd,
            cwd=str(variant_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
    except OSError as e:
        print(f"  [ERROR] failed to run javac: {e}")
        return

    if res.returncode == 0:
        print("  [OK] javac succeeded")
    else:
        print(f"  [FAIL] javac rc={res.returncode}")
        (variant_path / ".rebuild_stdout.txt").write_text(res.stdout, encoding="utf-8")
        (variant_path / ".rebuild_stderr.txt").write_text(res.stderr, encoding="utf-8")
        print("        stdout/stderr saved to .rebuild_stdout.txt / .rebuild_stderr.txt")


def main():
    root = Path(BUGGY_SYSTEMS_ROOT).resolve()
    if not root.exists():
        print(f"[FATAL] BUGGY_SYSTEMS_ROOT not found: {root}")
        return

    for mutant_name in list_dir(str(root)):
        # if mutant_name != "_MultipleBugs_.NOB_2.ID_1":
        #     continue
        mutant_path = root / mutant_name
        config_csv = mutant_path / "config.report.csv"
        if not config_csv.exists():
            try:
                shutil.copy(mutant_path / "config.report.csv.done", config_csv)
                print("文件复制成功！")
            except FileNotFoundError:
                print("文件不存在！")
            # print(f"[SKIP] {mutant_name}: no config.report.csv")
            # continue

        failed_products = get_failed_product_names(str(config_csv))
        if not failed_products:
            print(f"[INFO] {mutant_name}: no failed products")
            continue

        variants_dir = mutant_path / "variants"
        if not variants_dir.exists():
            print(f"[WARN] {mutant_name}: variants dir not found: {variants_dir}")
            continue

        print(f"\n[===== Mutant {mutant_name} =====]")
        print(f"[INFO] failed products: {failed_products}")

        for variant in failed_products:
            variant_path = variants_dir / variant
            if not variant_path.exists():
                print(f"[WARN] variant dir not found: {variant_path}")
                continue
            recompile_variant(variant_path)

if __name__ == "__main__":
    main()
