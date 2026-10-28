from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .domain import Configuration, JavaTestCase


PACKAGE = re.compile(r"\bpackage\s+([\w.]+)\s*;")
ASSERT_FAILURE = re.compile(r"expected:<(.+?)> but was:<(.+?)>")


def run(command: list[str], cwd: Path, log: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + "\n[STDERR]\n" + result.stderr, encoding="utf-8")
    return result


@dataclass(frozen=True)
class Build:
    main: Path
    tests: Path
    product_classes: tuple[str, ...]


@dataclass(frozen=True)
class Execution:
    outcome: str
    graph_path: Path
    log_path: Path
    expected: str | None
    actual: str | None


class BrflRuntime:
    def __init__(self, work_root: Path, java_home: Path):
        root = Path(__file__).resolve().parent.parent
        self.brfl = root / "vendor" / "brfl"
        self.bridge = root / "java" / "ppfl" / "CounterfactualInferenceExport.java"
        self.work_root = work_root.resolve()
        self.java = java_home / "bin" / "java.exe"
        self.javac = java_home / "bin" / "javac.exe"
        self.jar = java_home / "bin" / "jar.exe"
        self.lib = self.brfl / "lib"
        self.classes = self.work_root / "brfl-classes"
        self.agent = self.work_root / "brfl-agent.jar"
        for path in (self.java, self.javac, self.jar, self.lib, self.bridge):
            if not path.exists():
                raise FileNotFoundError(path)

    def build_brfl(self) -> None:
        bridge_class = self.classes / "ppfl" / "CounterfactualInferenceExport.class"
        if self.agent.is_file() and bridge_class.is_file():
            return
        self.classes.mkdir(parents=True, exist_ok=True)
        sources = sorted((self.brfl / "src" / "main" / "java").rglob("*.java"))
        response = self.work_root / "brfl-sources.txt"
        response.write_text("\n".join(x.relative_to(self.brfl).as_posix() for x in sources) + "\n", encoding="ascii")
        log = self.work_root / "brfl-build.log"
        result = run([str(self.javac), "-encoding", "UTF-8", "-cp", str(self.lib / "*"),
                      "-d", str(self.classes), "@" + str(response)], self.brfl, log)
        if result.returncode:
            raise RuntimeError(f"BRFL编译失败：{log}")
        result = run([str(self.javac), "-encoding", "UTF-8", "-cp",
                      os.pathsep.join((str(self.classes), str(self.lib / "*"))),
                      "-d", str(self.classes), str(self.bridge)], self.work_root, log)
        if result.returncode:
            raise RuntimeError(f"因子图导出程序编译失败：{log}")
        manifest = self.brfl / "src" / "main" / "resources" / "META-INF" / "MANIFEST.MF"
        result = run([str(self.jar), "cfm", str(self.agent), str(manifest),
                      "-C", str(self.classes), "."], self.work_root, log)
        if result.returncode:
            raise RuntimeError(f"插桩代理打包失败：{log}")

    def compile_configuration(self, config: Configuration, output: Path,
                              extra_test: Path | None = None) -> Build:
        main_out, test_out = output / "build" / "main", output / "build" / "test"
        main_sources = sorted((config.variant_dir / "src").rglob("*.java"))
        test_sources = sorted((config.variant_dir / "test").rglob("*.java"))
        if extra_test:
            test_sources.append(extra_test)
        if not main_sources or not test_sources:
            raise FileNotFoundError(f"缺少产品或测试源码：{config.variant_dir}")
        main_out.mkdir(parents=True, exist_ok=True)
        test_out.mkdir(parents=True, exist_ok=True)
        log = output / "compile.log"
        result = run([str(self.javac), "-g", "-encoding", "UTF-8", "-cp", str(self.lib / "*"),
                      "-d", str(main_out), *(str(x) for x in main_sources)], output, log)
        if result.returncode:
            raise RuntimeError(f"产品重编译失败：{log}")
        cp = os.pathsep.join((str(main_out), str(self.lib / "*")))
        result = run([str(self.javac), "-g", "-encoding", "UTF-8", "-cp", cp,
                      "-d", str(test_out), *(str(x) for x in test_sources)], output, log)
        if result.returncode:
            raise RuntimeError(f"测试重编译失败：{log}")
        classes = []
        for path in main_sources:
            source = path.read_text(encoding="utf-8", errors="replace")
            package = PACKAGE.search(source)
            classes.append((package.group(1) + "." if package else "") + path.stem)
        return Build(main_out, test_out, tuple(classes))

    def execute(self, config: Configuration, test: JavaTestCase, output: Path,
                build: Build) -> Execution:
        stage = output / "runs" / test.class_name.replace(".", "_") / test.method_name
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        (stage / "infer.txt").write_text(",".join(["0.5"] * 10) + "\n", encoding="ascii")
        oracle = test.identity
        cp = os.pathsep.join((str(build.main), str(build.tests), str(self.agent), str(self.lib / "*")))
        instrumented = ":".join(sorted(set(build.product_classes + (test.class_name,))))
        command = [str(self.java), "-noverify",
                   f"-javaagent:{self.agent}=instrumentingclass={instrumented},logfile={oracle}.log,project={config.name}",
                   "-cp", cp, "org.junit.platform.console.ConsoleLauncher", "--details=none",
                   "--select-method", f"{test.class_name}#{test.method_name}"]
        log = stage / "test-run.log"
        result = run(command, stage, log, timeout=180)
        if result.returncode not in {0, 1}:
            raise RuntimeError(f"测试运行器异常退出（{result.returncode}）：{log}")
        outcome = "passed" if result.returncode == 0 else "failed"
        if f"###{test.class_name}::{test.method_name}" not in result.stdout:
            raise RuntimeError(f"目标测试没有被执行：{log}")
        trace = stage / "run.log"
        trace.write_text("\n".join(x for x in result.stdout.splitlines() if x.startswith(("opcode", "###"))) + "\n",
                         encoding="utf-8")
        source_dir = stage / "trace" / "logs" / "mytrace"
        if not source_dir.is_dir() or not any(source_dir.glob("*.source.log")):
            raise RuntimeError(f"插桩没有生成静态语句日志：{log}")
        graph = stage / "factor-graph.tsv"
        export_cp = os.pathsep.join((str(self.classes), str(self.lib / "*")))
        export_log = stage / "factor-inference.log"
        exported = run([str(self.java), "-cp", export_cp, "ppfl.CounterfactualInferenceExport",
                        str(source_dir), str(trace), oracle, outcome.upper(), str(graph)], stage, export_log)
        if exported.returncode:
            raise RuntimeError(f"BRFL因果推理失败：{export_log}")
        match = ASSERT_FAILURE.search(result.stdout)
        status = {"configuration": config.name, "test": oracle, "outcome": outcome,
                  "expected": match.group(1) if match else None, "actual": match.group(2) if match else None}
        (stage / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return Execution(outcome, graph, log, status["expected"], status["actual"])
