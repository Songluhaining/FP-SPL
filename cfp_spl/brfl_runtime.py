"""管理移植的BRFL：构建代理、javac -g重编译、插桩执行、导出带后验的图。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .domain import Configuration, JavaTestCase
from .feature_sources import load_feature_map


PACKAGE = re.compile(r"\bpackage\s+([\w.]+)\s*;")
ASSERT_FAILURE = re.compile(r"expected:<(.+?)> but was:<(.+?)>")


def failure_signature(log: str) -> dict:
    """JUnit报告中的异常类型及首个产品栈帧；不从opcode文本猜异常。"""
    match = re.search(r"=>\s+([\w.$]+)(?::|\s|$)", log)
    if not match:
        return {}
    tail = log[match.end():]
    frame = re.search(r"^\s+([\w.$]+)\(([^():]+\.java):(\d+)\)", tail, re.MULTILINE)
    return {"exception": match.group(1), "product_method": frame.group(1) if frame else None,
            "line": int(frame.group(3)) if frame else None,
            "assertion": any(x in match.group(1) for x in ("AssertionError", "AssertionFailedError", "ComparisonFailure"))}


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
    source_digest: str = ""


@dataclass(frozen=True)
class Execution:
    outcome: str
    graph_path: Path | None
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
        """从本仓库vendor目录编译代理及导出桥接程序，不调用原BRFL仓库。"""
        bridge_class = self.classes / "ppfl" / "CounterfactualInferenceExport.class"
        sources = sorted((self.brfl / "src" / "main" / "java").rglob("*.java"))
        # 源码修改后必须重建，避免旧桥接程序掩盖结果标记等修复。
        digest = hashlib.sha256(b"".join(p.read_bytes() for p in [*sources, self.bridge])).hexdigest()
        stamp = self.work_root / "source-digest.txt"
        if (self.agent.is_file() and bridge_class.is_file() and stamp.is_file()
                and stamp.read_text(encoding="ascii") == digest):
            return
        self.classes.mkdir(parents=True, exist_ok=True)
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
        stamp.write_text(digest, encoding="ascii")

    def compile_configuration(self, config: Configuration, output: Path,
                              extra_test: Path | None = None) -> Build:
        """保留调试行号重编译产品与测试；extra_test为可选的生成测试源码。"""
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
        digest = hashlib.sha256(b"".join(p.read_bytes() for p in [*main_sources, *test_sources])).hexdigest()
        return Build(main_out, test_out, tuple(classes), digest)

    def execute(self, config: Configuration, test: JavaTestCase, output: Path,
                build: Build, require_graph: bool = True) -> Execution:
        """执行一个测试；require_graph=False仍插桩，但跳过构图和推理。"""
        # Windows长路径会让Java日志目录无法清理；身份在status中保存，目录用短摘要。
        stamp = self.work_root / "source-digest.txt"
        runtime_digest = stamp.read_text(encoding="ascii") if stamp.is_file() else "unbuilt"
        run_id = hashlib.sha256((test.identity + test.full_source + build.source_digest + runtime_digest).encode()).hexdigest()[:12]
        stage = output / "runs" / run_id / ("graph" if require_graph else "outcome")
        cached_status = stage / "status.json"
        if cached_status.is_file() and (not require_graph or (stage / "factor-graph.tsv").is_file()):
            status = json.loads(cached_status.read_text(encoding="utf-8"))
            return Execution(status["outcome"], stage / "factor-graph.tsv" if require_graph else None, stage / "test-run.log",
                             status.get("expected"), status.get("actual"))
        if stage.exists():
            def tolerate_disappearing_trace(_function, _path, error):
                if isinstance(error[1], FileNotFoundError):
                    return
                raise error[1]
            shutil.rmtree(stage, onerror=tolerate_disappearing_trace)
        stage.mkdir(parents=True)
        # 当前固定写入10个0.5参数，尚未接入频谱先验或区域张量推理。
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
        match = ASSERT_FAILURE.search(result.stdout)
        status = {"configuration": config.name, "test": oracle, "outcome": outcome,
                  "expected": match.group(1) if match else None, "actual": match.group(2) if match else None}
        (stage / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        if not require_graph:
            return Execution(outcome, None, log, status["expected"], status["actual"])
        source_dir = stage / "trace" / "logs" / "mytrace"
        if not source_dir.is_dir() or not any(source_dir.glob("*.source.log")):
            raise RuntimeError(f"插桩没有生成静态语句日志：{log}")
        graph = stage / "factor-graph.tsv"
        export_cp = os.pathsep.join((str(self.classes), str(self.lib / "*")))
        export_log = stage / "factor-inference.log"
        exported = run([str(self.java), "-cp", export_cp, "ppfl.CounterfactualInferenceExport",
                        str(source_dir), str(trace), oracle,
                        "PASS" if outcome == "passed" else "FAIL", str(graph),
                        "--structure-only"], stage, export_log)
        if exported.returncode:
            raise RuntimeError(f"BRFL因果推理失败：{export_log}")
        return Execution(outcome, graph, log, status["expected"], status["actual"])

    def infer_joint(self, executions: list[tuple[Configuration, JavaTestCase, Execution]],
                    output: Path) -> Path:
        """各轨迹只有结构解析；在Java桥接中共享特征源码语句后统一推理。"""
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, (config, test, execution) in enumerate(executions):
            stage = execution.log_path.parent.resolve()
            mapping = output / f"source-map-{index}.tsv"
            feature_map = load_feature_map(config.variant_dir)
            mapping.write_text("".join(f"{cls}#{line}\t{identity}\n"
                                       for (cls, line), identity in sorted(feature_map.items())), encoding="utf-8")
            rows.append("\t".join((f"e{index}:{config.name}",
                                    str(stage / "trace" / "logs" / "mytrace"),
                                    str(stage / "run.log"), test.identity,
                                    "PASS" if execution.outcome == "passed" else "FAIL", str(mapping))))
        manifest = output / "executions.tsv"
        manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        (output / "infer.txt").write_text(",".join(["0.5"] * 10) + "\n", encoding="ascii")
        graph = output / "joint-factor-graph.tsv"
        cp = os.pathsep.join((str(self.classes), str(self.lib / "*")))
        result = run([str(self.java), "-cp", cp, "ppfl.CounterfactualInferenceExport",
                      "--joint", str(manifest), str(graph)], output, output / "joint-inference.log", timeout=300)
        if result.returncode:
            raise RuntimeError(f"联合构图推理失败：{output / 'joint-inference.log'}")
        return graph
