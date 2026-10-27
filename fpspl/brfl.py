"""Run BRFL's Java instrumenter and export per-test runtime dependency factors.

All generated classes and traces live below the caller's output directory. The
original dataset and BRFL source tree are read-only inputs.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


TRACE_LINE = ("opcode", "###")
JAVA_SOURCE = re.compile(r"\bpackage\s+([\w.]+)\s*;")
ASSERT_FAILURE = re.compile(r"expected:<(.+?)> but was:<(.+?)>")


def _run(command: list[str], cwd: Path, log: Path, timeout: int = 180) -> int:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + "\n[STDERR]\n" + result.stderr, encoding="utf-8")
    return result.returncode


class BrflRuntime:
    def __init__(self, brfl_root: Path, work_root: Path, java_home: Path | None = None):
        self.brfl_root = brfl_root.resolve()
        self.work_root = work_root.resolve()
        java_home = java_home or Path(os.environ.get("JAVA_HOME", ""))
        if not java_home.is_dir():
            raise FileNotFoundError("A Java 8 JDK is required; pass --java-home")
        self.java = java_home / "bin" / "java.exe"
        self.javac = java_home / "bin" / "javac.exe"
        self.jar = java_home / "bin" / "jar.exe"
        self.lib = self.brfl_root / "lib"
        self.classes = self.work_root / "brfl-classes"
        self.agent = self.work_root / "brfl-agent.jar"
        self.bridge = Path(__file__).resolve().parent.parent / "java" / "ppfl" / "FpsplTraceExport.java"
        for path in (self.java, self.javac, self.jar, self.bridge, self.lib):
            if not path.exists():
                raise FileNotFoundError(path)

    def build(self) -> None:
        if self.agent.is_file() and (self.classes / "ppfl" / "FpsplTraceExport.class").is_file():
            return
        self.classes.mkdir(parents=True, exist_ok=True)
        sources = sorted((self.brfl_root / "src" / "main" / "java").rglob("*.java"))
        if not sources:
            raise FileNotFoundError("BRFL Java sources are missing")
        # Java 8's Windows argument length is limited; relative paths also avoid
        # encoding the non-ASCII workspace path in javac's response file.
        source_list = self.work_root / "brfl-sources.txt"
        source_list.write_text(
            "\n".join(path.relative_to(self.brfl_root).as_posix() for path in sources) + "\n",
            encoding="ascii",
        )
        log = self.work_root / "brfl-build.log"
        command = [str(self.javac), "-encoding", "UTF-8", "-cp", str(self.lib / "*"),
                   "-d", str(self.classes), "@" + str(source_list)]
        if _run(command, self.brfl_root, log, timeout=300):
            raise RuntimeError(f"BRFL compilation failed: {log}")
        if _run([str(self.javac), "-encoding", "UTF-8", "-cp",
                 os.pathsep.join((str(self.classes), str(self.lib / "*"))),
                 "-d", str(self.classes), str(self.bridge)], self.work_root, log, timeout=120):
            raise RuntimeError(f"BRFL graph exporter compilation failed: {log}")
        manifest = self.brfl_root / "src" / "main" / "resources" / "META-INF" / "MANIFEST.MF"
        if _run([str(self.jar), "cfm", str(self.agent), str(manifest),
                 "-C", str(self.classes), "."], self.work_root, log, timeout=120):
            raise RuntimeError(f"BRFL agent packaging failed: {log}")

    def recompile_variant(self, variant_dir: Path, variant_work: Path) -> tuple[Path, Path, list[str]]:
        main_out, test_out = variant_work / "build" / "main", variant_work / "build" / "test"
        main_sources = sorted((variant_dir / "src").rglob("*.java"))
        test_sources = sorted((variant_dir / "test").rglob("*.java"))
        if not main_sources or not test_sources:
            raise FileNotFoundError(f"Java source/test files missing: {variant_dir}")
        main_out.mkdir(parents=True, exist_ok=True)
        test_out.mkdir(parents=True, exist_ok=True)
        compile_log = variant_work / "compile.log"
        if not any(main_out.rglob("*.class")):
            command = [str(self.javac), "-g", "-encoding", "UTF-8", "-cp", str(self.lib / "*"),
                       "-d", str(main_out), *(str(path) for path in main_sources)]
            if _run(command, variant_work, compile_log):
                raise RuntimeError(f"Production source recompilation failed: {compile_log}")
        if not any(test_out.rglob("*.class")):
            cp = os.pathsep.join((str(main_out), str(self.lib / "*")))
            command = [str(self.javac), "-g", "-encoding", "UTF-8", "-cp", cp,
                       "-d", str(test_out), *(str(path) for path in test_sources)]
            if _run(command, variant_work, compile_log):
                raise RuntimeError(f"Test recompilation failed: {compile_log}")
        classes = []
        for path in main_sources:
            source = path.read_text(encoding="utf-8", errors="replace")
            package = JAVA_SOURCE.search(source)
            classes.append((package.group(1) + "." if package else "") + path.stem)
        return main_out, test_out, classes

    def trace_test(
        self, variant_dir: Path, variant_work: Path, coverage_file: Path,
        expected_outcome: str, built: tuple[Path, Path, list[str]],
    ) -> tuple[Path, dict]:
        test_name = coverage_file.name.removesuffix(".coverage.xml")
        if not re.fullmatch(r"[\w.]+_ESTest\.test\w+", test_name):
            raise ValueError(f"Unexpected test identity: {test_name}")
        test_class, method = test_name.rsplit(".", 1)
        sut_name = test_class.removesuffix("_ESTest")
        package, short_name = (sut_name.rsplit(".", 1) if "." in sut_name else ("", sut_name))
        oracle = f"{package}::{short_name}_{method}"
        main_out, test_out, main_classes = built
        stage = variant_work / "tests" / test_name
        stage.mkdir(parents=True, exist_ok=True)
        graph_file = stage / "graph.tsv"
        status_file = stage / "status.json"
        if graph_file.is_file() and status_file.is_file():
            import json
            return graph_file, json.loads(status_file.read_text(encoding="utf-8"))
        (stage / "infer.txt").write_text(",".join(["0.5"] * 10) + "\n", encoding="ascii")
        cp = os.pathsep.join((str(main_out), str(test_out), str(self.agent), str(self.lib / "*")))
        instrumented = ":".join(sorted(set(main_classes + [test_class])))
        raw_log = stage / "test-run.log"
        command = [str(self.java), "-noverify",
                   f"-javaagent:{self.agent}=instrumentingclass={instrumented},logfile={oracle.replace('::', '.')}.log,project={variant_dir.name}",
                   "-cp", cp, "org.junit.platform.console.ConsoleLauncher", "--details=none",
                   "--select-method", f"{test_class}#{method}"]
        result = subprocess.run(command, cwd=stage, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=120, check=False)
        raw_log.write_text(result.stdout + "\n[STDERR]\n" + result.stderr, encoding="utf-8")
        expected_rc = 1 if expected_outcome == "failed" else 0
        if result.returncode != expected_rc or f"###{test_class}::{method}" not in result.stdout:
            raise RuntimeError(f"Observed test outcome differs from coverage ({result.returncode}): {raw_log}")
        run_dir = stage / "trace" / "logs" / "run"
        source_dir = stage / "trace" / "logs" / "mytrace"
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_file = run_dir / f"{oracle.replace('::', '.')}.log"
        trace_file.write_text(
            "".join(line for line in result.stdout.splitlines(keepends=True) if line.startswith(TRACE_LINE)),
            encoding="utf-8",
        )
        if trace_file.stat().st_size == 0:
            raise RuntimeError(f"BRFL inserted no runtime events: {raw_log}")
        export_log = stage / "graph-build.log"
        export_cp = os.pathsep.join((str(self.classes), str(self.lib / "*")))
        status = "FAIL" if expected_outcome == "failed" else "PASS"
        export_command = [str(self.java), "-cp", export_cp, "ppfl.FpsplTraceExport",
                          str(source_dir), str(run_dir), oracle, status, str(graph_file)]
        if _run(export_command, stage, export_log, timeout=120):
            raise RuntimeError(f"BRFL graph export failed: {export_log}")
        match = ASSERT_FAILURE.search(result.stdout)
        observation = {"test": test_name, "expected_outcome": expected_outcome,
                       "observed_outcome": "failed" if result.returncode == 1 else "passed",
                       "failure_actual": match.group(2) if match else None,
                       "failure_expected": match.group(1) if match else None}
        import json
        status_file.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
        return graph_file, observation
