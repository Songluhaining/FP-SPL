"""用仓库源码编译插桩代理，在隔离目录重编译并执行各个测试。"""
import hashlib
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = str(ROOT / "lib" / "*")


def command(args, folder, timeout=120):
    result = subprocess.run([str(s) for s in args], cwd=folder, capture_output=True,
                            timeout=timeout, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError((result.stderr + result.stdout)[-2000:])
    return result.stdout


class Runtime:
    def __init__(self, home, cache):
        self.java = home / "bin/java.exe"
        self.javac = home / "bin/javac.exe"
        self.jar = home / "bin/jar.exe"
        self.cache = cache.resolve()
        self.classes = self.cache / "classes"
        self.agent = self.cache / "agent.jar"

    def build(self):
        sources = sorted((ROOT / "src/main/java").rglob("*.java"))
        bridge = ROOT / "fp_research/java/CounterfactualInferenceExport.java"
        sources.append(bridge)
        digest = hashlib.sha256()
        for path in sources:
            digest.update(path.read_bytes())
        manifest = ROOT / "src/main/resources/META-INF/MANIFEST.MF"
        digest.update(manifest.read_bytes())
        stamp = self.cache / "source.sha256"
        if self.agent.exists() and stamp.exists() and stamp.read_text() == digest.hexdigest():
            return
        self.classes.mkdir(parents=True, exist_ok=True)
        response = self.cache / "sources.txt"
        response.write_text("\n".join(p.relative_to(ROOT).as_posix() for p in sources), encoding="ascii")
        command([self.javac, "-encoding", "UTF-8", "-cp", LIB, "-d", self.classes, "@" + str(response)], ROOT)
        command([self.jar, "cfm", self.agent, manifest, "-C", self.classes, "."], self.cache)
        stamp.write_text(digest.hexdigest(), encoding="ascii")

    def compile(self, variant, out):
        main, tests = out / "main", out / "tests"
        main.mkdir(parents=True, exist_ok=True)
        tests.mkdir(parents=True, exist_ok=True)
        sources = sorted((variant / "src").rglob("*.java"))
        test_sources = sorted((variant / "test").rglob("*.java"))
        command([self.javac, "-g", "-encoding", "UTF-8", "-cp", LIB, "-d", main, *sources], out)
        command([self.javac, "-g", "-encoding", "UTF-8", "-cp", os.pathsep.join([str(main), LIB]),
                 "-d", tests, *test_sources], out)
        names = []
        for p in sources:
            package = re.search(r"\bpackage\s+([\w.]+)\s*;", p.read_text(encoding="utf-8", errors="replace"))
            names.append((package[1] + "." if package else "") + p.stem)
        return main, tests, names

    def trace(self, build, name, stage, expected):
        stage.mkdir(parents=True, exist_ok=True)
        main, tests, names = build
        cls, method = name.rsplit(".", 1)
        cp = os.pathsep.join([str(main), str(tests), str(self.agent), LIB])
        options = "instrumentingclass=" + ":".join(sorted(set(names + [cls]))) + ",logfile=" + name + ".log,project=fp"
        args = [self.java, "-noverify", "-javaagent:" + str(self.agent) + "=" + options, "-cp", cp,
                "org.junit.platform.console.ConsoleLauncher", "--details=summary", "--disable-ansi-colors",
                "--select-method", cls + "#" + method]
        proc = subprocess.run(args, cwd=stage, capture_output=True, timeout=90, encoding="utf8", errors="replace")
        log = proc.stdout + proc.stderr
        if (not re.search(r"\b1 tests? started\s*\]", log)
                or proc.returncode != int(expected == "failed") or f"###{cls}::{method}" not in log):
            raise RuntimeError("运行结果与单测记录不符 " + log[-1600:])
        trace = stage / "run.log"
        trace.write_text("\n".join(s for s in proc.stdout.splitlines() if s.startswith(("###","opcode"))) + "\n", encoding="utf8")
        return stage / "trace/logs/mytrace", trace

    def infer(self, folder, manifest):
        (folder / "infer.txt").write_text(",".join(["0.5"] * 10), encoding="ascii")
        path = folder / "executions.tsv"
        path.write_text("\n".join("\t".join(map(str,row)) for row in manifest), encoding="utf8")
        graph = folder / "posterior.tsv"
        command([self.java, "-Xmx768m", "-cp", os.pathsep.join([str(self.classes), LIB]),
                 "ppfl.CounterfactualInferenceExport", "--joint", path, graph], folder, timeout=300)
        return graph
