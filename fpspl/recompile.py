"""Isolated adaptation of BRFL's recompile_failed_products.py.

BRFL recompiles src/test with javac -g and a JUnit/Clover/EvoSuite classpath.
Here both failed and passing variants are rebuilt under FP-SPL's output tree,
so the dataset and the original BRFL checkout are never modified.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable


JAVA_SOURCE = re.compile(r"\bpackage\s+([\w.]+)\s*;")


def recompile_variant(variant_dir: Path, variant_work: Path, javac: Path, lib: Path,
                      run: Callable) -> tuple[Path, Path, list[str]]:
    main_out, test_out = variant_work / "build" / "main", variant_work / "build" / "test"
    main_sources = sorted((variant_dir / "src").rglob("*.java"))
    test_sources = sorted((variant_dir / "test").rglob("*.java"))
    if not main_sources or not test_sources:
        raise FileNotFoundError(f"Java source/test files missing: {variant_dir}")
    main_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)
    compile_log = variant_work / "compile.log"
    # As in BRFL, -g preserves bytecode-to-source line mappings needed by
    # the runtime dependency graph. Compilation stays outside the dataset.
    if not any(main_out.rglob("*.class")):
        command = [str(javac), "-g", "-encoding", "UTF-8", "-cp", str(lib / "*"),
                   "-d", str(main_out), *(str(path) for path in main_sources)]
        if run(command, variant_work, compile_log):
            raise RuntimeError(f"Production source recompilation failed: {compile_log}")
    if not any(test_out.rglob("*.class")):
        cp = os.pathsep.join((str(main_out), str(lib / "*")))
        command = [str(javac), "-g", "-encoding", "UTF-8", "-cp", cp,
                   "-d", str(test_out), *(str(path) for path in test_sources)]
        if run(command, variant_work, compile_log):
            raise RuntimeError(f"Test recompilation failed: {compile_log}")
    classes = []
    for path in main_sources:
        source = path.read_text(encoding="utf-8", errors="replace")
        package = JAVA_SOURCE.search(source)
        classes.append((package.group(1) + "." if package else "") + path.stem)
    return main_out, test_out, classes
