from __future__ import annotations

import re
from pathlib import Path

from .domain import JavaTestCase, TestPair


TOKEN = re.compile(r"[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||\S")
METHOD = re.compile(
    r"@Test(?:\s*\([^)]*\))?\s+(?:public\s+)?(?:final\s+)?void\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[^\{]+)?\{",
    re.MULTILINE,
)
PACKAGE = re.compile(r"\bpackage\s+([\w.]+)\s*;")
CALL = re.compile(r"\.\s*([A-Za-z_$][\w$]*)\s*\(")
ASSERTION = re.compile(r"\b(?:assert\w*|fail|verify\w*)\s*\([^;]*", re.IGNORECASE)


def _closing_brace(text: str, opening: int) -> int:
    depth = 0
    quote = None
    escaped = False
    line_comment = False
    block_comment = False
    i = opening
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
        elif block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                i += 1
        elif quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
        elif ch == "/" and nxt == "/":
            line_comment = True
            i += 1
        elif ch == "/" and nxt == "*":
            block_comment = True
            i += 1
        elif ch in {'"', "'"}:
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("测试方法的大括号不配对")


def extract_tests(source_path: Path) -> list[JavaTestCase]:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    package = PACKAGE.search(text)
    package_name = package.group(1) if package else ""
    class_name = source_path.stem
    full_class = f"{package_name}.{class_name}" if package_name else class_name
    tests = []
    for match in METHOD.finditer(text):
        opening = text.find("{", match.start())
        closing = _closing_brace(text, opening)
        tests.append(JavaTestCase(full_class, match.group(1), source_path, text[opening + 1:closing], text))
    return tests


def discover_tests(variant_dir: Path) -> list[JavaTestCase]:
    test_root = variant_dir / "test"
    return [case for path in sorted(test_root.rglob("*.java")) for case in extract_tests(path)]


def failed_test_identities(variant_dir: Path) -> set[str]:
    root = variant_dir / "coverage" / "failed"
    return {p.name.removesuffix(".coverage.xml") for p in root.glob("*.coverage.xml")}


def canonical_tokens(body: str) -> tuple[str, ...]:
    body = re.sub(r"//.*?$|/\*.*?\*/", " ", body, flags=re.MULTILINE | re.DOTALL)
    return tuple(token.lower() for token in TOKEN.findall(body))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a or b else 0.0


def oracle_signature(test: JavaTestCase) -> set[str]:
    return {" ".join(canonical_tokens(x)) for x in ASSERTION.findall(test.body)}


def test_similarity(left: JavaTestCase, right: JavaTestCase) -> tuple[float, float]:
    lt, rt = set(canonical_tokens(left.body)), set(canonical_tokens(right.body))
    body_score = _jaccard(lt, rt)
    call_score = _jaccard(set(CALL.findall(left.body)), set(CALL.findall(right.body)))
    oracle_score = _jaccard(oracle_signature(left), oracle_signature(right))
    identity = 1.0 if left.method_name == right.method_name else 0.0
    return 0.45 * body_score + 0.25 * call_score + 0.25 * oracle_score + 0.05 * identity, oracle_score


def pair_tests(
    failing: JavaTestCase, candidates: list[JavaTestCase], threshold: float = 0.55
) -> TestPair:
    scored = [(candidate, *test_similarity(failing, candidate)) for candidate in candidates]
    if not scored:
        return TestPair(failing, None, "missing", 0.0, 0.0)
    candidate, score, oracle = max(scored, key=lambda item: (item[1], item[2], item[0].identity))
    exact = canonical_tokens(failing.body) == canonical_tokens(candidate.body) and oracle == 1.0
    if exact:
        return TestPair(failing, candidate, "exact", score, oracle)
    if score >= threshold and oracle >= 0.5:
        return TestPair(failing, candidate, "similar", score, oracle)
    return TestPair(failing, None, "missing", score, oracle, rationale="没有通过内容与判定条件双重门槛")
