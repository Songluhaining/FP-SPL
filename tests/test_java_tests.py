from pathlib import Path

from cfp_spl.java_tests import extract_tests, pair_tests


def write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_nested_braces_and_exact_counterfactual_pair(tmp_path):
    source = """package p; import org.junit.Test; class T_ESTest {
    @Test public void test0() throws Exception { if (true) { call(); } assertEquals(1, value()); }
    }"""
    left = extract_tests(write(tmp_path, "T_ESTest.java", source))[0]
    right = extract_tests(write(tmp_path, "U_ESTest.java", source.replace("T_ESTest", "U_ESTest")))[0]
    pair = pair_tests(left, [right])
    assert pair.provenance == "exact"
    assert pair.oracle_similarity == 1.0
