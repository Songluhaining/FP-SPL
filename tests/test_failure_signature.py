from types import SimpleNamespace

from cfp_spl.brfl_runtime import failure_signature
from cfp_spl.pipeline import CounterfactualPipeline


def test_signature_uses_junit_failure_report():
    result = failure_signature("opcode=\n => java.lang.ArrayIndexOutOfBoundsException: -1\n"
                               "       EmailSystem.Client.resetClients(Client.java:201)\n")
    assert result == {"exception": "java.lang.ArrayIndexOutOfBoundsException",
                      "product_method": "EmailSystem.Client.resetClients", "line": 201, "assertion": False}
    assert not failure_signature("opcode=AssertionError")


def test_original_product_crash_is_valid_even_before_assertions(tmp_path):
    graph = tmp_path / "structure.tsv"
    graph.write_text("#BRFL_FACTOR_GRAPH\t2\n", encoding="utf-8")
    log = tmp_path / "execution.log"
    log.write_text("=> java.lang.ArrayIndexOutOfBoundsException: -1\n"
                   "       p.C.reset(C.java:19)\n", encoding="utf-8")
    original = {"exception": "java.lang.ArrayIndexOutOfBoundsException",
                "product_method": "p.C.reset", "line": 30, "assertion": False}
    class Runtime:
        def compile_configuration(self, *args):
            return None
        def execute(self, *args, **kwargs):
            return SimpleNamespace(outcome="failed", log_path=log, graph_path=graph)
    pipeline = object.__new__(CounterfactualPipeline)
    pipeline.runtime = Runtime()
    pipeline.generator = SimpleNamespace(failure_context=lambda *args: {"failure_signature": original})
    test = SimpleNamespace(full_source="source", source_path=tmp_path / "T.java",
                           class_name="T", method_name="test0")
    result = pipeline._validate_generated(SimpleNamespace(name="A"), SimpleNamespace(name="B"),
                                          test, tmp_path, original_test=test)
    assert result["valid"]
    assert result["executions"]["A"]["matches_original_exception"]
    assert "A" in pipeline._direct_generated

    # 同类型异常发生在不同产品方法时，也不能冒充原故障。
    log.write_text("=> java.lang.ArrayIndexOutOfBoundsException: -1\n"
                   "       p.C.unrelated(C.java:19)\n", encoding="utf-8")
    result = pipeline._validate_generated(SimpleNamespace(name="A"), SimpleNamespace(name="B"),
                                          test, tmp_path, original_test=test)
    assert not result["valid"]
