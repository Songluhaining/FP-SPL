from pathlib import Path
from types import SimpleNamespace

from cfp_spl.pipeline import CounterfactualPipeline


def test_runtime_exception_is_not_confirmed_failure(tmp_path):
    class Runtime:
        def compile_configuration(self, *args):
            return None

        def execute(self, config, test, work, compiled, require_graph):
            log = tmp_path / "execution.log"
            log.write_text("java.lang.NullPointerException", encoding="utf-8")
            return SimpleNamespace(outcome="failed", log_path=log)

    pipeline = object.__new__(CounterfactualPipeline)
    pipeline.runtime = Runtime()
    test = SimpleNamespace(full_source="source", source_path=Path("T.java"),
                           class_name="T", method_name="test0")
    result = pipeline._validate_generated(SimpleNamespace(name="A"), SimpleNamespace(name="B"),
                                          test, tmp_path)
    assert not result["valid"]
    assert "非断言异常" in result["error"]
