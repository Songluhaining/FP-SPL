import json
from pathlib import Path

from cfp_spl.domain import Configuration, JavaTestCase
from cfp_spl.generator import CounterfactualTestGenerator, DeepSeekClient, _read_api_key


def _configuration(root: Path, name: str, source: str) -> Configuration:
    variant = root / name
    (variant / "src").mkdir(parents=True)
    (variant / "src" / "Account.java").write_text(source, encoding="utf-8")
    return Configuration(name, (1,), ("Base",), "passed" if name == "A" else "failed", variant)


def test_api_key_is_read_without_being_copied(tmp_path: Path):
    key_file = tmp_path / "secret.txt"
    key_file.write_text("  sk-test-secret\n", encoding="utf-8")
    assert _read_api_key(key_file) == "sk-test-secret"


def test_deepseek_request_uses_json_mode(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            answer = {"class_name": "p.T", "method_name": "test0",
                      "java_source": "package p; public class T {}", "rationale": "same"}
            envelope = {"id": "id-1", "model": "deepseek-flash", "usage": {"total_tokens": 7},
                        "choices": [{"message": {"content": json.dumps(answer)}}]}
            return json.dumps(envelope).encode()

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    answer, metadata = DeepSeekClient("sk-hidden", attempts=1).complete({"input": "x"})
    assert answer["class_name"] == "p.T"
    assert captured["authorization"] == "Bearer sk-hidden"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert metadata["usage"]["total_tokens"] == 7


def test_generator_includes_both_products_and_validates_generated_test(tmp_path: Path, monkeypatch):
    a = _configuration(tmp_path, "A", "package main; public class Account { public int value(){return 1;} }")
    b = _configuration(tmp_path, "B", "package main; public class Account { public int value(){return 0;} }")
    failed = JavaTestCase(
        "main.Account_ESTest", "test0", tmp_path / "Account_ESTest.java",
        "assertEquals(1, new Account().value());",
        "package main; import org.junit.Test; import static org.junit.Assert.*; "
        "public class Account_ESTest { @Test public void test0(){ assertEquals(1, new Account().value()); } }",
    )

    def fake_complete(self, payload):
        identity = payload["required_output_identity"]
        package, _, simple = identity["class_name"].rpartition(".")
        source = (f"package {package}; import org.junit.Test; import static org.junit.Assert.*; "
                  f"public class {simple} {{ @Test public void {identity['method_name']}() {{ "
                  "assertEquals(1, new Account().value()); } }")
        return {**identity, "java_source": source, "rationale": "保持相同输入与断言"}, {"provider": "deepseek"}

    monkeypatch.setattr(DeepSeekClient, "complete", fake_complete)
    key = tmp_path / "key.txt"
    key.write_text("sk-test", encoding="utf-8")
    pair = CounterfactualTestGenerator(None, deepseek_key_file=key).generate(
        a, b, failed, tmp_path / "requests"
    )
    assert pair.provenance == "generated"
    assert pair.passing is not None
    request = json.loads(pair.generation_request.read_text(encoding="utf-8"))
    assert "return 1" in request["configuration_A_product_sources"][0]["source"]
    assert "return 0" in request["configuration_B_product_sources"][0]["source"]
    assert "sk-test" not in pair.generation_request.read_text(encoding="utf-8")


def test_model_selects_existing_counterfactual_test(tmp_path: Path, monkeypatch):
    a = _configuration(tmp_path, "A", "package main; public class Account {}")
    b = _configuration(tmp_path, "B", "package main; public class Account {}")
    failed = JavaTestCase(
        "main.FailedTest", "testDebit", tmp_path / "FailedTest.java",
        "assertEquals(5, account.debit(3));", "",
    )
    unrelated = JavaTestCase(
        "main.ATest", "testName", tmp_path / "ATest.java",
        'assertEquals("x", account.name());', "",
    )
    equivalent = JavaTestCase(
        "main.ATest", "testWithdraw", tmp_path / "ATest.java",
        "assertEquals(5, account.debit(3));", "",
    )

    def fake_select(self, payload):
        identities = {item["identity"] for item in payload["candidates_in_A"]}
        assert equivalent.identity in identities
        return {
            "selected_identity": equivalent.identity,
            "confidence": 0.92,
            "rationale": "相同输入、调用和断言",
        }, {"provider": "deepseek"}

    monkeypatch.setattr(DeepSeekClient, "select", fake_select)
    key = tmp_path / "key.txt"
    key.write_text("sk-test", encoding="utf-8")
    pair = CounterfactualTestGenerator(None, deepseek_key_file=key).select_existing(
        a, b, failed, [unrelated, equivalent], tmp_path / "requests"
    )
    assert pair.provenance == "model_selected"
    assert pair.passing == equivalent
    assert "0.920" in pair.rationale
