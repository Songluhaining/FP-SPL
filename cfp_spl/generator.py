from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from collections import Counter
from typing import Callable

from .configurations import feature_delta
from .domain import Configuration, JavaTestCase, TestPair
from .java_tests import ASSERTION, METHOD, _closing_brace, canonical_tokens, extract_tests, oracle_signature, test_similarity


def _test_context(test: JavaTestCase, keep_test: bool = True) -> str:
    """保留当前测试、导入、字段和辅助初始化，删除同类中无关的其他测试方法。

    自动生成测试类可能有数百方法，整类发送会淹没实际故障上下文。
    """
    text = test.full_source
    for match in reversed(list(METHOD.finditer(text))):
        if not keep_test or match.group(1) != test.method_name:
            end = _closing_brace(text, text.find("{", match.start())) + 1
            text = text[:match.start()] + text[end:]
    return text


def _request_digest(payload: dict, model: str, endpoint: str) -> str:
    """输入、模型、服务或生成协议变化时，不复用旧模型响应。"""
    raw = json.dumps(["joint-counterfactual-v3", model, endpoint, payload],
                     ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


CLASS_NAME = re.compile(r"^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*$")
JAVA_CLASS = re.compile(r"\bpublic\s+class\s+([A-Za-z_$][\w$]*)\b")
JAVA_PACKAGE = re.compile(r"\bpackage\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*;")


def _default_key_file() -> Path | None:
    """优先使用环境变量；本项目布局下也自动发现仓库同级的密钥文件。"""
    configured = os.environ.get("DEEPSEEK_API_KEY_FILE")
    if configured:
        return Path(configured).expanduser()
    sibling = Path(__file__).resolve().parents[2] / "deepseek-api-key.txt"
    return sibling if sibling.is_file() else None


def _read_api_key(path: Path | None) -> str | None:
    value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value
    if path is None:
        path = _default_key_file()
    if path is None or not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"模型接口密钥文件为空：{path}")
    return value


def _source_context(config: Configuration, limit: int = 160_000) -> list[dict[str, str]]:
    """收集产品接口实现，给模型足够信息判断A、B共同可调用的接口。"""
    result: list[dict[str, str]] = []
    consumed = 0
    root = config.variant_dir / "src"
    for path in sorted(root.rglob("*.java")):
        source = path.read_text(encoding="utf-8", errors="replace")
        remaining = limit - consumed
        if remaining <= 0:
            break
        clipped = source[:remaining]
        item = {"path": path.relative_to(root).as_posix(), "source": clipped}
        if len(clipped) < len(source):
            item["truncated"] = "true"
        result.append(item)
        consumed += len(clipped)
        if len(clipped) < len(source):
            break
    return result


def _answer_from_content(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, count=1, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content, count=1)
    answer = json.loads(content)
    if not isinstance(answer, dict):
        raise ValueError("模型响应的顶层必须是JSON对象")
    return answer


class DeepSeekClient:
    """使用标准库调用DeepSeek的对话生成接口，避免额外运行依赖。"""

    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com", timeout: int = 300,
                 attempts: int = 3):
        self.api_key = api_key
        self.model = model
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.attempts = attempts

    def _request(self, payload: dict[str, Any], system: str, instruction: str,
                 max_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction + "\n" +
                 json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "stream": False,
        }
        encoded = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            request = urllib.request.Request(
                self.endpoint, data=encoded, method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
                if envelope["choices"][0].get("finish_reason") in {"length", "content_filter"}:
                    raise ValueError("模型输出被截断或过滤，不能作为完整测试")
                content = envelope["choices"][0]["message"]["content"]
                if not content:
                    raise ValueError("模型返回了空内容")
                answer = _answer_from_content(content)
                metadata = {
                    "provider": "deepseek", "model": envelope.get("model", self.model),
                    "response_id": envelope.get("id"), "usage": envelope.get("usage", {}),
                    "attempt": attempt,
                }
                return answer, metadata
            except urllib.error.HTTPError as exc:
                last_error = RuntimeError(f"模型接口返回HTTP {exc.code}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                    KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
            if attempt < self.attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"模型接口调用失败：{last_error}") from last_error

    def complete(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        system = (
            "你是Java测试工程师。请生成一个反事实JUnit测试，使它在两个软件产品配置中都能编译和执行，"
            "并保持给定失败测试的输入场景、可观察行为和判定条件。不得删除、放宽或反转判定条件；"
            "不得伪造产品状态；不得使用只存在于一侧的产品接口。只输出一个JSON对象，不要输出代码围栏或解释。"
            "可实现时JSON必须包含class_name、method_name、java_source和rationale四个字符串字段。"
            "若A缺少对应功能、无法保持原故障判定目标，请返回not_applicable=true及rationale，禁止伪造测试。"
        )
        return self._request(payload, system,
                             "请根据以下JSON输入生成测试，并以JSON格式返回：", 12_000)

    def select(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        system = (
            "你是Java测试工程师。请判断配置A中的哪个已有JUnit测试与配置B中的失败测试具有反事实可比性。"
            "可比测试必须保持可比较的输入场景、相同产品行为和判定目标；允许小幅输入差异。"
            "必须检查被调用的产品方法语义。isReadable和isEncrypted等不同属性即使断言文本相同也不等价。"
            "必须考虑完整测试类中的初始化、辅助方法及异常预期，不能仅看断言文本。"
            "如果候选均不等价，selected_identity必须为空字符串。只输出JSON对象，包含"
            "selected_identity、confidence和rationale三个字段。confidence取0到1。"
        )
        return self._request(payload, system,
                             "请从以下候选中选择反事实测试，并以JSON格式返回：", 2_000)

    def review(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._request(payload,
            "你是独立的测试审查员。审查生成测试是否保留原失败测试的关键输入场景、被测行为和故障判定目标。"
            "以original_failure_execution中的实际失败断言和调用为审查中心，允许省去无关特征的其他检查。"
            "A也失败可能正是待识别的假通过，不能仅因A/B都会失败而拒绝有效测试。"
            "原故障若在初始化中崩溃，保留同一初始化调用即可保留故障目标，原测试尚未执行的后续断言可省去。"
            "允许适配特征接口差异，但不能绕开失败目标、伪造状态、削弱断言或强行制造异常。"
            "只输出JSON，包含scenario_comparable、behavior_preserved、oracle_preserved、"
            "no_shortcut四个布尔值以及rationale理由。不确定时返回false。",
            "请逐项审查以下原测试、两侧产品和生成测试：", 3000)


class CounterfactualTestGenerator:
    """生成跨配置反事实测试；外部命令优先，否则直接调用DeepSeek。"""

    def __init__(self, command: str | None, timeout: int = 300,
                 deepseek_key_file: Path | None = None,
                 deepseek_model: str = "deepseek-chat",
                 deepseek_base_url: str = "https://api.deepseek.com"):
        self.command = command
        self.timeout = timeout
        self.deepseek_key_file = deepseek_key_file
        self.deepseek_model = deepseek_model
        self.deepseek_base_url = deepseek_base_url
        self.failure_contexts: dict[tuple[str, str], dict] = {}

    def failure_context(self, config: Configuration, test: JavaTestCase) -> dict:
        return self.failure_contexts.get((str(config.variant_dir), test.identity), {})

    def select_existing(self, passing: Configuration, failing: Configuration,
                        failed_test: JavaTestCase, candidates: list[JavaTestCase],
                        request_dir: Path, candidate_limit: int = 12) -> TestPair:
        if not candidates:
            return TestPair(failed_test, None, "missing", 0.0, 0.0,
                            rationale="配置A没有已有测试")
        ranked = sorted(
            ((candidate, *test_similarity(failed_test, candidate)) for candidate in candidates),
            key=lambda item: (-item[1], -item[2], item[0].identity),
        )[:candidate_limit]
        request_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{failed_test.class_name.replace('.', '_')}.{failed_test.method_name}.selection"
        request = request_dir / f"{stem}.request.json"
        response = request_dir / f"{stem}.response.json"
        payload = {
            "task": "从配置A已有测试中选择与配置B失败测试具有反事实可比性的测试",
            "passing_configuration_A": passing.name,
            "failing_configuration_B": failing.name,
            "feature_delta": feature_delta(passing, failing),
            "original_failure_execution": self.failure_context(failing, failed_test),
            "failed_test_in_B": {
                "identity": failed_test.identity,
                "body": failed_test.body,
                "source": _test_context(failed_test),
                "oracle_signature": sorted(oracle_signature(failed_test)),
            },
            "candidates_in_A": [{
                "identity": candidate.identity,
                "body": candidate.body,
                "oracle_signature": sorted(oracle_signature(candidate)),
                "lexical_similarity": similarity,
                "oracle_similarity": oracle,
            } for candidate, similarity, oracle in ranked],
            "candidate_class_sources": {candidate.class_name: _test_context(candidate, keep_test=False)
                                         for candidate, _, _ in ranked},
            "response_schema": {
                "selected_identity": "候选identity；没有合格候选时为空字符串",
                "confidence": "0到1之间的数值",
                "rationale": "输入场景、行为与判定目标是否一致的理由",
            },
        }
        digest = _request_digest(payload, self.deepseek_model, self.deepseek_base_url)
        response = request_dir / f"{stem}.{digest[:16]}.response.json"
        request.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            if response.is_file():
                cached = json.loads(response.read_text(encoding="utf-8"))
                answer = {key: value for key, value in cached.items() if key != "_metadata"}
                metadata = cached.get("_metadata", {"provider": "cached"})
            else:
                api_key = _read_api_key(self.deepseek_key_file)
                if not api_key:
                    return TestPair(failed_test, None, "missing", 0.0, 0.0, request,
                                    "未找到DeepSeek接口密钥")
                answer, metadata = DeepSeekClient(
                    api_key, self.deepseek_model, self.deepseek_base_url, self.timeout
                ).select(payload)
            identity = answer.get("selected_identity", "")
            confidence = float(answer.get("confidence", 0.0))
            rationale = str(answer.get("rationale", ""))
            available = {candidate.identity: (candidate, similarity, oracle)
                         for candidate, similarity, oracle in ranked}
            if identity and identity not in available:
                raise ValueError("模型选择了候选列表之外的测试")
            response.write_text(json.dumps({**answer, "_metadata": metadata},
                                           ensure_ascii=False, indent=2), encoding="utf-8")
            if not identity or confidence < 0.60:
                return TestPair(failed_test, None, "missing", 0.0, 0.0, request,
                                rationale or "模型未找到可靠的已有反事实测试")
            selected, similarity, oracle = available[identity]
            return TestPair(failed_test, selected, "model_selected", similarity, oracle,
                            request, f"模型置信度={confidence:.3f}；{rationale}")
        except Exception as exc:
            (request_dir / f"{stem}.error.txt").write_text(str(exc), encoding="utf-8")
            return TestPair(failed_test, None, "missing", 0.0, 0.0, request,
                            f"已有测试选择失败：{exc}")

    def _payload(self, passing: Configuration, failing: Configuration,
                 failed_test: JavaTestCase) -> dict[str, Any]:
        digest = hashlib.sha1(
            f"{passing.name}|{failing.name}|{failed_test.identity}".encode("utf-8")
        ).hexdigest()[:10]
        package = failed_test.class_name.rpartition(".")[0]
        simple_class = f"GeneratedCounterfactual_{digest}"
        class_name = f"{package}.{simple_class}" if package else simple_class
        method_name = f"testCounterfactual_{digest}"
        return {
            "task": "生成可同时在配置A和配置B中编译执行的JUnit反事实测试，保持原失败测试的行为和判定条件",
            "passing_configuration_A": passing.name,
            "failing_configuration_B": failing.name,
            "feature_delta": feature_delta(passing, failing),
            "original_failure_execution": self.failure_context(failing, failed_test),
            "required_output_identity": {"class_name": class_name, "method_name": method_name},
            "failed_test_in_B": {
                "identity": failed_test.identity,
                "source": _test_context(failed_test),
                "body": failed_test.body,
                "oracle_signature": sorted(oracle_signature(failed_test)),
            },
            "configuration_A_product_sources": _source_context(passing),
            "configuration_B_product_sources": _source_context(failing),
            "constraints": [
                "输出完整、可独立编译的JUnit测试类，并严格使用required_output_identity指定的类名和方法名",
                "沿用原测试使用的JUnit版本和断言库",
                "保留原失败测试的输入场景、可观察行为和判定条件，不得降低断言强度",
                "只使用配置A和配置B共同存在且签名兼容的产品接口",
                "兼容Java 8，不使用外部新增依赖，不访问网络、文件系统、时间或随机源",
                "测试必须具有确定性，不得捕获并忽略异常，不得用条件分支跳过断言",
                "优先最小修改原测试的必要接口与初始化；禁止凭方法名猜测API，必须以两侧源码为依据",
                "优先围绕original_failure_execution标明的实际失败断言及调用构造最小可比场景；可省去与该故障判定无关的其他特征检查，但不能换成另一属性或功能",
                "若原故障是产品异常而非断言失败，保留触发该异常的产品调用与输入即可；不必保留根本未执行的无关断言，正常返回即为通过，异常逃逸即为失败",
            ],
            "response_schema": {
                "class_name": class_name,
                "method_name": method_name,
                "java_source": "完整Java源代码字符串",
                "rationale": "说明输入场景、调用链和判定条件如何对应原失败测试",
            },
        }

    @staticmethod
    def _validate_answer(answer: dict[str, Any], payload: dict[str, Any]) -> None:
        required = ("class_name", "method_name", "java_source", "rationale")
        if any(not isinstance(answer.get(name), str) or not answer[name].strip() for name in required):
            raise ValueError(f"模型响应必须包含非空字符串字段：{required}")
        expected = payload["required_output_identity"]
        if answer["class_name"] != expected["class_name"] or answer["method_name"] != expected["method_name"]:
            raise ValueError("模型没有使用指定的唯一测试类名和方法名")
        if not CLASS_NAME.fullmatch(answer["class_name"]):
            raise ValueError("模型返回的Java类名不合法")
        package, _, simple = answer["class_name"].rpartition(".")
        source_package = JAVA_PACKAGE.search(answer["java_source"])
        actual_package = source_package.group(1) if source_package else ""
        source_class = JAVA_CLASS.search(answer["java_source"])
        if actual_package != package or source_class is None or source_class.group(1) != simple:
            raise ValueError("生成源码中的包名或公开类名与响应字段不一致")

    def _external(self, request: Path, response: Path, request_dir: Path) -> dict[str, Any]:
        command = [part.format(request=str(request), response=str(response))
                   for part in shlex.split(self.command or "")]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=self.timeout, check=False)
        (request_dir / f"{request.stem}.generator.log").write_text(
            completed.stdout + "\n[STDERR]\n" + completed.stderr, encoding="utf-8"
        )
        if completed.returncode or not response.is_file():
            raise RuntimeError("外部测试生成命令失败")
        return json.loads(response.read_text(encoding="utf-8"))

    def generate(self, passing: Configuration, failing: Configuration,
                 failed_test: JavaTestCase, request_dir: Path,
                 validator: Callable[[JavaTestCase], dict] | None = None,
                 max_rounds: int = 3) -> TestPair:
        """最多三轮修复；validator返回真实编译/执行结果，有效才结束。

        无validator时仅完成源码结构校验，不能据此认定生成测试有效。
        每轮请求/响应/验证均保留；非机械复制必须复核场景、行为及判定目标。
        """
        request_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{failed_test.class_name.replace('.', '_')}.{failed_test.method_name}"
        request = request_dir / f"{stem}.request.json"
        payload = self._payload(passing, failing, failed_test)
        request.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        feedback = None
        try:
            for generation_round in range(1, max_rounds + 1):
                generation_payload = dict(payload)
                if feedback:
                    generation_payload["repair_feedback"] = feedback
                digest = _request_digest(generation_payload, self.deepseek_model,
                                         self.deepseek_base_url + str(self.command))
                prefix = f"{stem}.round-{generation_round}.{digest[:16]}"
                round_request = request_dir / f"{prefix}.request.json"
                response = request_dir / f"{prefix}.response.json"
                round_request.write_text(json.dumps(generation_payload, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
                if response.is_file():
                    cached = json.loads(response.read_text(encoding="utf-8"))
                    answer = {key: value for key, value in cached.items() if key != "_metadata"}
                    metadata = cached.get("_metadata", {})
                elif self.command:
                    answer = self._external(round_request, response, request_dir)
                    metadata = {"provider": "external-command"}
                else:
                    api_key = _read_api_key(self.deepseek_key_file)
                    if not api_key:
                        raise ValueError("未找到DeepSeek接口密钥，也未配置外部测试生成命令")
                    answer, metadata = DeepSeekClient(
                        api_key, self.deepseek_model, self.deepseek_base_url, self.timeout
                    ).complete(generation_payload)
                response.write_text(json.dumps({**answer, "_metadata": metadata},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
                if answer.get("not_applicable") is True:
                    return TestPair(failed_test, None, "missing", 0.0, 0.0, round_request,
                                    "模型报告无法构造可比测试（不是阴性证明）：" + str(answer.get("rationale", "")))
                try:
                    self._validate_answer(answer, payload)
                    source_path = (request_dir / "generated").joinpath(
                        *answer["class_name"].split(".")).with_suffix(".java")
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                    source_path.write_text(answer["java_source"], encoding="utf-8")
                    cases = extract_tests(source_path)
                    if len(cases) != 1 or cases[0].method_name != answer["method_name"]:
                        raise ValueError("生成结果必须且只能包含指定测试方法")
                    generated = cases[0]
                    # 先拒绝空/恒真断言，再由独立模型审查语义而非断言字符串重合。
                    original = Counter(" ".join(canonical_tokens(x))
                                       for x in ASSERTION.findall(failed_test.body))
                    actual = Counter(" ".join(canonical_tokens(x))
                                     for x in ASSERTION.findall(generated.body))
                    signature = payload.get("original_failure_execution", {}).get("failure_signature", {})
                    crash_oracle = bool(signature.get("exception") and not signature.get("assertion"))
                    if ((not original or not actual) and not crash_oracle) or re.search(r"assertTrue\s*\(\s*true\s*\)|assertFalse\s*\(\s*false\s*\)", generated.body):
                        raise ValueError("必须原样保留所有断言的判定目标；不得输出空断言或恒真断言")
                    if re.search(r"\b(?:Assume|assume\w*)\s*\(", generated.body):
                        raise ValueError("不得跳过测试")
                    # 非机械复制由独立模型审查场景/行为/判定目标，不能只看文本断言重合。
                    if canonical_tokens(generated.body) != canonical_tokens(failed_test.body):
                        review_payload = {**payload, "generated_java_source": answer["java_source"]}
                        review_digest = _request_digest(review_payload, self.deepseek_model,
                                                        self.deepseek_base_url)
                        review_file = request_dir / f"{stem}.{review_digest[:16]}.review.json"
                        if review_file.is_file():
                            review = json.loads(review_file.read_text(encoding="utf-8"))
                        else:
                            review_key = _read_api_key(self.deepseek_key_file)
                            if not review_key:
                                raise ValueError("非机械复制测试需要模型语义复核；未提供接口密钥")
                            review, review_metadata = DeepSeekClient(
                                review_key, self.deepseek_model, self.deepseek_base_url, self.timeout
                            ).review(review_payload)
                            review["_metadata"] = review_metadata
                            review_file.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
                        required = ("scenario_comparable", "behavior_preserved", "oracle_preserved", "no_shortcut")
                        if not all(review.get(key) is True for key in required):
                            raise ValueError(f"语义复核未通过：{review.get('rationale', '')}")
                    validation = validator(generated) if validator else {
                        "valid": True, "scope": "structure-only-not-execution-validated"}
                except Exception as exc:
                    validation = {"valid": False, "error": str(exc)}
                (request_dir / f"{prefix}.validation.json").write_text(
                    json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
                if validation.get("valid"):
                    similarity, oracle = test_similarity(failed_test, generated)
                    return TestPair(failed_test, generated, "generated", similarity, oracle,
                                    round_request, answer["rationale"])
                feedback = {"previous_java_source": answer.get("java_source", ""),
                            "validation": validation,
                            "instruction": "依据诊断修复接口和初始化，保留原输入场景及全部断言；禁止强行制造失败。"}
            raise ValueError(f"{max_rounds}轮生成均未通过验证：{validation}")
        except Exception as exc:
            (request_dir / f"{stem}.error.txt").write_text(str(exc), encoding="utf-8")
            return TestPair(failed_test, None, "missing", 0.0, 0.0, request,
                            f"反事实测试生成失败：{exc}")
