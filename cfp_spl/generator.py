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

from .configurations import feature_delta
from .domain import Configuration, JavaTestCase, TestPair
from .java_tests import extract_tests, oracle_signature, test_similarity


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

    def __init__(self, api_key: str, model: str = "deepseek-flash",
                 base_url: str = "https://api.deepseek.com", timeout: int = 300,
                 attempts: int = 3):
        self.api_key = api_key
        self.model = model
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.attempts = attempts

    def complete(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        system = (
            "你是Java测试工程师。请生成一个反事实JUnit测试，使它在两个软件产品配置中都能编译和执行，"
            "并保持给定失败测试的输入场景、可观察行为和判定条件。不得删除、放宽或反转判定条件；"
            "不得伪造产品状态；不得使用只存在于一侧的产品接口。只输出一个JSON对象，不要输出代码围栏或解释。"
            "JSON必须包含class_name、method_name、java_source和rationale四个字符串字段。"
        )
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "请根据以下JSON输入生成测试，并以JSON格式返回：\n" +
                 json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": 12_000,
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
                # 不保存响应正文，避免服务端回显请求源码或认证信息。
                last_error = RuntimeError(f"模型接口返回HTTP {exc.code}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                    KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
            if attempt < self.attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"模型接口调用失败：{last_error}") from last_error


class CounterfactualTestGenerator:
    """生成跨配置反事实测试；外部命令优先，否则直接调用DeepSeek。"""

    def __init__(self, command: str | None, timeout: int = 300,
                 deepseek_key_file: Path | None = None,
                 deepseek_model: str = "deepseek-flash",
                 deepseek_base_url: str = "https://api.deepseek.com"):
        self.command = command
        self.timeout = timeout
        self.deepseek_key_file = deepseek_key_file
        self.deepseek_model = deepseek_model
        self.deepseek_base_url = deepseek_base_url

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
            "required_output_identity": {"class_name": class_name, "method_name": method_name},
            "failed_test_in_B": {
                "identity": failed_test.identity,
                "source": failed_test.full_source,
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
                 failed_test: JavaTestCase, request_dir: Path) -> TestPair:
        request_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{failed_test.class_name.replace('.', '_')}.{failed_test.method_name}"
        request = request_dir / f"{stem}.request.json"
        response = request_dir / f"{stem}.response.json"
        payload = self._payload(passing, failing, failed_test)
        request.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            if self.command:
                answer = self._external(request, response, request_dir)
                metadata = {"provider": "external-command"}
            else:
                api_key = _read_api_key(self.deepseek_key_file)
                if not api_key:
                    return TestPair(failed_test, None, "missing", 0.0, 0.0, request,
                                    "未找到DeepSeek接口密钥，也未配置外部测试生成命令")
                client = DeepSeekClient(
                    api_key, self.deepseek_model, self.deepseek_base_url, self.timeout
                )
                generation_payload = payload
                for generation_round in range(1, 3):
                    answer, metadata = client.complete(generation_payload)
                    try:
                        self._validate_answer(answer, payload)
                        metadata["generation_round"] = generation_round
                        break
                    except ValueError as exc:
                        if generation_round == 2:
                            raise
                        generation_payload = {
                            **payload,
                            "correction_request": (
                                f"上一次响应未通过结构校验：{exc}。请重新生成，严格遵守指定身份和响应结构。"
                            ),
                        }
            self._validate_answer(answer, payload)
            response.write_text(json.dumps({**answer, "_metadata": metadata},
                                           ensure_ascii=False, indent=2), encoding="utf-8")
            source_path = (request_dir / "generated").joinpath(
                *answer["class_name"].split(".")).with_suffix(".java")
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(answer["java_source"], encoding="utf-8")
            cases = [case for case in extract_tests(source_path)
                     if case.method_name == answer["method_name"]]
            if len(cases) != 1:
                raise ValueError("生成结果不含指定的唯一测试方法")
            generated = cases[0]
            similarity, oracle = test_similarity(failed_test, generated)
            return TestPair(failed_test, generated, "generated", similarity, oracle,
                            request, answer["rationale"])
        except Exception as exc:
            (request_dir / f"{stem}.error.txt").write_text(str(exc), encoding="utf-8")
            return TestPair(failed_test, None, "missing", 0.0, 0.0, request,
                            f"反事实测试生成失败：{exc}")
