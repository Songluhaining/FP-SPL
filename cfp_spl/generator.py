from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from .configurations import feature_delta
from .domain import Configuration, JavaTestCase, TestPair
from .java_tests import extract_tests, oracle_signature, test_similarity


class CounterfactualTestGenerator:
    """通过外部命令接入任意大模型，仓库本身不绑定服务商。"""

    def __init__(self, command: str | None, timeout: int = 300):
        self.command = command
        self.timeout = timeout

    def generate(
        self,
        passing: Configuration,
        failing: Configuration,
        failed_test: JavaTestCase,
        request_dir: Path,
    ) -> TestPair:
        request_dir.mkdir(parents=True, exist_ok=True)
        request = request_dir / f"{failed_test.class_name.replace('.', '_')}.{failed_test.method_name}.request.json"
        response = request.with_suffix(".response.json")
        payload = {
            "task": "生成可同时在两个配置编译的JUnit反事实测试；保持原测试意图与判定条件",
            "passing_configuration": passing.name,
            "failing_configuration": failing.name,
            "feature_delta": feature_delta(passing, failing),
            "failed_test": {
                "identity": failed_test.identity,
                "source": failed_test.full_source,
                "body": failed_test.body,
                "oracle_signature": sorted(oracle_signature(failed_test)),
            },
            "constraints": [
                "输出一个完整、可独立编译的JUnit测试类",
                "使用不会与数据集原有测试类重名的新类名",
                "不得删除或弱化原测试的判定条件",
                "测试失败时必须指向与原失败测试相同的可观察行为",
                "只使用两个配置都可调用的产品接口",
            ],
            "response_schema": {
                "class_name": "含包名的类名",
                "method_name": "测试方法名",
                "java_source": "完整Java源代码",
                "rationale": "为何具有反事实可比性",
            },
        }
        request.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if not self.command:
            return TestPair(failed_test, None, "missing", 0.0, 0.0, request, "未配置测试生成命令")
        command = [part.format(request=str(request), response=str(response)) for part in shlex.split(self.command)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout, check=False)
        (request_dir / f"{request.stem}.generator.log").write_text(
            completed.stdout + "\n[STDERR]\n" + completed.stderr, encoding="utf-8"
        )
        if completed.returncode or not response.is_file():
            return TestPair(failed_test, None, "missing", 0.0, 0.0, request, "测试生成命令失败")
        answer = json.loads(response.read_text(encoding="utf-8"))
        source_dir = request_dir / "generated"
        source_path = source_dir.joinpath(*answer["class_name"].split(".")).with_suffix(".java")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(answer["java_source"], encoding="utf-8")
        cases = [x for x in extract_tests(source_path) if x.method_name == answer["method_name"]]
        if len(cases) != 1:
            return TestPair(failed_test, None, "missing", 0.0, 0.0, request, "生成结果不含指定测试方法")
        generated = cases[0]
        similarity, oracle = test_similarity(failed_test, generated)
        return TestPair(failed_test, generated, "generated", similarity, oracle, request, answer.get("rationale", ""))
