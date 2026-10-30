"""识别主流程：配置近邻 → 测试配对 → 重编译执行 → 因果证据 → 配置排序。

本模块只使用原始通过/失败结果，不读取真假通过评价标签。
各执行先独立解析依赖结构，再共享特征源码语句节点，只在统一图上推理。
"""

from __future__ import annotations

import csv
import hashlib
import json
import traceback
import platform
from dataclasses import asdict
from pathlib import Path

from .brfl_runtime import BrflRuntime, failure_signature
from .causal import assess_pair
from .configurations import feature_delta, load_configurations, nearest_failing
from .domain import Configuration, FactorGraph, JavaTestCase, PairAssessment, TestPair
from .factor_graph import load_factor_graph
from .feature_sources import load_feature_map
from .generator import CounterfactualTestGenerator
from .java_tests import discover_tests, failed_test_identities, pair_tests
from .joint_causal import assess_joint, trace_comparability
from datetime import datetime, timezone


def _failed_tests(config: Configuration) -> list[JavaTestCase]:
    identities = failed_test_identities(config.variant_dir)
    tests = discover_tests(config.variant_dir)
    selected = [test for test in tests if test.identity in identities]
    missing = identities - {test.identity for test in selected}
    if missing:
        raise ValueError(f"覆盖报告中的失败测试在源码中不存在：{sorted(missing)}")
    return selected


def _serialize_assessment(value: PairAssessment) -> dict:
    return asdict(value)


class CounterfactualPipeline:
    """组织各模块；人工检查算法时建议先读 _run_version。"""
    def __init__(self, dataset: Path, output: Path, java_home: Path,
                 generator_command: str | None = None, neighbors: int = 1,
                 similarity_threshold: float = 0.55, top_k: int = 20,
                 deepseek_key_file: Path | None = None,
                 deepseek_model: str = "deepseek-chat",
                 deepseek_base_url: str = "https://api.deepseek.com", max_passing: int = 0):
        self.dataset = dataset.resolve()
        self.output = output.resolve()
        self.runtime = BrflRuntime(self.output / "runtime", java_home.resolve())
        self.generator = CounterfactualTestGenerator(
            generator_command, deepseek_key_file=deepseek_key_file,
            deepseek_model=deepseek_model, deepseek_base_url=deepseek_base_url,
        )
        self.neighbors = neighbors
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.max_passing = max_passing

    def run(self, version_name: str | None = None, plan_only: bool = False) -> dict:
        """处理一个或全部缺陷版本，将各版本计数写入 summary.json。

        plan_only仅跳过Java执行，配对阶段仍可能调用模型和生成测试。
        """
        self.output.mkdir(parents=True, exist_ok=True)
        code_root = Path(__file__).resolve().parent.parent
        sources = [*sorted((code_root / "cfp_spl").glob("*.py")),
                   *sorted((code_root / "java").rglob("*.java")),
                   *sorted((code_root / "vendor" / "brfl" / "src").rglob("*.java"))]
        (self.output / "run-manifest.json").write_text(json.dumps({
            "started_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(self.dataset),
            "version": version_name, "max_passing": self.max_passing, "plan_only": plan_only,
            "neighbors_with_boundary_ties": self.neighbors, "model": self.generator.deepseek_model,
            "model_endpoint": self.generator.deepseek_base_url, "python": platform.python_version(),
            "inference": "one_joint_graph_per_A_BRFL_LBP", "ground_truth_used": False,
            "source_sha256": {str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources}}, ensure_ascii=False, indent=2), encoding="utf-8")
        versions = [self.dataset / version_name] if version_name else [
            p for p in sorted(self.dataset.iterdir()) if (p / "config.report.csv.done").is_file()
        ]
        if not versions:
            raise FileNotFoundError("没有找到含config.report.csv.done的缺陷版本")
        if not plan_only:
            self.runtime.build_brfl()
        summaries = {}
        for version in versions:
            summaries[version.name] = self._run_version(version, plan_only)
        (self.output / "summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summaries

    def _run_version(self, version: Path, plan_only: bool) -> dict:
        """一个缺陷版本内完成配对、执行和排序，各配对异常单独记录。"""
        out = self.output / version.name
        out.mkdir(parents=True, exist_ok=True)
        configs = load_configurations(version)
        passing = [c for c in configs if c.outcome == "passed"]
        if self.max_passing:
            passing = passing[:self.max_passing]
        test_cache = {c.name: discover_tests(c.variant_dir) for c in configs}
        feature_maps = {c.name: load_feature_map(c.variant_dir) for c in configs}
        # 第一阶段：只以原本通过配置A为候选，寻找最近失败配置B及其失败测试。
        # 顺序为确定性匹配、模型选择、模型生成；已有相似匹配会直接被接受。
        plan, jobs = [], []
        self._direct_generated = set()
        failure_builds = {}
        for config_a in passing:
            for config_b, distance in nearest_failing(config_a, configs, self.neighbors):
                for failed_test in _failed_tests(config_b):
                    skip_after_direct = config_a.name in self._direct_generated
                    context_key = (str(config_b.variant_dir), failed_test.identity)
                    if not plan_only and not skip_after_direct and context_key not in self.generator.failure_contexts:
                        work = out / "original-failures" / config_b.name
                        try:
                            if config_b.name not in failure_builds:
                                failure_builds[config_b.name] = self.runtime.compile_configuration(config_b, work)
                            execution = self.runtime.execute(config_b, failed_test, work,
                                                             failure_builds[config_b.name], require_graph=False)
                            diagnostic = execution.log_path.read_text(encoding="utf-8", errors="replace")
                            diagnostic = "\n".join(line for line in diagnostic.splitlines()
                                                   if not line.startswith(("opcode", "###")))
                            if "Failures (" in diagnostic:
                                diagnostic = diagnostic[diagnostic.index("Failures ("):].split("Test run finished", 1)[0]
                            self.generator.failure_contexts[context_key] = {
                                "outcome": execution.outcome, "expected": execution.expected,
                                "actual": execution.actual, "failure_signature": failure_signature(diagnostic),
                                "failure_diagnostic": diagnostic[-12000:]}
                        except Exception as exc:
                            self.generator.failure_contexts[context_key] = {"diagnostic_error": str(exc)}
                    with (out / "planning-events.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                            "event": "test_pair_planning", "configuration_A": config_a.name,
                            "configuration_B": config_b.name, "failed_test": failed_test.identity,
                            "distance": distance}, ensure_ascii=False) + "\n")
                    pair = pair_tests(failed_test, test_cache[config_a.name], self.similarity_threshold)
                    if skip_after_direct:
                        pair = TestPair(failed_test, None, "skipped_after_direct_evidence", 0.0, 0.0,
                                        rationale="A已通过新测试复现故障，保留全部并列近邻计划，停止该A的额外模型调用")
                    if not skip_after_direct and pair.provenance != "exact":
                        pair = self.generator.select_existing(
                            config_a, config_b, failed_test, test_cache[config_a.name],
                            out / "generation-requests" / config_a.name / config_b.name,
                        )
                    if not skip_after_direct and pair.passing is None:
                        pair = self.generator.generate(
                            config_a, config_b, failed_test,
                            out / "generation-requests" / config_a.name / config_b.name,
                            validator=None if plan_only else lambda test, a=config_a, b=config_b:
                                self._validate_generated(a, b, test, out, failed_test),
                        )
                    row = {
                        "passing_configuration": config_a.name,
                        "failing_configuration": config_b.name,
                        "euclidean_distance": distance,
                        "feature_delta": feature_delta(config_a, config_b),
                        "failing_test": failed_test.identity,
                        "passing_test": pair.passing.identity if pair.passing else None,
                        "test_provenance": pair.provenance,
                        "test_similarity": pair.similarity,
                        "oracle_similarity": pair.oracle_similarity,
                        "generation_request": str(pair.generation_request) if pair.generation_request else None,
                        "rationale": pair.rationale,
                    }
                    plan.append(row)
                    jobs.append((config_a, config_b, pair, row))
        (out / "pair-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        if plan_only:
            return {"passing_configurations": len(passing), "pairs": len(plan), "executed": 0}

        # 第二阶段：按配对实际执行。缓存仅避免本次运行中重复编译和重复取图。
        assessments, errors = [], []
        build_cache = {}
        execution_cache = {}

        def build(config: Configuration):
            if config.name not in build_cache:
                build_cache[config.name] = self.runtime.compile_configuration(
                    config, out / "configurations" / config.name
                )
            return build_cache[config.name]

        def execute(config: Configuration, test: JavaTestCase, work: Path, compiled,
                    require_graph: bool = True):
            source_digest = hashlib.sha1(test.full_source.encode("utf-8")).hexdigest()
            key = (config.name, test.identity, source_digest, str(work), require_graph)
            if key not in execution_cache:
                execution_cache[key] = self.runtime.execute(
                    config, test, work, compiled, require_graph=require_graph
                )
            return execution_cache[key]

        # 每个A的所有最近B/失败测试共同进入一个图；相同执行仅加载一次。
        pending = {c.name: [] for c in passing}
        event_file = out / "events.jsonl"
        def event(kind, **data):
            with event_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                         "event": kind, **data}, ensure_ascii=False) + "\n")
        for config_a, config_b, pair, row in jobs:
            event("pair_started", **row)
            try:
                if pair.passing is None:
                    assessment = assess_pair(
                        config_a.name, config_b.name, pair, FactorGraph(), None, None,
                        config_a.variant_dir / "src", feature_maps[config_b.name],
                        feature_maps[config_a.name], self.top_k)
                    assessments.append(assessment)
                    event("missing_counterfactual", configuration=config_a.name, reason=pair.rationale)
                    continue
                generated = pair.provenance == "generated"
                if generated:
                    digest = hashlib.sha256((config_a.name + config_b.name + pair.passing.full_source).encode()).hexdigest()[:16]
                    work_a, work_b = out / "g" / digest / "a", out / "g" / digest / "b"
                    build_b = self.runtime.compile_configuration(config_b, work_b, pair.passing.source_path)
                    build_a = self.runtime.compile_configuration(config_a, work_a, pair.passing.source_path)
                    failing_test = pair.passing
                else:
                    work_a, work_b = out / "configurations" / config_a.name, out / "configurations" / config_b.name
                    build_a, build_b = build(config_a), build(config_b)
                    failing_test = pair.failing
                # 只检查结果时仍记录轨迹。后续结构导出不进行单侧推理。
                execution_b = execute(config_b, failing_test, work_b, build_b, require_graph=False)
                execution_a = execute(config_a, pair.passing, work_a, build_a, require_graph=False)
                if execution_b.outcome != "failed":
                    raise RuntimeError("B没有复现失败")
                if execution_a.outcome == "failed":
                    if not generated:
                        raise RuntimeError("选中的已有A测试未通过，不是反事实通过证据")
                    assessments.append(assess_pair(
                        config_a.name, config_b.name, pair, FactorGraph(), None, "failed",
                        config_a.variant_dir / "src", feature_maps[config_b.name],
                        feature_maps[config_a.name], self.top_k))
                    event("direct_failure", configuration=config_a.name, test=pair.passing.identity)
                    continue
                execution_b = execute(config_b, failing_test, work_b, build_b)
                execution_a = execute(config_a, pair.passing, work_a, build_a)
                comparison = trace_comparability(load_factor_graph(execution_a.graph_path),
                    load_factor_graph(execution_b.graph_path), feature_maps[config_a.name], feature_maps[config_b.name])
                event("trace_comparability", configuration=config_a.name, reference=config_b.name, **comparison)
                if not comparison["usable"]:
                    if not generated:
                        # 即使文字或模型认为相近，动态结果依赖路径仍需验证。
                        # 原配对保留在事件日志，尝试生成一次替代，不无限扩张任务。
                        replacement = self.generator.generate(config_a, config_b, pair.failing,
                            out / "generation-requests" / config_a.name / config_b.name,
                            validator=lambda test, a=config_a, b=config_b: self._validate_generated(a, b, test, out, pair.failing))
                        replacement_row = {**row, "test_provenance": replacement.provenance,
                                           "passing_test": replacement.passing.identity if replacement.passing else None,
                                           "rationale": replacement.rationale}
                        jobs.append((config_a, config_b, replacement, replacement_row))
                        event("trace_mismatch_generation_fallback", configuration=config_a.name)
                        continue
                    raise RuntimeError("生成测试的A/B结果路径没有共同特征语句，证据不足")
                pending[config_a.name].append((config_a, config_b, pair, execution_a, execution_b,
                                                pair.passing, failing_test))
                event("pair_validated", configuration=config_a.name, reference=config_b.name)
            except Exception as exc:
                errors.append({**row, "phase": "execution", "error": str(exc), "traceback": traceback.format_exc()})
                event("pair_error", configuration=config_a.name, error=str(exc))
        for name, items in pending.items():
            if not items:
                continue
            try:
                executions, seen = [], set()
                for a, b, pair, ea, eb, ta, tb in items:
                    for config, test, execution in ((a, ta, ea), (b, tb, eb)):
                        key = (config.name, test.identity, str(execution.log_path))
                        if key not in seen:
                            executions.append((config, test, execution))
                            seen.add(key)
                event("joint_inference_started", configuration=name, executions=len(executions))
                joint_path = self.runtime.infer_joint(executions, out / "joint" / name)
                joint = load_factor_graph(joint_path)
                # 可追踪的过程验证：图中必须同时包含真假结果观测，静态身份不能重复。
                observations = [v["observed_value"] for v in joint.values.values() if v["observed"]]
                if True not in observations or False not in observations:
                    raise RuntimeError("联合图没有同时包含通过与失败的观测值")
                identities = [s.stable_key for s in joint.statements.values()]
                if len(set(identities)) != len(identities):
                    raise RuntimeError("联合图出现重复静态语句身份")
                for a, b, pair, ea, eb, ta, tb in items:
                    assessments.append(assess_joint(a.name, b.name, pair, joint,
                        load_factor_graph(ea.graph_path), load_factor_graph(eb.graph_path),
                        feature_maps[a.name], feature_maps[b.name], top_k=self.top_k))
                event("joint_inference_finished", configuration=name, statements=len(joint.statements),
                      values=len(joint.values), factors=len(joint.factors),
                      passed_observations=observations.count(True), failed_observations=observations.count(False))
            except Exception as exc:
                errors.append({"passing_configuration": name, "phase": "joint_inference",
                               "error": str(exc), "traceback": traceback.format_exc()})
                event("joint_inference_error", configuration=name, error=str(exc))
        assessment_rows = [_serialize_assessment(x) for x in assessments]
        (out / "pair-assessments.json").write_text(
            json.dumps(assessment_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_ranking(out / "ranking.csv", passing, assessments)
        return {"passing_configurations": len(passing), "pairs": len(plan),
                "executed": len(assessments), "errors": len(errors)}

    def _validate_generated(self, a: Configuration, b: Configuration,
                            test: JavaTestCase, out: Path, original_test: JavaTestCase | None = None) -> dict:
        """反馈真实编译诊断；B必须断言失败，A异常失败不能当作确认假通过。

        每个源码版本使用独立目录，避免修复后误用上轮class和执行结果。
        编译执行只能验证机械条件，语义等价性仍需人工检查请求与测试。
        """
        digest = hashlib.sha256((a.name + b.name + test.full_source).encode()).hexdigest()[:16]
        result = {"valid": False, "executions": {}}
        original = self.generator.failure_context(b, original_test).get("failure_signature", {}) if original_test else {}
        structures = {}
        for side, config in (("B", b), ("A", a)):
            work = out / "v" / digest / side
            try:
                compiled = self.runtime.compile_configuration(config, work, test.source_path)
                execution = self.runtime.execute(config, test, work, compiled, require_graph=False)
                log = execution.log_path.read_text(encoding="utf-8", errors="replace")
                signature = failure_signature(log)
                assertion_failure = signature.get("assertion", False)
                same_exception = bool(original and not original.get("assertion")
                    and signature.get("exception") == original.get("exception")
                    and signature.get("product_method") == original.get("product_method"))
                result["executions"][side] = {"outcome": execution.outcome,
                                                 "assertion_failure": assertion_failure,
                                                 "failure_signature": signature, "matches_original_exception": same_exception}
                if execution.outcome == "failed" and not assertion_failure and not same_exception:
                    raise ValueError("执行因非断言异常失败，不构成可靠故障证据")
                if execution.outcome == "failed" and original and not original.get("assertion") and not same_exception:
                    raise ValueError("没有复现原产品异常的类型和位置，不能替换原故障判定")
                if side == "B" and execution.outcome != "failed":
                    raise ValueError("B未复现失败；不能通过修改断言强行复现")
                if side == "B" or execution.outcome == "passed":
                    structure = self.runtime.execute(config, test, work, compiled, require_graph=True)
                    structures[side] = load_factor_graph(structure.graph_path)
            except Exception as exc:
                result["side"] = side
                result["error"] = str(exc)
                # 只反馈本次生成的诊断日志，不传密钥或环境变量。
                result["diagnostics"] = {
                    name: path.read_text(encoding="utf-8", errors="replace")[-16000:]
                    for name, path in (("compile", work / "compile.log"),
                                       ("execution", next((work / "runs").glob("*/outcome/test-run.log"),
                                                          work / "missing-execution.log")))
                    if path.is_file()
                }
                return result
        if result["executions"]["A"]["outcome"] == "passed":
            comparison = trace_comparability(structures["A"], structures["B"],
                                             load_feature_map(a.variant_dir), load_feature_map(b.variant_dir))
            result["trace_comparability"] = comparison
            if not comparison["usable"]:
                result["error"] = "A/B结果依赖路径没有共同的特征源码语句，请保持被测功能与输入场景"
                return result
        result["valid"] = True
        if result["executions"]["A"]["outcome"] == "failed":
            if not hasattr(self, "_direct_generated"):
                self._direct_generated = set()
            self._direct_generated.add(a.name)
        return result

    @staticmethod
    def _write_ranking(path: Path, passing: list[Configuration], assessments: list[PairAssessment]) -> None:
        """按证据等级、最佳配对分数排序；多个配对不做概率融合。"""
        by_config: dict[str, list[PairAssessment]] = {c.name: [] for c in passing}
        for item in assessments:
            by_config[item.passing_configuration].append(item)
        rows = []
        priority = {"confirmed_false_passing": 3, "strong_suspect": 2, "weak_suspect": 1, "insufficient": 0}
        for name, evidence in by_config.items():
            best = max(evidence, key=lambda x: (priority[x.evidence_level], x.score), default=None)
            rows.append({
                "configuration": name,
                "evidence_level": best.evidence_level if best else "insufficient",
                "score": best.score if best else 0.0,
                "reference_configuration": best.failing_configuration if best else "",
                "reference_test": best.failing_test if best else "",
                "counterfactual_test": best.passing_test if best else "",
                "supporting_pairs": sum(x.evidence_level != "insufficient" for x in evidence),
                "total_pairs": len(evidence),
            })
        rows.sort(key=lambda x: (-priority[x["evidence_level"]], -float(x["score"]), x["configuration"]))
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["configuration"])
            writer.writeheader()
            writer.writerows(rows)
