"""识别主流程：配置近邻 → 测试配对 → 重编译执行 → 因果证据 → 配置排序。

本模块只使用原始通过/失败结果，不读取真假通过评价标签。
注意：当前实现分别对A、B取图推理，然后比较后验；并非联合因子图。
"""

from __future__ import annotations

import csv
import hashlib
import json
import traceback
from dataclasses import asdict
from pathlib import Path

from .brfl_runtime import BrflRuntime
from .causal import assess_pair
from .configurations import feature_delta, load_configurations, nearest_failing
from .domain import Configuration, FactorGraph, JavaTestCase, PairAssessment
from .factor_graph import load_factor_graph
from .feature_sources import load_feature_map
from .generator import CounterfactualTestGenerator
from .java_tests import discover_tests, failed_test_identities, pair_tests


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
                 deepseek_base_url: str = "https://api.deepseek.com"):
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

    def run(self, version_name: str | None = None, plan_only: bool = False) -> dict:
        """处理一个或全部缺陷版本，将各版本计数写入 summary.json。

        plan_only仅跳过Java执行，配对阶段仍可能调用模型和生成测试。
        """
        self.output.mkdir(parents=True, exist_ok=True)
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
        test_cache = {c.name: discover_tests(c.variant_dir) for c in configs}
        feature_maps = {c.name: load_feature_map(c.variant_dir) for c in configs}
        # 第一阶段：只以原本通过配置A为候选，寻找最近失败配置B及其失败测试。
        # 顺序为确定性匹配、模型选择、模型生成；已有相似匹配会直接被接受。
        plan, jobs = [], []
        for config_a in passing:
            for config_b, distance in nearest_failing(config_a, configs, self.neighbors):
                for failed_test in _failed_tests(config_b):
                    pair = pair_tests(failed_test, test_cache[config_a.name], self.similarity_threshold)
                    if pair.passing is None:
                        pair = self.generator.select_existing(
                            config_a, config_b, failed_test, test_cache[config_a.name],
                            out / "generation-requests" / config_a.name / config_b.name,
                        )
                    if pair.passing is None:
                        pair = self.generator.generate(
                            config_a, config_b, failed_test,
                            out / "generation-requests" / config_a.name / config_b.name,
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

        for config_a, config_b, pair, row in jobs:
            try:
                if pair.passing is None:
                    assessment = assess_pair(
                        config_a.name, config_b.name, pair,
                        failing_graph=FactorGraph(),
                        passing_graph=None, passing_outcome=None,
                        passing_source_root=config_a.variant_dir / "src",
                        failing_feature_map=feature_maps[config_b.name],
                        passing_feature_map=feature_maps[config_a.name], top_k=self.top_k,
                    )
                else:
                    generated = pair.provenance == "generated"
                    if generated:
                        # 生成测试需在同一源码下分别对A/B重新编译；短目录避免Windows路径过长。
                        generated_id = hashlib.sha1(
                            f"{config_a.name}|{config_b.name}|{pair.passing.identity}".encode("utf-8")
                        ).hexdigest()[:12]
                        work_b = out / "g" / generated_id / "b"
                        work_a = out / "g" / generated_id / "a"
                        build_b = self.runtime.compile_configuration(config_b, work_b, pair.passing.source_path)
                        build_a = self.runtime.compile_configuration(config_a, work_a, pair.passing.source_path)
                        failing_test = pair.passing
                    else:
                        work_b = out / "configurations" / config_b.name
                        work_a = out / "configurations" / config_a.name
                        build_b, build_a = build(config_b), build(config_a)
                        failing_test = pair.failing
                    if generated:
                        # 先验证执行结果；B必须失败。A也失败时直接形成漏检证据，
                        # 不再依赖因子图导出是否成功。这里仍会加载插桩代理。
                        execution_b = execute(
                            config_b, failing_test, work_b, build_b, require_graph=False
                        )
                        if execution_b.outcome != "failed":
                            raise RuntimeError("生成的反事实测试没有在未通过配置B上复现失败")
                        execution_a = execute(
                            config_a, pair.passing, work_a, build_a, require_graph=False
                        )
                        if execution_a.outcome == "failed":
                            assessment = assess_pair(
                                config_a.name, config_b.name, pair, FactorGraph(), None,
                                execution_a.outcome, config_a.variant_dir / "src",
                                feature_maps[config_b.name], feature_maps[config_a.name],
                                self.top_k,
                            )
                            assessments.append(assessment)
                            continue
                        execution_a = execute(config_a, pair.passing, work_a, build_a)
                        try:
                            execution_b = execute(config_b, failing_test, work_b, build_b)
                        except RuntimeError:
                            # 兼容回退：生成测试的B图失败时使用原失败测试的B图。
                            # 此时A/B图来自不同测试，人工审查应关注其可比性。
                            regular_build_b = build(config_b)
                            execution_b = execute(
                                config_b, pair.failing,
                                out / "configurations" / config_b.name, regular_build_b,
                            )
                    else:
                        execution_b = execute(config_b, failing_test, work_b, build_b)
                        if execution_b.outcome != "failed":
                            raise RuntimeError("反事实测试没有在未通过配置B上复现失败")
                        execution_a = execute(config_a, pair.passing, work_a, build_a)
                    assessment = assess_pair(
                        config_a.name, config_b.name, pair,
                        load_factor_graph(execution_b.graph_path),
                        load_factor_graph(execution_a.graph_path), execution_a.outcome,
                        config_a.variant_dir / "src",
                        feature_maps[config_b.name], feature_maps[config_a.name], self.top_k,
                    )
                assessments.append(assessment)
            except Exception as exc:
                # 运行错误不形成有效阴性证据；ranking中无证据配置仍会显示证据不足。
                errors.append({**row, "error": str(exc), "traceback": traceback.format_exc()})
        assessment_rows = [_serialize_assessment(x) for x in assessments]
        (out / "pair-assessments.json").write_text(
            json.dumps(assessment_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_ranking(out / "ranking.csv", passing, assessments)
        return {"passing_configurations": len(passing), "pairs": len(plan),
                "executed": len(assessments), "errors": len(errors)}

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
