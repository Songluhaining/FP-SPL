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
from .feature_sources import load_feature_map, source_identity
from .generator import CounterfactualTestGenerator
from .java_tests import discover_tests, failed_test_identities, pair_tests
from .spectrum import load_spectrum_priors, serialize_spectrum
from .tensor_inference import infer_factor_graph


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
    def __init__(self, dataset: Path, output: Path, java_home: Path,
                 generator_command: str | None = None, neighbors: int = 1,
                 similarity_threshold: float = 0.55, top_k: int = 20,
                 inference: str = "tensor", spectrum_metric: str = "ensemble",
                 max_region_variables: int = 14):
        self.dataset = dataset.resolve()
        self.output = output.resolve()
        self.runtime = BrflRuntime(self.output / "runtime", java_home.resolve())
        self.generator = CounterfactualTestGenerator(generator_command)
        self.neighbors = neighbors
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.inference = inference
        self.spectrum_metric = spectrum_metric
        self.max_region_variables = max_region_variables

    def run(self, version_name: str | None = None, plan_only: bool = False) -> dict:
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
        out = self.output / version.name
        out.mkdir(parents=True, exist_ok=True)
        configs = load_configurations(version)
        passing = [c for c in configs if c.outcome == "passed"]
        test_cache = {c.name: discover_tests(c.variant_dir) for c in configs}
        feature_maps = {c.name: load_feature_map(c.variant_dir) for c in configs}
        spectrum_priors = {
            c.name: load_spectrum_priors(c.variant_dir, self.spectrum_metric) for c in configs
        }
        spectrum_out = out / "spectrum-priors"
        spectrum_out.mkdir(exist_ok=True)
        for config in configs:
            (spectrum_out / f"{config.name}.json").write_text(
                json.dumps(serialize_spectrum(spectrum_priors[config.name]), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        plan, jobs = [], []
        for config_a in passing:
            for config_b, distance in nearest_failing(config_a, configs, self.neighbors):
                for failed_test in _failed_tests(config_b):
                    pair = pair_tests(failed_test, test_cache[config_a.name], self.similarity_threshold)
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

        assessments, errors = [], []
        build_cache = {}
        execution_cache = {}
        inference_cache = {}

        def build(config: Configuration):
            if config.name not in build_cache:
                build_cache[config.name] = self.runtime.compile_configuration(
                    config, out / "configurations" / config.name
                )
            return build_cache[config.name]

        def execute(config: Configuration, test: JavaTestCase, work: Path, compiled):
            source_digest = hashlib.sha1(test.full_source.encode("utf-8")).hexdigest()
            key = (config.name, test.identity, source_digest, str(work))
            if key not in execution_cache:
                execution_cache[key] = self.runtime.execute(config, test, work, compiled)
            return execution_cache[key]

        def inferred_graph(config: Configuration, execution):
            key = (config.name, str(execution.graph_path), self.inference)
            if key not in inference_cache:
                graph = load_factor_graph(execution.graph_path)
                if self.inference == "tensor":
                    diagnostics = infer_factor_graph(
                        graph, spectrum_priors[config.name], feature_maps[config.name],
                        max_region_variables=self.max_region_variables,
                    )
                    (execution.graph_path.parent / "tensor-inference.json").write_text(
                        json.dumps(diagnostics.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    posterior_path = execution.graph_path.parent / "tensor-posteriors.tsv"
                    with posterior_path.open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.writer(stream, delimiter="\t")
                        writer.writerow((
                            "node", "statement", "feature_source", "spectrum_error_prior",
                            "posterior_error", "observed",
                        ))
                        for node, statement in sorted(graph.statements.items()):
                            source = source_identity(statement.stable_key, feature_maps[config.name])
                            evidence = spectrum_priors[config.name].get(source)
                            writer.writerow((
                                node, statement.stable_key, source,
                                evidence.error_prior if evidence else 0.01,
                                statement.suspiciousness, int(statement.observed),
                            ))
                inference_cache[key] = graph
            return inference_cache[key]

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
                        work_b = out / "generated-builds" / config_b.name / config_a.name / pair.passing.method_name
                        work_a = out / "generated-builds" / config_a.name / config_b.name / pair.passing.method_name
                        build_b = self.runtime.compile_configuration(config_b, work_b, pair.passing.source_path)
                        build_a = self.runtime.compile_configuration(config_a, work_a, pair.passing.source_path)
                        failing_test = pair.passing
                    else:
                        work_b = out / "configurations" / config_b.name
                        work_a = out / "configurations" / config_a.name
                        build_b, build_a = build(config_b), build(config_a)
                        failing_test = pair.failing
                    execution_b = execute(config_b, failing_test, work_b, build_b)
                    if execution_b.outcome != "failed":
                        raise RuntimeError("反事实测试没有在未通过配置B上复现失败")
                    execution_a = execute(config_a, pair.passing, work_a, build_a)
                    assessment = assess_pair(
                        config_a.name, config_b.name, pair,
                        inferred_graph(config_b, execution_b),
                        inferred_graph(config_a, execution_a), execution_a.outcome,
                        config_a.variant_dir / "src",
                        feature_maps[config_b.name], feature_maps[config_a.name], self.top_k,
                    )
                assessments.append(assessment)
            except Exception as exc:
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
