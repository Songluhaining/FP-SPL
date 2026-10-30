from __future__ import annotations

from pathlib import Path

from .domain import FactorGraph, PairAssessment, StatementEvidence, TestPair
from .factor_graph import causal_statement_keys, statement_index
from .feature_sources import FeatureMap, source_identity


def source_contains(stable_key: str, source_root: Path) -> bool:
    parts = stable_key.rsplit("#", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return False
    head, line, _ = parts
    class_name = head.split(":", 1)[0]
    relative = Path(*class_name.split("."))
    candidates = list(source_root.rglob(relative.name.split("$")[0] + ".java"))
    return any(int(line) <= len(p.read_text(encoding="utf-8", errors="replace").splitlines()) for p in candidates)


def assess_pair(
    passing_config: str,
    failing_config: str,
    pair: TestPair,
    failing_graph: FactorGraph,
    passing_graph: FactorGraph | None,
    passing_outcome: str | None,
    passing_source_root: Path,
    failing_feature_map: FeatureMap | None = None,
    passing_feature_map: FeatureMap | None = None,
    top_k: int = 20,
    min_failing_suspiciousness: float = 0.55,
    min_gap: float = 0.15,
) -> PairAssessment:
    if passing_outcome == "failed":
        return PairAssessment(
            passing_config, failing_config, pair.failing.identity,
            pair.passing.identity if pair.passing else None, "failed", "failed",
            pair.provenance, pair.similarity, "confirmed_false_passing", 1.0, [],
            ["反事实测试已在原本通过的配置上复现失败"],
        )
    if passing_graph is None:
        return PairAssessment(
            passing_config, failing_config, pair.failing.identity, None, "failed", None,
            pair.provenance, pair.similarity, "insufficient", 0.0, [],
            ["缺少经过编译和结果验证的反事实通过测试"],
        )
    failing_feature_map = failing_feature_map or {}
    passing_feature_map = passing_feature_map or {}
    failed_by_location, passed_by_location = statement_index(failing_graph), statement_index(passing_graph)
    failed_index = {}
    for stmt in failed_by_location.values():
        identity = source_identity(stmt.stable_key, failing_feature_map)
        if identity not in failed_index or stmt.suspiciousness > failed_index[identity].suspiciousness:
            failed_index[identity] = stmt
    passed_index = {}
    for stmt in passed_by_location.values():
        identity = source_identity(stmt.stable_key, passing_feature_map)
        if identity not in passed_index or stmt.suspiciousness > passed_index[identity].suspiciousness:
            passed_index[identity] = stmt
    passing_cone = {
        source_identity(key, passing_feature_map) for key in causal_statement_keys(passing_graph)
    }
    ranked = sorted(failed_index.values(), key=lambda x: (-x.suspiciousness, x.stable_key))[:top_k]
    evidence = []
    for stmt in ranked:
        identity = source_identity(stmt.stable_key, failing_feature_map)
        counterpart = passed_index.get(identity)
        passing_susp = counterpart.suspiciousness if counterpart else 0.0
        evidence.append(
            StatementEvidence(
                stmt.stable_key, identity, stmt.suspiciousness, passing_susp,
                stmt.suspiciousness - passing_susp,
                identity in set(passing_feature_map.values()) or source_contains(stmt.stable_key, passing_source_root),
                counterpart is not None,
                identity in passing_cone,
            )
        )
    qualifying = [
        x for x in evidence
        if x.failing_suspiciousness >= min_failing_suspiciousness
        and x.posterior_gap >= min_gap and x.present_in_a and x.executed_in_a and x.causal_in_a
    ]
    overlap = len(qualifying) / max(1, min(top_k, len(ranked)))
    gap = sum(x.posterior_gap for x in qualifying) / max(1, len(qualifying))
    test_strength = {"exact": 1.0, "model_selected": 0.80,
                     "generated": 0.75, "similar": 0.65}.get(pair.provenance, 0.0)
    score = min(0.99, 0.35 * test_strength + 0.25 * pair.similarity + 0.25 * min(1.0, overlap * 5) + 0.15 * max(0.0, gap))
    reasons = []
    if qualifying:
        reasons.append(f"{len(qualifying)}条B侧高可疑语句在A中实际执行并位于结果因果路径")
    if not qualifying:
        reasons.append("没有语句同时满足源码存在、实际执行、因果可达和后验差异")
    strong_test = pair.provenance == "exact" or (
        pair.provenance in {"model_selected", "generated"} and pair.oracle_similarity >= 0.8
    )
    if strong_test and len(qualifying) >= 2 and score >= 0.65:
        level = "strong_suspect"
    elif qualifying and score >= 0.4:
        level = "weak_suspect"
    else:
        level = "insufficient"
    return PairAssessment(
        passing_config, failing_config, pair.failing.identity,
        pair.passing.identity if pair.passing else None, "failed", "passed",
        pair.provenance, pair.similarity, level, score, evidence, reasons,
    )
