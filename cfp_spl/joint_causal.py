"""统一后验的配置证据：不再比较A/B两次推理的后验差值。"""
from .domain import FactorGraph, PairAssessment, StatementEvidence, TestPair
from .factor_graph import causal_statement_keys
from .feature_sources import source_identity


def trace_comparability(graph_a, graph_b, map_a, map_b):
    def mapped(graph, mapping, paths=False):
        keys = causal_statement_keys(graph) if paths else [s.stable_key for s in graph.statements.values()]
        return {source_identity(key, mapping) for key in keys if source_identity(key, mapping) != key}
    sa, sb = mapped(graph_a, map_a), mapped(graph_b, map_b)
    pa, pb = mapped(graph_a, map_a, True), mapped(graph_b, map_b, True)
    return {"shared_statements": len(sa & sb), "statement_jaccard": len(sa & sb) / max(1, len(sa | sb)),
            "shared_result_path_statements": len(pa & pb),
            "result_path_jaccard": len(pa & pb) / max(1, len(pa | pb)),
            "usable": bool(pa & pb)}


def assess_joint(a: str, b: str, pair: TestPair, joint: FactorGraph,
                 graph_a: FactorGraph, graph_b: FactorGraph, map_a: dict, map_b: dict,
                 threshold: float = 0.55, top_k: int = 20) -> PairAssessment:
    # 候选必须位于B的失败依赖路径，也在A的配对执行路径中出现。
    # 单侧图只用于执行覆盖/依赖关系，没有单侧概率推理。
    path_a = {source_identity(key, map_a) for key in causal_statement_keys(graph_a)}
    path_b = {source_identity(key, map_b) for key in causal_statement_keys(graph_b)}
    common = path_a & path_b
    ranked = sorted((s for s in joint.statements.values()
                     if s.stable_key in common and not s.stable_key.startswith("unmapped:")),
                    key=lambda s: (-s.suspiciousness, s.stable_key))[:top_k]
    evidence = [StatementEvidence(s.stable_key, s.stable_key, s.suspiciousness,
                                  0.0, 0.0, True, True, True) for s in ranked]
    score = max((s.suspiciousness for s in ranked), default=0.0)
    level = "strong_suspect" if ranked and score >= threshold else "insufficient"
    return PairAssessment(a, b, pair.failing.identity,
                          pair.passing.identity if pair.passing else None,
                          "failed", "passed", pair.provenance, pair.similarity, level, score,
                          evidence, ["统一因子图后验；仅检查A/B共同执行且结果依赖可达的特征语句",
                                     "旧字段passing_suspiciousness与posterior_gap不适用，固定为0"])
