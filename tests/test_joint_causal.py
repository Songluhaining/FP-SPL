from cfp_spl.domain import FactorGraph, JavaTestCase, StatementPosterior, TestPair as Pair
from cfp_spl.joint_causal import assess_joint, trace_comparability


def structure(raw):
    graph = FactorGraph()
    graph.statements["s"] = StatementPosterior("s", raw, raw, .5, False, False)
    graph.values["v"] = {"observed": True}
    graph.incidences = [("f", "s", "STMT"), ("f", "v", "DEF")]
    return graph


def test_joint_posterior_uses_shared_source_without_posterior_gap(tmp_path):
    ga, gb = structure("p.C:m:()V#20#0"), structure("p.C:m:()V#30#7")
    ma, mb = {("p.C", 20): "Base.p.C:9"}, {("p.C", 30): "Base.p.C:9"}
    joint = structure("Base.p.C:9")
    joint.statements["s"] = StatementPosterior("s", "Base.p.C:9", "Base.p.C:9", .1, False, False)
    test = JavaTestCase("p.T", "test0", tmp_path / "T.java", "", "")
    pair = Pair(test, test, "exact", 1., 1.)
    assessment = assess_joint("A", "B", pair, joint, ga, gb, ma, mb)
    assert assessment.evidence_level == "strong_suspect"
    assert abs(assessment.score - .9) < 1e-8
    assert assessment.statements[0].posterior_gap == 0
    assert trace_comparability(ga, gb, ma, mb)["usable"]


def test_different_feature_implementations_do_not_form_common_result_path():
    ga, gb = structure("p.C:m:()V#20#0"), structure("p.C:m:()V#20#0")
    result = trace_comparability(ga, gb, {("p.C", 20): "Base.p.C:9"},
                               {("p.C", 20): "Encrypt.p.C:9"})
    assert not result["usable"]
    assert result["shared_result_path_statements"] == 0
