from pathlib import Path

from cfp_spl.causal import assess_pair
from cfp_spl.domain import FactorGraph, JavaTestCase, StatementPosterior, TestPair as CounterfactualPair


def graph(suspicious, observed=True):
    g = FactorGraph()
    g.statements["s0"] = StatementPosterior("s0", "p.C:m:()V#1#0", "p.C:m:()V#1#0", 1 - suspicious, False, False)
    g.values["v0"] = {"observed": observed}
    g.factors["f0"] = {}
    g.incidences = [("f0", "s0", "STMT"), ("f0", "v0", "DEF")]
    return g


def test_source_presence_alone_never_creates_false_positive(tmp_path):
    source = tmp_path / "p" / "C.java"
    source.parent.mkdir()
    source.write_text("class C {}\n", encoding="utf-8")
    test = JavaTestCase("p.T", "test0", source, "assertTrue(true);", "")
    pair = CounterfactualPair(test, test, "exact", 1.0, 1.0)
    result = assess_pair("A", "B", pair, graph(0.9), graph(0.9), "passed", tmp_path)
    assert result.evidence_level == "insufficient"


def test_direct_counterfactual_failure_confirms_false_passing(tmp_path):
    test = JavaTestCase("p.T", "test0", tmp_path / "T.java", "", "")
    pair = CounterfactualPair(test, test, "exact", 1.0, 1.0)
    result = assess_pair("A", "B", pair, graph(0.9), None, "failed", tmp_path)
    assert result.evidence_level == "confirmed_false_passing"
