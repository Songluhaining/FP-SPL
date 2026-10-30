from cfp_spl.evaluate_clap import _top_budget


def test_top_budget_marks_only_highest_scores():
    rows = [
        {"version": "v", "configuration": name, "score": score, "prediction": 0}
        for name, score in (("a", 0.1), ("b", 0.9), ("c", 0.5))
    ]
    selected = _top_budget(rows, 2)
    assert {row["configuration"] for row in selected if row["prediction"]} == {"b", "c"}
