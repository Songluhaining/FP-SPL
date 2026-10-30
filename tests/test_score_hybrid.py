from cfp_spl.score_hybrid import _evaluate


def test_equal_budget_uses_highest_scores():
    keys = [("v", name) for name in ("a", "b", "c", "d")]
    truth = {keys[0]: 1, keys[1]: 0, keys[2]: 1, keys[3]: 0}
    scores = {keys[0]: 0.9, keys[1]: 0.8, keys[2]: 0.7, keys[3]: 0.1}
    result = _evaluate(keys, truth, scores, budget=2)
    assert result["predicted_positive"] == 2
    assert result["confusion"]["true_positive"] == 1
    assert result["confusion"]["false_positive"] == 1
