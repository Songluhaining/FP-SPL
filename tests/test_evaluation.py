from cfp_spl.evaluation import _average_precision, _metrics, _roc_auc


def test_ranking_metrics_handle_ties_without_name_order_bias():
    labels = [1, 0, 1, 0]
    scores = [0.0, 0.0, 0.0, 0.0]
    assert _average_precision(labels, scores) == 0.5
    assert _roc_auc(labels, scores) == 0.5
    result = _metrics(labels, [0, 0, 0, 0], scores)
    assert result["false_passing_recall"] == 0.0
    assert result["accuracy"] == 0.5
