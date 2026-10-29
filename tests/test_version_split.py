from cfp_spl.version_split import fold_of, split_versions


def test_version_fold_is_stable_across_partial_version_sets():
    versions = {f"ID_{number}" for number in range(30)}
    training, validation = split_versions(versions, validation_fold=2)
    partial = versions - {"ID_1", "ID_8", "ID_19"}
    partial_training, partial_validation = split_versions(partial, validation_fold=2)
    assert partial_validation == validation & partial
    assert partial_training == training & partial
    assert training.isdisjoint(validation)
    assert training | validation == versions


def test_invalid_version_fold_is_rejected():
    try:
        fold_of("ID_1", 1)
    except ValueError as exc:
        assert "至少为二" in str(exc)
    else:
        raise AssertionError("应拒绝只有一折的切分")
