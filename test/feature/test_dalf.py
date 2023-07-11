def test_instantiation():
    from easy_local_features.feature.baseline_dalf import DALF_baseline
    model = DALF_baseline()
    assert model is not None