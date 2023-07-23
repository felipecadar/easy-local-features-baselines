def test_instantiation():
    from easy_local_features.feature.baseline_aliked import ALIKED_baseline
    model = ALIKED_baseline()
    assert model is not None