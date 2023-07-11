def test_instantiation():
    from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
    model = SuperPoint_baseline()
    assert model is not None