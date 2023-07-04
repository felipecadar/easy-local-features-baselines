def test_instantiation():
    from easy_local_features.features.baseline_r2d2 import R2D2_baseline
    model = R2D2_baseline()
    assert model is not None