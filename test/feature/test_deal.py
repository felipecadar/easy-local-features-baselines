def test_instantiation():
    from easy_local_features.feature.baseline_deal import DEAL_baseline
    model = DEAL_baseline()
    assert model is not None