def test_instantiation():
    from easy_local_features.matching.baseline_loftr import LoFTR_baseline
    model = LoFTR_baseline()
    assert model is not None