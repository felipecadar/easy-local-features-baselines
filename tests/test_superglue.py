def test_instantiation():
    from easy_local_features.matching.baseline_superglue import SuperGlue_baseline
    model = SuperGlue_baseline()
    assert model is not None