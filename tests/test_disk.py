def test_instantiation():
    from easy_local_features.baseline_disk import DISK_baseline
    model = DISK_baseline()
    assert model is not None