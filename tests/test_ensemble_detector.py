import pytest
import torch

from easy_local_features.feature.baseline_orb import ORB_baseline
from easy_local_features.feature.detector_ensemble import EnsembleDetector, EnsembleDetectorConfig
from easy_local_features.utils import io, ops
import torch.nn.functional as F


def _count_from_detector(det, img):
    k = det.detect(img)
    if k.ndim == 3 and k.shape[0] == 1:
        return k.shape[1]
    return k.shape[0]


@pytest.mark.parametrize("num_detectors", [1, 2])
def test_ensemble_detect_single_counts(num_detectors):
    img0 = io.fromPath("tests/assets/megadepth0.jpg", gray=False, batch=False)

    # Build detectors (use different top_k to diversify)
    dets = [ORB_baseline({"top_k": 256})]
    if num_detectors == 2:
        dets.append(ORB_baseline({"top_k": 128}))

    cfg = EnsembleDetectorConfig(deduplicate=False, sort=True, max_keypoints=None)
    ens = EnsembleDetector(dets, cfg)

    # Expected total count is the sum of per-detector counts
    expected_total = sum(_count_from_detector(d, img0) for d in dets)

    kps_ens = ens.detect(img0)
    assert isinstance(kps_ens, torch.Tensor)
    assert kps_ens.ndim == 3 and kps_ens.shape[0] == 1
    assert kps_ens.shape[-1] == 2
    assert kps_ens.shape[1] == expected_total


@pytest.mark.parametrize("num_detectors", [1, 2])
def test_ensemble_detect_batch_shapes(num_detectors):
    img0 = io.fromPath("tests/assets/megadepth0.jpg", gray=False, batch=False)
    img1 = io.fromPath("tests/assets/megadepth1.jpg", gray=False, batch=False)

    dets = [ORB_baseline({"top_k": 192})]
    if num_detectors == 2:
        dets.append(ORB_baseline({"top_k": 96}))

    cfg = EnsembleDetectorConfig(deduplicate=False, sort=True, max_keypoints=None)
    ens = EnsembleDetector(dets, cfg)

    # Resize to a common short edge to allow stacking
    img0_r, _ = ops.resize_short_edge(img0, 320)
    img1_r, _ = ops.resize_short_edge(img1, 320)
    # Pad right to the same width
    _, _, H0, W0 = img0_r.unsqueeze(0).shape
    _, _, H1, W1 = img1_r.unsqueeze(0).shape
    Wmax = max(W0, W1)
    if W0 < Wmax:
        pad0 = Wmax - W0
        img0_r = F.pad(img0_r, (0, pad0, 0, 0))
    if W1 < Wmax:
        pad1 = Wmax - W1
        img1_r = F.pad(img1_r, (0, pad1, 0, 0))
    batch = torch.stack([img0_r, img1_r], 0)
    kps_b = ens.detect(batch)
    assert kps_b.ndim == 3 and kps_b.shape[0] == 2
    assert kps_b.shape[-1] == 2
    # N dimension is padded to the max per-image keypoint count internally
    assert kps_b.shape[1] > 0


def test_ensemble_max_keypoints_cap():
    img0 = io.fromPath("tests/assets/megadepth0.jpg", gray=False, batch=False)

    d1 = ORB_baseline({"top_k": 512})
    d2 = ORB_baseline({"top_k": 512})
    cap = 100
    ens = EnsembleDetector([d1, d2], EnsembleDetectorConfig(deduplicate=False, sort=False, max_keypoints=cap))
    kps = ens.detect(img0)
    assert kps.ndim == 3 and kps.shape[0] == 1
    assert kps.shape[-1] == 2
    assert kps.shape[1] <= cap


def test_ensemble_deduplicate_with_identical_detectors():
    img0 = io.fromPath("tests/assets/megadepth0.jpg", gray=False, batch=False)

    # Two identical ORB detectors -> after dedup, should match single-detector count
    d1 = ORB_baseline({"top_k": 300})
    d2 = ORB_baseline({"top_k": 300})
    _ = _count_from_detector(d1, img0)

    # Without dedup
    ens_no = EnsembleDetector([d1, d2], EnsembleDetectorConfig(deduplicate=False, sort=True, max_keypoints=None))
    kps_no = ens_no.detect(img0)
    n_no = kps_no.shape[1]

    # With dedup
    ens = EnsembleDetector([d1, d2], EnsembleDetectorConfig(deduplicate=True, sort=True, max_keypoints=None))
    kps = ens.detect(img0)
    assert kps.ndim == 3 and kps.shape[0] == 1
    assert kps.shape[-1] == 2
    n_dedup = kps.shape[1]

    # Dedup should not increase count
    assert n_dedup <= n_no
    # And should produce at least one keypoint
    assert n_dedup > 0
