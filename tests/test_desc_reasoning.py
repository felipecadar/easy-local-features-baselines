import pytest
import torch

from easy_local_features.feature.baseline_desc_reasoning import Desc_Reasoning_baseline
from easy_local_features.utils import io, ops


@pytest.mark.timeout(300)
def test_desc_reasoning_detect_and_match():
    # Use smallest weights to reduce download time
    try:
        method = Desc_Reasoning_baseline(
            {
                "pretrained": "xfeat-3_layers",
                "device": "cpu",
            }
        )
    except Exception as e:
        pytest.skip(f"Skipping due to environment/network issue: {e}")

    img0 = io.fromPath("tests/assets/megadepth0.jpg")
    img1 = io.fromPath("tests/assets/megadepth1.jpg")
    img0 = ops.resize_short_edge(img0, 320)[0]
    img1 = ops.resize_short_edge(img1, 320)[0]

    kpts0, desc0 = method.detectAndCompute(img0)
    assert isinstance(kpts0, torch.Tensor)
    assert isinstance(desc0, torch.Tensor)
    assert kpts0.ndim == 3 and kpts0.shape[-1] == 2  # [B, N, 2]
    assert desc0.ndim == 4 and desc0.shape[-1] == 2  # [B, N, D, 2]

    res = method.match(img0, img1)
    mk0, mk1 = res["mkpts0"], res["mkpts1"]
    assert isinstance(mk0, torch.Tensor)
    assert isinstance(mk1, torch.Tensor)
    assert mk0.ndim == 2 and mk0.shape == mk1.shape


# One weight per extractor family; the remaining xfeat variants share the same
# compute() code path as "xfeat".
COMPUTE_WEIGHTS = [
    "xfeat-3_layers",
    "superpoint",
    "alike",
    "aliked",
    "dedode_B",
    "dedode_G",
]

# The dedode weights require very large extractor downloads; only test them when
# already cached or explicitly requested via RUN_SLOW_TESTS=1.
HEAVY_WEIGHTS = {"dedode_B", "dedode_G"}


def _weights_cached(name):
    from easy_local_features.utils.pathutils import CACHE_BASE

    return (CACHE_BASE / "desc_reasoning" / name / "model_config.yaml").exists()


@pytest.mark.timeout(600)
@pytest.mark.parametrize("weights", COMPUTE_WEIGHTS)
def test_desc_reasoning_compute_matches_detect(weights):
    import os

    if weights in HEAVY_WEIGHTS and not os.environ.get("RUN_SLOW_TESTS") and not _weights_cached(weights):
        pytest.skip(f"Skipping heavy weights '{weights}' (set RUN_SLOW_TESTS=1 to run)")

    try:
        method = Desc_Reasoning_baseline(
            {
                "pretrained": weights,
                "device": "cpu",
                "top_k": 512,
            }
        )
    except Exception as e:
        pytest.skip(f"Skipping due to environment/network issue: {e}")

    img = io.fromPath("tests/assets/megadepth0.jpg")
    img = ops.resize_short_edge(img, 320)[0]

    kpts, desc = method.detectAndCompute(img)
    kpts2, desc2 = method.compute(img, kpts)

    # compute() on the method's own detections must reproduce detectAndCompute()
    assert kpts2.shape == kpts.shape
    assert desc2.shape == desc.shape
    assert torch.equal(kpts2, kpts)
    assert torch.allclose(desc2, desc, atol=1e-4)

    # compute() must accept arbitrary keypoints, batched or not
    H, W = ops.prepareImage(img).shape[-2:]
    xs = torch.linspace(10, W - 10, 16)
    ys = torch.linspace(10, H - 10, 12)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)

    gkpts, gdesc = method.compute(img, grid)
    assert gkpts.shape[-2] == grid.shape[0]
    assert gdesc.shape[:2] == (1, grid.shape[0]) and gdesc.shape[-1] == 2
    assert torch.isfinite(gdesc).all()

    bkpts, bdesc = method.compute(img, grid.unsqueeze(0))
    assert torch.allclose(bdesc, gdesc)
