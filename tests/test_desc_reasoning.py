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
