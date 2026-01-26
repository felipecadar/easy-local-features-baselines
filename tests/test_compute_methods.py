
import pytest
import torch
import numpy as np
import os
from easy_local_features.utils import io
from easy_local_features.feature.baseline_disk import DISK_baseline
from easy_local_features.feature.baseline_dalf import DALF_baseline
from easy_local_features.feature.baseline_dedode import DeDoDe_baseline

# Use a real image from assets
ASSET_PATH = "tests/assets/megadepth0.jpg"

@pytest.fixture
def image():
    if not os.path.exists(ASSET_PATH):
        pytest.skip(f"Asset file {ASSET_PATH} not found")
    img = io.fromPath(ASSET_PATH)
    
    # Resize so max(h, w) == 512
    h, w = img.shape[-2:]
    scale = 512 / max(h, w)
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        img = torch.nn.functional.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
    return img

def test_disk_compute(image):
    extractor = DISK_baseline(conf={"top_k": 100})
    
    # 1. detectAndCompute
    kpts, descs = extractor.detectAndCompute(image)
    
    # 2. compute with tensor keypoints
    descs_computed = extractor.compute(image, kpts)
    
    assert descs_computed.shape == descs.shape
    assert torch.allclose(descs, descs_computed, atol=1e-4, rtol=1e-3)
    
    # 3. compute with numpy keypoints
    kpts_np = kpts.cpu().numpy()
    descs_computed_np = extractor.compute(image, kpts_np)
    assert torch.allclose(descs_computed, descs_computed_np, atol=1e-6)

def test_dalf_compute(image):
    # DALF can be heavy, limit top_k
    extractor = DALF_baseline(conf={"top_k": 100, "weights": "default"})
    
    # 1. detectAndCompute
    kpts, descs = extractor.detectAndCompute(image)
    
    # 2. compute
    descs_computed = extractor.compute(image, kpts)
    
    assert descs_computed.shape == descs.shape
    assert torch.allclose(descs, descs_computed, atol=1e-4, rtol=1e-3)

def test_dedode_compute(image):
    extractor = DeDoDe_baseline(conf={"top_k": 100})
    
    # 1. detectAndCompute
    kpts, descs = extractor.detectAndCompute(image)
    
    # 2. compute
    descs_computed = extractor.compute(image, kpts)
    
    assert descs_computed.shape == descs.shape
    assert torch.allclose(descs, descs_computed, atol=1e-4, rtol=1e-3)

