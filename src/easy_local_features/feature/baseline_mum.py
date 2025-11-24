"""
MuM baseline (descriptor-only), following the official MuM usage.
- Loads pretrained MuM ViT-Large/16 model
- Extracts normalized patch tokens as dense descriptors
- Bilinearly interpolates at provided keypoints
"""

import torch
import torch.nn.functional as F
from typing import TypedDict

from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from easy_local_features.feature.basemodel import BaseExtractor, MethodType
from easy_local_features.utils import ops
from easy_local_features.submodules.git_mum import mum_vitl16


class MuMConfig(TypedDict):
    pretrained: bool
    resize_size: tuple[int, int]


class MuM_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DESCRIPTOR_ONLY

    default_conf: MuMConfig = {
        "pretrained": True,
        "resize_size": (256, 256),  # MuM expects 256x256, divisible by 16
    }

    def __init__(self, conf: MuMConfig = {}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.device = torch.device("cpu")
        self.matcher = NearestNeighborMatcher()

        # Load MuM model
        self.model = mum_vitl16(pretrained=conf.pretrained)
        self.model.eval()

        # MuM specifics
        self.patch_size = 16
        self.embed_dim = 1024

    def sample_features(self, keypoints, features, s=None):
        """Sample features at keypoints using bilinear interpolation.
        
        Args:
            keypoints: [B, N, 2] or [N, 2]
            features: [B, C, H, W]
            s: patch size (unused, kept for compatibility)
        """
        if s is None:
            s = self.patch_size
        b, c, h, w = features.shape
        # Ensure keypoints are on the same device and dtype as features
        keypoints = keypoints.to(device=features.device, dtype=features.dtype)
        # Normalize keypoints to [-1, 1] for grid_sample
        keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
        keypoints = keypoints * 2 - 1
        # grid_sample expects [B, H, W, 2] but we have [B, N, 2], so view as [B, 1, N, 2]
        sampled = F.grid_sample(
            features, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
        )
        sampled = F.normalize(sampled.reshape(b, c, -1), p=2, dim=1)
        sampled = sampled.permute(0, 2, 1)
        return sampled

    def denseCompute(self, img):
        """Extract dense patch features from image."""
        # Prepare image tensor (0-1, BCHW)
        img = ops.prepareImage(img, gray=False, batch=True, imagenet=False).to(self.device)

        # Enforce 3-channel
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)

        # Resize to MuM's expected size (256x256)
        resize_size = tuple(self.conf.resize_size)
        img = F.interpolate(img, size=resize_size, mode="bilinear", align_corners=False)

        # Apply ImageNet normalization (same as transform_image)
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img = (img - mean) / std

        # Extract features
        with torch.no_grad():
            feats_dict = self.model.forward_features(img)
            patch_tokens = feats_dict['x_norm_patchtokens']  # [B, L, C]

        # Reshape to feature map [B, C, H, W]
        b, L, c = patch_tokens.shape
        h = w = int(L ** 0.5)  # Assume square grid
        feature_map = patch_tokens.permute(0, 2, 1).view(b, c, h, w)

        return {
            "global_descriptor": feats_dict['x_norm_cls_token'],  # [B, C]
            "feature_map": feature_map,  # [B, C, H, W]
        }

    def detectAndCompute(self, img, return_dict=None):
        raise NotImplementedError("MuM_baseline is descriptor-only; use compute() with external keypoints.")

    def detect(self, img, op=None):
        raise NotImplementedError("MuM_baseline is descriptor-only; use an external detector.")

    def compute(self, img, keypoints=None, return_dict=False):
        """Compute descriptors for provided keypoints."""
        if keypoints is None:
            raise ValueError("MuM_baseline.compute expects keypoints (descriptor-only method).")

        # Prepare image tensor to get original size
        img_tensor = ops.prepareImage(img, gray=False, batch=True, imagenet=False).to(self.device)
        original_h, original_w = img_tensor.shape[-2:]

        # Get dense features
        dense_result = self.denseCompute(img)
        feature_map = dense_result["feature_map"]

        # Scale keypoints to resized image coordinates for sampling
        resize_h, resize_w = self.conf.resize_size
        scale_h = resize_h / original_h
        scale_w = resize_w / original_w
        scaled_keypoints = keypoints * torch.tensor([scale_w, scale_h], device=keypoints.device, dtype=keypoints.dtype)

        # Sample at scaled keypoints
        descriptors = self.sample_features(scaled_keypoints, feature_map, s=self.patch_size)

        if return_dict:
            return {
                "descriptors": descriptors,
                "keypoints": keypoints,  # Return original keypoints
                "global_descriptor": dense_result["global_descriptor"],
                "feature_map": feature_map,
            }
        return keypoints, descriptors  # Return original keypoints

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    @property
    def has_detector(self):
        return False


if __name__ == "__main__":
    # Minimal smoke test: use SuperPoint to detect and MuM to compute descriptors.
    from easy_local_features.utils import io, vis, ops
    from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline

    # Load test images
    img0 = io.fromPath("tests/assets/megadepth0.jpg")
    img1 = io.fromPath("tests/assets/megadepth1.jpg")
    
    # Resize for faster testing
    img0, _ = ops.resize_short_edge(img0, 512)
    img1, _ = ops.resize_short_edge(img1, 512)
    
    # Create MuM descriptor and SuperPoint detector
    method = MuM_baseline({"pretrained": True})
    detector = SuperPoint_baseline({
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": 1024,
    })
    method.addDetector(detector)

    # Run detect-only, compute-only, and detect+compute for consistency
    kpts = method.detect(img0)
    kpts2, desc2 = method.detectAndCompute(img0)
    kpts3, desc3 = method.compute(img0, kpts)

    print("kpts:", kpts.shape)
    print("desc (via detectAndCompute):", desc2.shape)
    print("desc (via compute):", desc3.shape)

    # Basic sanity checks
    assert kpts.shape == kpts2.shape
    assert desc2.shape == desc3.shape
    print("MuM smoke test passed.")

    # Test matching
    matches = method.match(img0, img1)
    vis.plot_pair(img0, img1, title="MuM Matches")
    vis.plot_matches(matches["mkpts0"], matches["mkpts1"])
    vis.show()
