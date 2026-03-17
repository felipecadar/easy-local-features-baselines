"""
DINOv2 baseline (descriptor-only), following project conventions:
- Load via torch.hub (facebookresearch/dinov2)
- Use ImageNet normalization
- Dense patch descriptors via get_intermediate_layers(reshape=True),
  bilinearly interpolated at provided keypoints.
"""

import torch
import torch.nn.functional as F

from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from easy_local_features.feature.basemodel import BaseExtractor, MethodType
from easy_local_features.utils import ops
from typing import TypedDict, Optional


class DINOv2Config(TypedDict):
    weights: str
    allow_resize: bool
    normalize: str


class DINOv2_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DESCRIPTOR_ONLY
    available_weights = [
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
        "dinov2_vits14_reg",
        "dinov2_vitb14_reg",
        "dinov2_vitl14_reg",
        "dinov2_vitg14_reg",
    ]

    default_conf: DINOv2Config = {
        "weights": "dinov2_vits14",
        "allow_resize": True,
        # Normalization: 'imagenet' or 'none'
        "normalize": "imagenet",
    }

    # Map model variant to embedding dimension
    _dim_map = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
        "dinov2_vits14_reg": 384,
        "dinov2_vitb14_reg": 768,
        "dinov2_vitl14_reg": 1024,
        "dinov2_vitg14_reg": 1536,
    }

    def __init__(self, conf: DINOv2Config = {}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.device = torch.device("cpu")
        self.matcher = NearestNeighborMatcher()

        self.vit_size = 14  # DINOv2 uses patch size 14

        try:
            self.model = torch.hub.load("facebookresearch/dinov2", conf.weights)
        except Exception:
            self.model = torch.hub.load("facebookresearch/dinov2", conf.weights, force_reload=True)

        self.model.eval()

    def infer_dim(self) -> int:
        """Return the descriptor dimensionality for the current model variant."""
        return self._dim_map.get(self.conf.weights, self.model.embed_dim)

    def sample_features(self, keypoints, features, s=None, mode="bilinear"):
        if s is None:
            s = self.vit_size
        b, c, h, w = features.shape
        # Ensure keypoints are on the same device and dtype as features for grid_sample
        keypoints = keypoints.to(device=features.device, dtype=features.dtype)
        keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        sampled = torch.nn.functional.grid_sample(
            features, keypoints.view(b, 1, -1, 2), mode=mode, align_corners=False
        )
        sampled = torch.nn.functional.normalize(sampled.reshape(b, c, -1), p=2, dim=1)
        sampled = sampled.permute(0, 2, 1)
        return sampled

    @staticmethod
    def _apply_normalization(img: torch.Tensor, mode: str):
        """Apply per-channel normalization.
        mode: 'imagenet' | 'none'
        """
        if mode == "none":
            return img
        if mode == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
            return (img - mean) / std
        return img

    def denseCompute(self, img):
        # Prepare image tensor (0-1, BCHW)
        img = ops.prepareImage(img, gray=False, batch=True, imagenet=False).to(self.device)

        # Enforce 3-channel for normalization expectations
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)

        # Apply normalization
        norm_mode = str(self.conf.get("normalize", "imagenet")).lower()
        img = self._apply_normalization(img, norm_mode)

        if self.conf.allow_resize:
            patch = self.vit_size
            img = F.interpolate(img, [int(x // patch * patch) for x in img.shape[-2:]])

        desc, cls_token = self.model.get_intermediate_layers(
            img, n=1, return_class_token=True, reshape=True
        )[0]

        return {
            "global_descriptor": cls_token,
            "feature_map": desc,
        }

    def detectAndCompute(self, img, return_dict=None):
        raise NotImplementedError

    def detect(self, img, op=None):
        raise NotImplementedError

    def compute(self, img, keypoints=None, return_dict=False):
        # Prepare image tensor (0-1, BCHW)
        img = ops.prepareImage(img, gray=False, batch=True, imagenet=False).to(self.device)

        # Enforce 3-channel for normalization expectations
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)

        # Apply normalization
        norm_mode = str(self.conf.get("normalize", "imagenet")).lower()
        img = self._apply_normalization(img, norm_mode)

        if self.conf.allow_resize:
            patch = self.vit_size
            img = F.interpolate(img, [int(x // patch * patch) for x in img.shape[-2:]])

        desc, cls_token = self.model.get_intermediate_layers(
            img, n=1, return_class_token=True, reshape=True
        )[0]

        if keypoints is None:
            raise ValueError("DINOv2_baseline.compute expects keypoints (descriptor-only method).")

        descriptors = self.sample_features(keypoints, desc, s=self.vit_size)
        if return_dict:
            return {
                "descriptors": descriptors,
                "keypoints": keypoints,
                "global_descriptor": cls_token,
                "feature_map": desc,
            }
        return keypoints, descriptors

    def to(self, device):
        self.model.to(device)
        self.device = device

    @property
    def has_detector(self):
        return False


if __name__ == "__main__":
    from easy_local_features.utils import io, vis, ops
    from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline

    img0 = io.fromPath("tests/assets/megadepth0.jpg")
    img1 = io.fromPath("tests/assets/megadepth1.jpg")

    img0, _ = ops.resize_short_edge(img0, 512)
    img1, _ = ops.resize_short_edge(img1, 512)

    method = DINOv2_baseline({"weights": "dinov2_vitl14"})
    detector = SuperPoint_baseline(
        {
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": 1024,
        }
    )
    method.addDetector(detector)

    print(f"Descriptor dim: {method.infer_dim()}")

    kpts = method.detect(img0)
    kpts2, desc2 = method.detectAndCompute(img0)
    kpts3, desc3 = method.compute(img0, kpts)

    print("kpts:", kpts.shape)
    print("desc (via detectAndCompute):", desc2.shape)
    print("desc (via compute):", desc3.shape)

    assert kpts.shape == kpts2.shape
    assert desc2.shape == desc3.shape
    print("DINOv2 smoke test passed.")

    matches = method.match(img0, img1)
    vis.plot_pair(img0, img1, title="DINOv2 Matches")
    vis.plot_matches(matches["mkpts0"], matches["mkpts1"])
    vis.show()