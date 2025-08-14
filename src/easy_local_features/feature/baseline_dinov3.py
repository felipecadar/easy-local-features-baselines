"""
DINOv3 baseline (descriptor-only), following DINOv2 patterns and official DINOv3 usage:
- Requires Python >= 3.10 (hubconf uses PEP 604 | unions)
- Load via torch.hub (recommended: local repo + explicit weights)
- Use ImageNet normalization for LVD-1689M weights and SAT normalization for SAT-493M

Loading examples (from DINOv3 README):

    import torch
    REPO_DIR = "/path/to/local/dinov3"
    m = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights='<URL_OR_PATH>')
    # Or remote hub (auto-download default weights):
    m = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16')

This extractor samples dense patch descriptors with get_intermediate_layers(reshape=True),
then bilinearly interpolates at provided keypoints.
"""

import sys
import torch
import torch.nn.functional as F

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor, MethodType
from ..utils import ops


class DINOv3_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DESCRIPTOR_ONLY
    available_weights = [
        # ViT on web images
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
        "dinov3_vith16plus",
        "dinov3_vit7b16",
        # ConvNeXt on web images
        "dinov3_convnext_tiny",
        "dinov3_convnext_small",
        "dinov3_convnext_base",
        "dinov3_convnext_large",
        # Satellite imagery variants (if available)
        "dinov3_sat_vitl16",
        "dinov3_sat_vit7b16",
    ]

    default_conf = {
        "weights": "dinov3_vits16",  # default ViT-S/16
        "allow_resize": True,  # resize to multiples of patch size
        # Optional advanced loading options:
        # - If you cloned the repo locally and have checkpoints, set repo_dir and weights_path.
        # - Otherwise we'll try to fetch via torch.hub (facebookresearch/dinov3) if available.
        "repo_dir": None,
        "weights_path": None,
        "source": "hub",  # "hub" or "local"
        # Normalization: 'auto' (infer from weights), 'imagenet', 'sat', or 'none'
        "normalize": "auto",
    }

    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.device = torch.device("cpu")
        self.matcher = NearestNeighborMatcher()

        # Upstream DINOv3 hubconf uses PEP 604 unions (e.g., float | None), which
        # require Python >= 3.10 without from __future__ import annotations.
        if sys.version_info < (3, 10):
            raise RuntimeError(
                "DINOv3 requires Python >= 3.10 due to upstream type hints (PEP 604). "
                "Please run with Python 3.10+ (e.g., uv python 3.11) or provide a patched local repo "
                "and set conf={'source': 'local', 'repo_dir': <path>, 'weights_path': <ckpt>}"
            )

        # Patch/stride size heuristic based on ViT-*/16 variants; used for grid sampling.
        self.vit_size = 16 if "16" in conf.weights else 14

        # Try to load the model. Prefer hub unless explicitly told to use local.
        self.model = None
        if conf.get("source") == "local" and conf.get("repo_dir"):
            try:
                self.model = torch.hub.load(
                    conf.repo_dir,
                    conf.weights,
                    source="local",
                    weights=conf.get("weights_path"),
                )
            except Exception:
                self.model = None

        if self.model is None:
            try:
                # Try normal hub first
                if conf.get("weights_path"):
                    self.model = torch.hub.load(
                        "facebookresearch/dinov3", conf.weights, weights=conf.get("weights_path")
                    )
                else:
                    self.model = torch.hub.load("facebookresearch/dinov3", conf.weights)
            except Exception:
                # Last-resort: force reload (helps if hub cache is stale)
                if conf.get("weights_path"):
                    self.model = torch.hub.load(
                        "facebookresearch/dinov3", conf.weights, force_reload=True, weights=conf.get("weights_path")
                    )
                else:
                    self.model = torch.hub.load(
                        "facebookresearch/dinov3", conf.weights, force_reload=True
                    )

        self.model.eval()

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
        """Apply per-channel normalization in-place style.
        mode: 'imagenet' | 'sat' | 'none'
        """
        if mode == "none":
            return img
        if mode == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
            return (img - mean) / std
        if mode == "sat":
            mean = torch.tensor([0.430, 0.411, 0.296], device=img.device).view(1, 3, 1, 1)
            std = torch.tensor([0.213, 0.156, 0.143], device=img.device).view(1, 3, 1, 1)
            return (img - mean) / std
        return img

    def _infer_norm_mode(self) -> str:
        # If explicit, honor it.
        mode = str(self.conf.get("normalize", "auto")).lower()
        if mode in {"imagenet", "sat", "none"}:
            return mode
        # Auto: inspect weights_path for 'sat493m'
        w = self.conf.get("weights_path")
        if isinstance(w, str) and "sat493m" in w.lower():
            return "sat"
        return "imagenet"

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

        # Apply normalization per DINOv3 guidance
        norm_mode = self._infer_norm_mode()
        img = self._apply_normalization(img, norm_mode)

        if self.conf.allow_resize:
            # ensure divisible by patch size for ViT
            patch = self.vit_size
            img = F.interpolate(img, [int(x // patch * patch) for x in img.shape[-2:]])

        # Only ViT-style backbones expose get_intermediate_layers; guard for ConvNeXt
        if not hasattr(self.model, "get_intermediate_layers"):
            raise NotImplementedError(
                "DINOv3_baseline currently supports ViT backbones (dinov3_vit*)."
            )

        # DINOv3 ViT exposes get_intermediate_layers similar to DINOv2
        desc, cls_token = self.model.get_intermediate_layers(
            img, n=1, return_class_token=True, reshape=True
        )[0]

        if keypoints is None:
            raise ValueError("DINOv3_baseline.compute expects keypoints (descriptor-only method).")

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
    # Minimal smoke test: use SuperPoint to detect and DINOv3 to compute descriptors.
    from easy_local_features.utils import io, vis, ops
    from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline

    # Load an image from the test assets
    img0 = io.fromPath("tests/assets/megadepth0.jpg")
    img1 = io.fromPath("tests/assets/megadepth1.jpg")
    
    # resize for faster testing
    img0, _ = ops.resize_short_edge(img0, 320)
    img1, _ = ops.resize_short_edge(img1, 320)
    
    # Create DINOv3 descriptor and SuperPoint detector
    # Tip: for official weights, prefer local repo + weights path per README.
    # Example:
    # method = DINOv3_baseline({
    #     "source": "local",
    #     "repo_dir": "/path/to/local/dinov3",
    #     "weights": "dinov3_vits16",
    #     "weights_path": "/path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    # })
    # For convenience we fall back to remote hub here:
    method = DINOv3_baseline(
        {
            "weights": "dinov3_vits16",
            "weights_path": "dino_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        }
    )
    detector = SuperPoint_baseline(
        {
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": 1024,
        }
    )
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
    print("DINOv3 smoke test passed.")


    matches = method.match(img0, img1)
    vis.plot_pair(
        img0, img1,
        title="DINOv3 Matches",
    )
    vis.plot_matches(
        matches["mkpts0"],
        matches["mkpts1"],
    )
    vis.show()
    