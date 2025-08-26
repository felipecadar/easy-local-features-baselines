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
from pathlib import Path
import tempfile
import torch
import torch.nn.functional as F

from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from easy_local_features.feature.basemodel import BaseExtractor, MethodType
from easy_local_features.utils import ops
from easy_local_features.utils.download import getCache


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

        # Try to prepare a local weights path (auto-download for supported variants).
        self._maybe_prepare_weights()

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
                    try:
                        self.model = torch.hub.load(
                            "facebookresearch/dinov3", conf.weights, force_reload=True
                        )
                    except Exception as e:
                        raise RuntimeError(
                            (
                                "Failed to load DINOv3 model. If you're using a variant without an auto-downloadable "
                                "GitHub weight, please either: (1) provide 'weights_path' to a local .pth file, or (2) "
                                "let torch.hub download the official weights by ensuring internet access.\n"
                                "For manual download instructions see: https://github.com/facebookresearch/dinov3\n"
                                f"Details: {e}"
                            )
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

    def denseCompute(self, img):
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

        return {
            "global_descriptor": cls_token,
            "feature_map": desc,
        }


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

    # ---- Internal helpers for weight resolution and auto-download ----
    @staticmethod
    def _github_release_base_url():
        # All assets hosted under the repo release tag 'dinov3'
        return (
            "https://github.com/felipecadar/easy-local-features-baselines/releases/download/dinov3/"
        )

    @staticmethod
    def _github_variant_to_filename():
        # Variants we auto-download from the project's GitHub release.
        # Keys are the torch.hub model names used in conf['weights'].
        return {
            # ViT family
            "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
            "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
            # Satellite ViT weights
            "dinov3_sat_vitl16": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
            "dinov3_sat_vit7b16": "dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
            # ConvNeXt family (note: compute() currently only supports ViT backbones)
            "dinov3_convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
            "dinov3_convnext_small": "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
            "dinov3_convnext_base": "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
            "dinov3_convnext_large": "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
        }

    def _maybe_prepare_weights(self):
        """If no weights_path is provided, try to auto-resolve/download from our GitHub release
        for supported variants. Files are saved under a temporary folder on this system.
        """
        conf = self.conf
        if conf.get("weights_path"):
            return  # user provided explicit path

        variant = conf.get("weights")
        gh_map = self._github_variant_to_filename()
        if variant not in gh_map:
            # Not one of our GitHub-hosted assets; we'll rely on torch.hub (Facebook) below.
            return

        filename = gh_map[variant]
        # Save under system temporary directory
        tmp_base = Path(getCache("dinov3")) 
        weights_dir = tmp_base
        weights_dir.mkdir(parents=True, exist_ok=True)
        target_path = weights_dir / filename

        if not target_path.exists():
            url = self._github_release_base_url() + filename
            try:
                print(f"Auto-downloading DINOv3 weights for '{variant}' from: {url}")
                torch.hub.download_url_to_file(url, str(target_path))
            except Exception as e:
                raise FileNotFoundError(
                    (
                        f"Couldn't download weights for '{variant}' to {target_path}.\n"
                        f"Tried URL: {url}\n"
                        "You may: (1) provide 'weights_path' pointing to a local .pth file, or (2) use the official\n"
                        "Facebook weights via torch.hub (internet required). See:\n"
                        "https://github.com/facebookresearch/dinov3\n"
                        f"Details: {e}"
                    )
                )

        # Use the local file we ensured above
        conf["weights_path"] = str(target_path)


if __name__ == "__main__":
    # Minimal smoke test: use SuperPoint to detect and DINOv3 to compute descriptors.
    from easy_local_features.utils import io, vis, ops
    from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline

    # Load an image from the test assets
    img0 = io.fromPath("tests/assets/megadepth0.jpg")
    img1 = io.fromPath("tests/assets/megadepth1.jpg")
    
    # resize for faster testing
    img0, _ = ops.resize_short_edge(img0, 512)
    img1, _ = ops.resize_short_edge(img1, 512)
    
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
            "weights": "dinov3_vitl16",
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
    